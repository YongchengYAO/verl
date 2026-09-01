# Copyright 2025 Individual Contributor: Yongcheng Yao
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Text parsing shared by the MedVision rewards: tags, number/coordinate patterns, the final
``<answer>`` block, the per-step ``<step-k-answer>`` blocks, and the soft reasoning-structure score.

Coordinates are relative image positions in [0, 1] written as ``(x, y)`` in ``(w, h)`` order (the
MedVision parquet builders perform that conversion).
"""

import re

FLAGS = re.DOTALL  # tags are case-sensitive; values may span lines

NON_NEG_REAL = r"\d+(?:\.\d+)?"
NON_NEG_REAL_GROUP = rf"({NON_NEG_REAL})"
RELATIVE_POSITION = r"(?:0(?:\.\d+)?|1(?:\.0+)?)"  # [0, 1] inclusive
RELATIVE_POSITION_GROUP = rf"({RELATIVE_POSITION})"
COORD = rf"\(\s*{RELATIVE_POSITION}\s*,\s*{RELATIVE_POSITION}\s*\)"
COORD_GROUP = rf"\(\s*{RELATIVE_POSITION_GROUP}\s*,\s*{RELATIVE_POSITION_GROUP}\s*\)"


def tag(name):
    return rf"<{name}>"


def end(name):
    return rf"</{name}>"


# --------------------------------------------------------------------------------------------
# Final answer
# --------------------------------------------------------------------------------------------
def answer_patterns(num_values):
    """
    Regexes for a well-formed final answer: ``num_values`` non-negative decimals separated by
    commas inside ``<answer> </answer>``, optionally wrapped in ``( )`` or ``[ ]``.

    Returns:
        (check_pattern, group_pattern): the second captures the values.
    """
    values = r"\s*,\s*".join([NON_NEG_REAL] * num_values)
    values_group = r"\s*,\s*".join([NON_NEG_REAL_GROUP] * num_values)

    def wrap(v):
        return rf"{tag('answer')}\s*[\(\[]?\s*{v}\s*[\)\]]?\s*{end('answer')}"

    return wrap(values), wrap(values_group)


def match_answer(content, num_values):
    """Binary answer-format check: 1 if the ``<answer>`` block is well formed, else 0."""
    return 1 if re.search(answer_patterns(num_values)[0], content, FLAGS) else 0


def parse_ground_truth(ground_truth):
    """Numbers of a ground-truth string such as ``"12.5, 7.9"`` or ``"(0.1, 0.2, 0.5, 0.6)"``."""
    parts = re.sub(r"[()\[\]]", "", str(ground_truth)).split(",")
    return [float(p) for p in parts if p.strip()]


def parse_answer(solution, num_values):
    """
    Values of the final answer: the strict pattern first, otherwise the last ``num_values``
    numbers inside ``<answer> </answer>``. None if the answer cannot be parsed.
    """
    m = re.search(answer_patterns(num_values)[1], solution, FLAGS)
    if m:
        return [float(g) for g in m.groups()]
    m = re.search(rf"{tag('answer')}(.*?){end('answer')}", solution, FLAGS)
    if not m:
        return None
    nums = re.findall(r"-?\d+\.?\d*", m.group(1))
    if len(nums) < num_values:
        return None
    try:
        return [float(x) for x in nums[-num_values:]]
    except ValueError:
        return None


# --------------------------------------------------------------------------------------------
# CoT steps
# --------------------------------------------------------------------------------------------
def step_reasoning(k):
    return rf"{tag(f'step-{k}-reasoning')}.*?{end(f'step-{k}-reasoning')}"


def step_answer(k, n_points, group=False):
    """``<step-k-answer>`` holding ``n_points`` coordinates ``(x, y), ...`` or, for 0, one scalar."""
    if n_points == 0:
        value = NON_NEG_REAL_GROUP if group else NON_NEG_REAL
    else:
        value = r"\s*,\s*".join([COORD_GROUP if group else COORD] * n_points)
    return rf"{tag(f'step-{k}-answer')}.*?{value}.*?{end(f'step-{k}-answer')}"


def parse_step(solution, k, n_points):
    """
    Numbers declared in step ``k`` (``<step-k-reasoning>`` followed by ``<step-k-answer>``):
    ``2 * n_points`` coordinates, or one scalar for ``n_points == 0``. None if absent/malformed.
    """
    m = re.search(rf"{step_reasoning(k)}\s*.*?{step_answer(k, n_points, group=True)}", solution, FLAGS)
    return [float(g) for g in m.groups()] if m else None


# --------------------------------------------------------------------------------------------
# Soft reasoning-structure score
# --------------------------------------------------------------------------------------------
def soft_format_score(content, points_per_step):
    """
    Soft (partial-credit) score in [0, 1] for the ``<think>`` structure of a CoT response.

    ``points_per_step[k-1]`` is the number of coordinates expected in ``<step-k-answer>``
    (0 for a scalar measurement step). Credit is granted for: the ``<think>`` block (required:
    without it the score is 0), each step's reasoning tag, each step's well-formed answer tag,
    the numbers found for each step (proportional, up to the expected count), and the step
    order (full credit when every step is present in order, half when all but one are).
    """
    n_steps = len(points_per_step)
    w_think, w_order = 1, 2
    weights = [(1, 1, 2 if n else 1) for n in points_per_step]  # reasoning, answer tag, values
    max_score = w_think + w_order + sum(sum(w) for w in weights)

    think = re.search(rf"{tag('think')}(.*?){end('think')}", content, FLAGS)
    if not think:
        return 0.0
    body = think.group(1)
    score = float(w_think)
    positions = []
    for k, (n_points, (w_reason, w_tag, w_values)) in enumerate(zip(points_per_step, weights), start=1):
        reasoning = re.search(step_reasoning(k), body, FLAGS)
        answer = re.search(step_answer(k, n_points), body, FLAGS)
        if reasoning:
            score += w_reason
        if answer:
            score += w_tag
        starts = [m.start() for m in (reasoning, answer) if m]
        positions.append(min(starts) if starts else None)

        if answer:
            region = answer.group(0)
        elif reasoning:  # partial structure right after the reasoning block
            region = body[reasoning.end() : reasoning.end() + (2000 if n_points else 1000)]
        else:
            region = None
        if region:
            n_found = len(re.findall(NON_NEG_REAL, region))
            if n_points:
                score += min(n_found, 2 * n_points) * w_values / (2 * n_points)
            elif n_found:
                score += w_values

    present = [p for p in positions if p is not None]
    if present == sorted(present):
        if len(present) == n_steps:
            score += w_order
        elif len(present) >= n_steps - 1:
            score += w_order * 0.5
    return max(0.0, min(1.0, score / max_score))
