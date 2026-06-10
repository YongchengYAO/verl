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

"""
Temperature-based weighted sampling for multitask RL training.

Rebalances imbalanced multitask datasets (e.g. MedVision's 110K detection / 5.5K TL /
5.5K AD mix) by sampling tasks with probability proportional to count^(1/T):
T=1 reproduces the natural task proportions, larger T flattens them towards uniform.
Mirrors the math of TemperatureSamplerSFTTrainer in MedVision's sft_utils.py.

Enabled via config, e.g.:
    +data.temperature_sampler.enable=True \
    +data.temperature_sampler.T=5 \
    +data.temperature_sampler.task_key=ability \
    +data.temperature_sampler.task_group_map='medvision-angle:AD,medvision-distance:AD'
"""

from collections import Counter


def parse_task_group_map(spec):
    """
    Parses a 'label:group,label:group,...' string into a {label: group} dict.

    Labels not present in the map keep their own value as the group, so only
    labels that should be merged need to be listed.

    Args:
        spec: The mapping string (None/empty -> empty dict).

    Returns:
        A dict mapping raw task labels to group names.
    """
    group_map = {}
    if not spec:
        return group_map
    for item in str(spec).split(","):
        label, sep, group = item.strip().partition(":")
        if not sep or not label or not group:
            raise ValueError(f"task_group_map entries must be 'label:group', got {item!r}")
        group_map[label.strip()] = group.strip()
    return group_map


def compute_temperature_sample_weights(task_labels, temperature):
    """
    Computes per-sample weights so that task-level sampling probability is
    proportional to count(task)^(1/temperature).

    Per-sample weight for a sample of task i:
        weight_i = p(task_i) / count(task_i)
    which is uniform within a task and guarantees the task-level probabilities.

    Args:
        task_labels: One task label per sample.
        temperature: Sampling temperature T (> 0). T=1 -> proportional to counts
            (no rebalancing); larger T -> flatter task probabilities.

    Returns:
        A tuple (sample_weights, task_probs, task_counts).
    """
    if temperature <= 0:
        raise ValueError(f"temperature_sampler.T must be > 0, got {temperature}.")

    task_counts = Counter(task_labels)
    powered = {task: float(count) ** (1.0 / float(temperature)) for task, count in task_counts.items()}
    total = sum(powered.values())
    task_probs = {task: p / total for task, p in powered.items()}
    weight_per_task = {task: task_probs[task] / task_counts[task] for task in task_counts}
    sample_weights = [weight_per_task[label] for label in task_labels]
    return sample_weights, task_probs, dict(task_counts)


def create_temperature_sampler(data_config, dataset):
    """
    Builds a WeightedRandomSampler for temperature-based multitask rebalancing.

    Args:
        data_config: The data config; reads data_config.temperature_sampler
            (keys: T [required], task_key [default 'ability'], task_group_map)
            and data_config.seed for the sampling generator.
        dataset: The train dataset; must expose a `.dataframe` (HF datasets.Dataset)
            containing the task-label column (true for RLHFDataset and subclasses).

    Returns:
        A WeightedRandomSampler, or None if fewer than 2 distinct tasks are present
        (nothing to rebalance) so the caller can fall back to the default sampler.
    """
    import torch
    from torch.utils.data import WeightedRandomSampler

    cfg = data_config.get("temperature_sampler")
    temperature = cfg.get("T", None)
    if temperature is None:
        raise ValueError("temperature_sampler.T must be set when the temperature sampler is enabled.")
    task_key = cfg.get("task_key", "ability")
    group_map = parse_task_group_map(cfg.get("task_group_map", None))

    dataframe = getattr(dataset, "dataframe", None)
    if dataframe is None or task_key not in dataframe.column_names:
        raise ValueError(
            f"Temperature sampler requires a dataset with a '.dataframe' containing column '{task_key}'."
        )

    raw_labels = [str(label) for label in dataframe[task_key]]
    task_labels = [group_map.get(label, label) for label in raw_labels]
    sample_weights, task_probs, task_counts = compute_temperature_sample_weights(task_labels, float(temperature))

    if len(task_counts) <= 1:
        print("[Info] Temperature sampler enabled but only one task found; falling back to default sampling.")
        return None

    generator = torch.Generator()
    seed = data_config.get("seed")
    if seed is not None:
        generator.manual_seed(seed)

    print(f"[Info] Temperature sampler (T={temperature}) task counts: {task_counts}")
    print(f"[Info] Temperature sampler task probabilities: { {k: round(v, 6) for k, v in task_probs.items()} }")

    # replacement=True so minority-task samples can be drawn more often than their raw
    # cardinality within one epoch. WeightedRandomSampler is not stateful: on checkpoint
    # resume the StatefulDataLoader fast-forwards through a fresh draw, which is only
    # reproducible when data.seed is set.
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )
