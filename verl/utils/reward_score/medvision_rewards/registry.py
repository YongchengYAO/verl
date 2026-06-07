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

import matplotlib.pyplot as plt
import numpy as np

from verl.utils.reward_score.medvision_rewards.reward_fn import (
    exp_decay,
    gaussian_proxy,
)

REWARD_FUNCTIONS = {
    "Gaussian Proxy: exp(-x²/(2σ²)), σ²=0.5 (default)": gaussian_proxy,  # exp(-x²/(2σ²)), where σ²=0.5
    "Exponential Decay: exp(-kx), k=1 (default)": exp_decay,  # exp(-kx), where k=1
}


def plot_reward_functions() -> None:
    """
    Plots the available reward functions for visualization.
    """
    x = np.linspace(0, 10, 1000)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    plt.figure(figsize=(8, 6))

    for i, (name, func) in enumerate(REWARD_FUNCTIONS.items()):
        plt.plot(x, func(x), label=name, color=colors[i % len(colors)])

    plt.title("Reward Function")
    plt.xlabel("Error")
    plt.ylabel("Reward")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_reward_functions()
