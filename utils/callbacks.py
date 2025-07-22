"""
cumulative_reward_logger.py — TensorBoard Logging for Cumulative Rewards

This module defines a custom `CumulativeRewardLogger` callback for use with
Stable-Baselines3 training. It logs the cumulative reward collected across all
environments during the current training run to TensorBoard.

This is useful for diagnosing whether total reward accumulation is progressing
steadily across steps or diverging.

Reference:
- Used in conjunction with PPO/DQN training for Uniswap v3 CLM agents
- Integrates with SB3’s logger interface

Author: Ricardo Arcifa  
Affiliation: IEEE DAPPS 2025
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CumulativeRewardLogger(BaseCallback):
    """
    Custom callback to log cumulative reward to TensorBoard during training.

    Logs a single scalar value per step:
        - "custom/cumulative_rewards": Running sum of environment rewards

    This helps visualize reward accumulation over time and detect stagnation
    or reward collapse in training curves.

    Attributes:
        cumulative_reward (float): Total reward accumulated across all envs and steps.
    """

    def __init__(self, verbose=0):
        """
        Initialize the logger callback.

        Args:
            verbose (int): Verbosity level (0 = silent, 1 = verbose).
        """
        super().__init__(verbose)
        self.cumulative_reward = 0.0

    def _on_step(self) -> bool:
        """
        Called at each environment step during training.

        Accumulates the rewards from the current step and logs them
        under the TensorBoard namespace "custom/cumulative_rewards".

        Returns:
            bool: True to continue training, False to stop.
        """
        rewards = self.locals["rewards"]  # Reward(s) from current environment step
        self.cumulative_reward += np.sum(rewards)
        self.logger.record("custom/cumulative_rewards", self.cumulative_reward)
        return True
