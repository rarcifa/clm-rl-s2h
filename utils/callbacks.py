"""
cumulative_reward_logger.py — Custom Reward Tracking Callback for RL Training

This module defines a custom Stable-Baselines3 callback to track and log
cumulative rewards during training. It writes values to TensorBoard under
the tag 'custom/cumulative_rewards', allowing visual monitoring of
long-term return accumulation across episodes.

Author: Ricardo Arcifa  
Affiliation: IEEE DAPPS 2025
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CumulativeRewardLogger(BaseCallback):
    """
    Custom Stable-Baselines3 callback to log cumulative rewards.

    This callback accumulates all rewards observed during training and logs
    them at each step to TensorBoard. It is primarily used to monitor global
    return trends over the course of policy learning.

    Attributes:
        cumulative_reward (float): Total accumulated reward seen during training.
    """

    def __init__(self, verbose: int = 0):
        """
        Initialize the callback.

        Args:
            verbose (int): Verbosity level (0 = silent, 1 = info).
        """
        super().__init__(verbose)
        self.cumulative_reward = 0.0

    def _on_step(self) -> bool:
        """
        Called at each training step to update and log cumulative rewards.

        Returns:
            bool: True to continue training, False to stop.
        """
        rewards = self.locals["rewards"]  # Array of rewards from current step
        self.cumulative_reward += np.sum(rewards)
        self.logger.record("custom/cumulative_rewards", self.cumulative_reward)
        return True
