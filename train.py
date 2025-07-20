"""
train.py — RL Agent Training for Uniswap V3 LP Strategies

Trains PPO and DQN agents using a synthetic Uniswap V3 environment. Trained models are saved to disk for later evaluation.

Author: Ricardo Arcifa
Affiliation: IEEE DAPPS 2025
"""

import os
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor

from utils.callbacks import CumulativeRewardLogger
from environments.train_env import UniswapV3TrainEnv

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

# Train PPO agent
train_env_ppo = DummyVecEnv([lambda: UniswapV3TrainEnv(enable_logging=False)])
train_env_ppo = VecMonitor(train_env_ppo)
train_env_ppo = VecNormalize(train_env_ppo, norm_reward=True)

model_ppo = PPO(
    "MlpPolicy",
    train_env_ppo,
    verbose=1,
    tensorboard_log="./tensorboard_logs_ppo"
)
model_ppo.learn(total_timesteps=100_000, callback=CumulativeRewardLogger())
model_ppo.save("models/uniswap_v3_ppo_model")

# Train DQN agent
train_env_dqn = DummyVecEnv([lambda: UniswapV3TrainEnv(enable_logging=False)])
train_env_dqn = VecMonitor(train_env_dqn)
train_env_dqn = VecNormalize(train_env_dqn, norm_reward=True)

model_dqn = DQN(
    "MlpPolicy",
    train_env_dqn,
    verbose=1,
    tensorboard_log="./tensorboard_logs_dqn"
)
model_dqn.learn(total_timesteps=100_000, callback=CumulativeRewardLogger())
model_dqn.save("models/uniswap_v3_dqn_model")

print("Training complete. Models saved.") 