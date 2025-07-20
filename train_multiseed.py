"""
train_multiseed.py — Multi-Seed RL Agent Training for Uniswap V3 LP Strategies

Trains PPO and DQN agents using synthetic Uniswap V3 environments for multiple random seeds, as described in the paper. Each model is saved with the seed in the filename for later evaluation.

Author: Ricardo Arcifa
Affiliation: IEEE DAPPS 2025
"""

import os
import numpy as np
import random
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor

from utils.callbacks import CumulativeRewardLogger
from environments.train_env import UniswapV3TrainEnv

# Configuration
SEEDS = [0, 1, 2]
RESULTS_DIR = "results"
MODELS_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

for SEED in SEEDS:
    np.random.seed(SEED)
    random.seed(SEED)
    print(f"Training PPO and DQN for seed {SEED}...")

    # Train PPO
    train_env_ppo = DummyVecEnv([lambda: UniswapV3TrainEnv(enable_logging=False)])
    train_env_ppo = VecMonitor(train_env_ppo)
    train_env_ppo = VecNormalize(train_env_ppo, norm_reward=True)
    model_ppo = PPO(
        "MlpPolicy",
        train_env_ppo,
        verbose=1,
        n_steps=4096,
        batch_size=128,
        gamma=0.93,
        learning_rate=2.5e-4,
        ent_coef=0.05,
        vf_coef=0.9,
        clip_range=0.12,
        max_grad_norm=0.5,
        tensorboard_log=f"./tensorboard_logs_ppo/seed_{SEED}"
    )
    model_ppo.learn(total_timesteps=500_000, callback=CumulativeRewardLogger())
    model_ppo.save(f"models/uniswap_v3_ppo_model_seed{SEED}")

    # Train DQN
    train_env_dqn = DummyVecEnv([lambda: UniswapV3TrainEnv(enable_logging=False)])
    train_env_dqn = VecMonitor(train_env_dqn)
    train_env_dqn = VecNormalize(train_env_dqn, norm_reward=True)
    model_dqn = DQN(
        "MlpPolicy",
        train_env_dqn,
        verbose=1,
        learning_rate=1e-4,
        buffer_size=1000000,
        learning_starts=5000,
        batch_size=32,
        gamma=0.93,
        tau=0.8,
        target_update_interval=1000,
        train_freq=4,
        max_grad_norm=10,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,
        tensorboard_log=f"./tensorboard_logs_dqn/seed_{SEED}"
    )
    model_dqn.learn(total_timesteps=500_000, callback=CumulativeRewardLogger())
    model_dqn.save(f"models/uniswap_v3_dqn_model_seed{SEED}")

    print(f"Seed {SEED} training complete. Models saved.")

print("Multi-seed training complete. All models saved in models/ directory.") 