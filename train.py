"""
train_models.py — Training PPO and DQN Agents for Uniswap V3 Liquidity Management

This script trains two reinforcement learning agents (PPO and DQN) on a synthetic
price and volume scenario generated using geometric Brownian motion (GBM).

Each agent is trained using the `UniswapV3TrainEnv` environment for 365 simulated days.

Outputs:
- Trained models saved to ./models/
- Training logs saved to ./results/train_logs/
- Synthetic scenario saved for reproducibility
- TensorBoard logs saved for analysis

References:
- Synthetic data: Eq. (4), Section IV-A (price simulation)
- PPO/DQN setup: Section IV-E (training methodology)
- Reward function: Eq. (9)

Author: Ricardo Arcifa  
Affiliation: IEEE DAPPS 2025
"""

import os
import numpy as np

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from envs.train_env import UniswapV3TrainEnv
from utils.scenario import generate_scenario_arrays

# Prepare folders
os.makedirs("results/train_logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Generate synthetic GBM scenario (365 steps, drift μ, volatility σ)
synthetic_scenario = generate_scenario_arrays(
    steps=365,
    initial_price=2200.0,
    mu=0.0001,
    sigma=0.06
)
shock_array, prices_eval, sqrt_prices_eval, volumes_eval = synthetic_scenario

# Save the synthetic scenario for reproducibility
np.savez("results/train_logs/synthetic_training_scenario.npz",
         shock=shock_array,
         price=prices_eval,
         sqrt_price=sqrt_prices_eval,
         volume=volumes_eval)

# PPO Training
train_env_ppo = UniswapV3TrainEnv(
    initial_investment=10000,
    reposition_gas_cost_range=(10, 50),
    protocol_fee_fraction=0.1,
    tick_spacing=10,
    steps=365,
    initial_price=2200.0,
    enable_logging=True,
    csv_filename="./results/train_logs/ppo_simulation_train.csv",
    mu=0.0001,
    sigma=0.06,
    shocks=shock_array,
    prices=prices_eval,
    sqrt_prices=sqrt_prices_eval,
    volumes=volumes_eval
)

# Wrap environment with monitoring and normalization
train_vec_ppo = DummyVecEnv([lambda: train_env_ppo])
train_vec_ppo = VecMonitor(train_vec_ppo, filename="./ppo_logs")
train_vec_ppo = VecNormalize(train_vec_ppo, norm_reward=True)

# PPO agent configuration
model_ppo = PPO(
    "MlpPolicy",
    env=train_vec_ppo,
    verbose=1,
    n_steps=4096,              # rollout buffer size
    batch_size=128,
    gamma=0.93,
    learning_rate=2.5e-4,
    ent_coef=0.05,
    vf_coef=0.9,
    clip_range=0.12,
    max_grad_norm=0.5,
    tensorboard_log="./tensorboard_logs_ppo/"
)

# Train the PPO agent
model_ppo.learn(total_timesteps=500_000)
model_ppo.save("./models/uniswap_v3_ppo_model")

# DQN Training
train_env_dqn = UniswapV3TrainEnv(
    initial_investment=10000,
    reposition_gas_cost_range=(10, 50),
    protocol_fee_fraction=0.1,
    tick_spacing=10,
    steps=365,
    initial_price=2200.0,
    enable_logging=True,
    csv_filename="./results/train_logs/dqn_simulation_train.csv",
    mu=0.0001,
    sigma=0.06
)

train_vec_dqn = DummyVecEnv([lambda: train_env_dqn])
train_vec_dqn = VecMonitor(train_vec_dqn, filename="./dqn_logs")
train_vec_dqn = VecNormalize(train_vec_dqn, norm_reward=True)

# DQN agent configuration
model_dqn = DQN(
    "MlpPolicy",
    env=train_vec_dqn,
    verbose=1,
    learning_rate=1e-4,
    buffer_size=1_000_000,
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
    tensorboard_log="./tensorboard_logs_dqn/"
)

# Train the DQN agent
model_dqn.learn(total_timesteps=500_000)
model_dqn.save("./models/uniswap_v3_dqn_model")

# Done
print("Training completed and models saved to ./models/")
