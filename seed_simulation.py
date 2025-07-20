"""
seed_simulation.py — Multi-Seed Evaluation for RL and Heuristic Strategies

Runs multi-seed evaluation for PPO, DQN, and heuristic strategies on Uniswap V3 LP scenarios. For each seed, a new scenario is generated, and all strategies are evaluated and logged.

Author: Ricardo Arcifa
Affiliation: IEEE DAPPS 2025
"""

import os
import numpy as np
import pandas as pd
import random
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environments.eval_env import UniswapV3EvalEnv
from strategies.heuristic import run_heuristic
from utils.core import generate_scenario_arrays

# Configuration
N_SEEDS = 50
STEPS = 365
INITIAL_PRICE = 2200.0
INITIAL_INVESTMENT = 10000
RESULTS_DIR = "results"
EVAL_LOGS_DIR = os.path.join(RESULTS_DIR, "eval_logs/seeds/")
MODELS_DIR = "models"
PPO_MODEL_PATH = os.path.join(MODELS_DIR, "uniswap_v3_ppo_model")
DQN_MODEL_PATH = os.path.join(MODELS_DIR, "uniswap_v3_dqn_model")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(EVAL_LOGS_DIR, exist_ok=True)

# Main Seed Loop
for SEED in range(N_SEEDS):
    np.random.seed(SEED)
    random.seed(SEED)

    # Generate scenario for this seed
    shocks_eval, prices_eval, sqrt_prices_eval, volumes_eval = generate_scenario_arrays(
        steps=STEPS,
        initial_price=INITIAL_PRICE,
        mu=0.0001,
        sigma=0.06
    )
    pool_liquidity_eval = np.random.uniform(1_000_000, 5_000_000, size=STEPS)

    # Evaluate PPO
    ppo_eval_file = os.path.join(EVAL_LOGS_DIR, f"ppo_seed{SEED}.csv")
    eval_env_ppo = DummyVecEnv([lambda: UniswapV3EvalEnv(
        shocks_eval, prices_eval, sqrt_prices_eval, volumes_eval, pool_liquidity_eval,
        strategy_name="ppo", enable_logging=True, csv_filename=ppo_eval_file
    )])
    eval_env_ppo = VecNormalize(eval_env_ppo, norm_reward=False, training=False)
    model_ppo = PPO.load(PPO_MODEL_PATH, env=eval_env_ppo)
    obs = eval_env_ppo.reset()
    done = [False]
    while not done[0]:
        action, _ = model_ppo.predict(obs, deterministic=True)
        obs, _, done, _ = eval_env_ppo.step(action)

    # Evaluate DQN
    dqn_eval_file = os.path.join(EVAL_LOGS_DIR, f"dqn_seed{SEED}.csv")
    eval_env_dqn = DummyVecEnv([lambda: UniswapV3EvalEnv(
        shocks_eval, prices_eval, sqrt_prices_eval, volumes_eval, pool_liquidity_eval,
        strategy_name="dqn", enable_logging=True, csv_filename=dqn_eval_file
    )])
    eval_env_dqn = VecNormalize(eval_env_dqn, norm_reward=False, training=False)
    model_dqn = DQN.load(DQN_MODEL_PATH, env=eval_env_dqn)
    obs = eval_env_dqn.reset()
    done = [False]
    while not done[0]:
        action, _ = model_dqn.predict(obs, deterministic=True)
        obs, _, done, _ = eval_env_dqn.step(action)

    # Heuristic: price trigger
    heur_price_file = os.path.join(EVAL_LOGS_DIR, f"heur_price_seed{SEED}.csv")
    run_heuristic(
        shocks_eval, prices_eval, sqrt_prices_eval,
        pool_liquidity_eval, volumes_eval,
        strategy_name="heur_price",
        trigger="price",
        initial_price=prices_eval[0],
        csv_filename=heur_price_file
    )

    # Heuristic: volatility trigger
    heur_vol_file = os.path.join(EVAL_LOGS_DIR, f"heur_vol_seed{SEED}.csv")
    run_heuristic(
        shocks_eval, prices_eval, sqrt_prices_eval,
        pool_liquidity_eval, volumes_eval,
        strategy_name="heur_vol",
        trigger="vol",
        initial_price=prices_eval[0],
        csv_filename=heur_vol_file
    )

    print(f"Seed {SEED} done.")

print(f"✓ {N_SEEDS}-seed evaluation finished — see {os.path.join(RESULTS_DIR, 'seed_runs.csv')}")

# Optionally, print summary stats
if os.path.exists(os.path.join(RESULTS_DIR, "seed_runs.csv")):
    df = pd.read_csv(os.path.join(RESULTS_DIR, "seed_runs.csv"))
    print(df.groupby("strategy")["apr"].describe()) 