"""
evaluate_models.py — Batch Evaluation of PPO, DQN, and Heuristic LP Strategies on Uniswap V3

This script runs deterministic evaluations across 50 random seeds for:
- PPO and DQN models trained on synthetic CLM environments
- Two heuristic strategies (price-triggered and volatility-triggered rebalancing)

Each simulation runs for 365 steps on the same historical ETH/USDC evaluation dataset.

Outputs:
- One CSV log per run under: results/eval_logs/
- Summary results are recorded by each environment

References:
- PPO/DQN logic from Section IV-F
- Heuristics based on Eq. (10) and Eq. (11) triggers
- Evaluation setting follows Section V-A in Arcifa et al., IEEE DAPPS 2025

Author: Ricardo Arcifa  
Affiliation: IEEE DAPPS 2025
"""

import os
import numpy as np
import pandas as pd
import random

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.eval_env import UniswapV3EvalEnv
from strategies.heuristic import run_heuristic
from utils.liquidity import get_initial_liquidity_setup, update_tick_liquidity

# Ensure output folder exists
os.makedirs("results/eval_logs", exist_ok=True)

# Run evaluation for 50 seeds
for i in range(50):
    SEED = i
    np.random.seed(SEED)
    random.seed(SEED)

    # Load pre-generated evaluation scenario (realistic market data)
    df = pd.read_csv("./data/evaluation_scenario_data.csv")
    df["Day"] = pd.to_datetime(df["Day"])
    df = df[["Day", "Price", "Inverse_Price", "Volume", "Pool_Liquidity", "Sqrt_Price", "Shock"]]

    shocks_eval = df["Shock"].values
    prices_eval = df["Price"].values
    sqrt_prices_eval = df["Sqrt_Price"].values
    volumes_eval = df["Volume"].values
    pool_liquidity_eval = df["Pool_Liquidity"].values

    # PPO Evaluation
    eval_env_ppo = UniswapV3EvalEnv(
        shocks_eval,
        prices_eval,
        sqrt_prices_eval,
        volumes_eval,
        pool_liquidity_eval,
        strategy_name="ppo",
        seed=SEED,
        initial_investment=10000,
        reposition_gas_cost_range=(10, 50),
        protocol_fee_fraction=0.1,
        tick_spacing=10,
        enable_logging=True,
        csv_filename=f"results/eval_logs/ppo_seed{SEED}.csv",
    )
    eval_env_ppo.reset()

    # Reapply deterministic initial liquidity setup
    init = get_initial_liquidity_setup(10000, prices_eval[0], 10)
    eval_env_ppo.eth_balance = init["eth_balance"]
    eval_env_ppo.usdc_balance = init["usdc_balance"]
    eval_env_ppo.position.update({
        "lower_tick": init["lower_tick"],
        "upper_tick": init["upper_tick"],
        "liquidity": init["liquidity"],
        "active": True
    })
    eval_env_ppo.tick_liquidity = {}
    update_tick_liquidity(eval_env_ppo.tick_liquidity, init["lower_tick"], init["liquidity"])
    update_tick_liquidity(eval_env_ppo.tick_liquidity, init["upper_tick"], -init["liquidity"])

    # Load PPO model and run episode
    vec_env_ppo = DummyVecEnv([lambda: eval_env_ppo])
    vec_env_ppo = VecNormalize(vec_env_ppo, norm_reward=False, training=False)
    model_ppo = PPO.load("./models/uniswap_v3_ppo_model", env=vec_env_ppo)

    obs = vec_env_ppo.reset()
    done = [False]
    while not done[0]:
        action, _ = model_ppo.predict(obs, deterministic=True)
        obs, _, done, _ = vec_env_ppo.step(action)

    vec_env_ppo.close()

    # DQN Evaluation
    eval_env_dqn = UniswapV3EvalEnv(
        shocks_eval,
        prices_eval,
        sqrt_prices_eval,
        volumes_eval,
        pool_liquidity_eval,
        strategy_name="dqn",
        seed=SEED,
        initial_investment=10000,
        reposition_gas_cost_range=(10, 50),
        protocol_fee_fraction=0.1,
        tick_spacing=10,
        enable_logging=True,
        csv_filename=f"results/eval_logs/dqn_seed{SEED}.csv",
    )
    eval_env_dqn.reset()

    # Apply same deterministic init as PPO
    init = get_initial_liquidity_setup(10000, prices_eval[0], 10)
    eval_env_dqn.eth_balance = init["eth_balance"]
    eval_env_dqn.usdc_balance = init["usdc_balance"]
    eval_env_dqn.position.update({
        "lower_tick": init["lower_tick"],
        "upper_tick": init["upper_tick"],
        "liquidity": init["liquidity"],
        "active": True
    })
    eval_env_dqn.tick_liquidity = {}
    update_tick_liquidity(eval_env_dqn.tick_liquidity, init["lower_tick"], init["liquidity"])
    update_tick_liquidity(eval_env_dqn.tick_liquidity, init["upper_tick"], -init["liquidity"])

    vec_env_dqn = DummyVecEnv([lambda: eval_env_dqn])
    vec_env_dqn = VecNormalize(vec_env_dqn, norm_reward=False, training=False)
    model_dqn = DQN.load("./models/uniswap_v3_dqn_model", env=vec_env_dqn)

    obs = vec_env_dqn.reset()
    done = [False]
    while not done[0]:
        action, _ = model_dqn.predict(obs, deterministic=True)
        obs, _, done, _ = vec_env_dqn.step(action)

    vec_env_dqn.close()

    # Heuristic Simulations: Price-triggered and Volatility-triggered
    run_heuristic(
        shocks_eval,
        prices_eval,
        sqrt_prices_eval,
        pool_liquidity_eval,
        volumes_eval,
        strategy_name="heur_price",
        trigger="price",
        seed=SEED,
        initial_price=float(prices_eval[0]),
        csv_filename=f"results/eval_logs/heur_price_seed{SEED}.csv",
    )

    run_heuristic(
        shocks_eval,
        prices_eval,
        sqrt_prices_eval,
        pool_liquidity_eval,
        volumes_eval,
        strategy_name="heur_vol",
        trigger="vol",
        seed=SEED,
        initial_price=float(prices_eval[0]),
        csv_filename=f"results/eval_logs/heur_vol_seed{SEED}.csv",
    )

# Done
print("Evaluation for all 50 seeds completed. Results saved in results/eval_logs/")
