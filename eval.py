"""
eval.py — RL Agent and Heuristic Evaluation for Uniswap V3 LP Strategies

Loads trained PPO and DQN models, evaluates them on a fixed scenario, and runs baseline heuristic strategies for comparison. Evaluation logs are saved to disk.

Author: Ricardo Arcifa
Affiliation: IEEE DAPPS 2025
"""

import os
import pandas as pd
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environments.eval_env import UniswapV3EvalEnv
from strategies.heuristic import run_heuristic

# Ensure results directories exist
os.makedirs("results/eval_logs", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Load evaluation scenario from CSV
# Columns: Day, Price, Inverse_Price, Volume, Pool_Liquidity, Sqrt_Price, Shock

df = pd.read_csv("data/evaluation_scenario_data.csv")
df = df[['Day', 'Price', 'Inverse_Price', 'Volume', 'Pool_Liquidity', 'Sqrt_Price', 'Shock']]
df['Day'] = pd.to_datetime(df['Day'])

shocks_eval = df['Shock'].values
prices_eval = df['Price'].values
sqrt_prices_eval = df['Sqrt_Price'].values
volumes_eval = df['Volume'].values
pool_liquidity_eval = df['Pool_Liquidity'].values

# Evaluate PPO on fixed scenario
# Logs to results/eval_logs/ppo_eval.csv

eval_env_ppo = DummyVecEnv([lambda: UniswapV3EvalEnv(
    shocks_eval, prices_eval, sqrt_prices_eval, volumes_eval, pool_liquidity_eval,
    strategy_name="ppo", enable_logging=True, csv_filename="results/eval_logs/ppo_eval.csv"
)])
eval_env_ppo = VecNormalize(eval_env_ppo, norm_reward=False, training=False)
model_ppo = PPO.load("models/uniswap_v3_ppo_model", env=eval_env_ppo)

obs = eval_env_ppo.reset()
done = [False]
while not done[0]:
    action, _ = model_ppo.predict(obs, deterministic=True)
    obs, _, done, _ = eval_env_ppo.step(action)

# Evaluate DQN on fixed scenario
# Logs to results/eval_logs/dqn_eval.csv

eval_env_dqn = DummyVecEnv([lambda: UniswapV3EvalEnv(
    shocks_eval, prices_eval, sqrt_prices_eval, volumes_eval, pool_liquidity_eval,
    strategy_name="dqn", enable_logging=True, csv_filename="results/eval_logs/dqn_eval.csv"
)])
eval_env_dqn = VecNormalize(eval_env_dqn, norm_reward=False, training=False)
model_dqn = DQN.load("models/uniswap_v3_dqn_model", env=eval_env_dqn)

obs = eval_env_dqn.reset()
done = [False]
while not done[0]:
    action, _ = model_dqn.predict(obs, deterministic=True)
    obs, _, done, _ = eval_env_dqn.step(action)

# Run heuristic strategy: price trigger
run_heuristic(
    shocks_eval, prices_eval, sqrt_prices_eval,
    pool_liquidity_eval, volumes_eval,
    strategy_name="heur_price",
    trigger="price",
    initial_price=prices_eval[0],
    csv_filename="results/eval_logs/heur_price.csv"
)

# Run heuristic strategy: volatility trigger
run_heuristic(
    shocks_eval, prices_eval, sqrt_prices_eval,
    pool_liquidity_eval, volumes_eval,
    strategy_name="heur_vol",
    trigger="vol",
    initial_price=prices_eval[0],
    csv_filename="results/eval_logs/heur_vol.csv"
)

print("All models evaluated and heuristic strategies completed.") 