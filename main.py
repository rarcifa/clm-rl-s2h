"""
run_all.py — Script to Train and Evaluate RL and Heuristic CLM Strategies

This is the main entry point to execute the full pipeline:
    1. Trains PPO and DQN agents using `train.py`
    2. Evaluates trained agents and heuristics using `evaluate.py`

Assumes both scripts are in the same directory and properly configured.

Author: Ricardo Arcifa  
Affiliation: IEEE DAPPS 2025
"""

import subprocess

# Step 1: Train PPO and DQN agents
print("Running Training...")
subprocess.run(["python", "train.py"], check=True)

# Step 2: Evaluate agents and heuristics over 50 seeds
print("Running Evaluation...")
subprocess.run(["python", "evaluate.py"], check=True)
