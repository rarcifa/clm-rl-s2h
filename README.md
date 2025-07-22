# CLM-RL-S2H: Deep RL for Concentrated Liquidity Management on Uniswap v3

This repository implements a full synthetic-to-historical evaluation framework for **Concentrated Liquidity Management (CLM)** using deep reinforcement learning (RL). We benchmark **Proximal Policy Optimization (PPO)** and **Deep Q-Networks (DQN)** against two common heuristic strategies for managing Uniswap v3 LP positions.

**Reference**
This codebase supports the experiments and analysis presented in the following paper:

> **Optimizing Concentrated Liquidity Management: A Synthetic-to-Historical Deep Reinforcement Learning Strategy**
> Ricardo Arcifa, Yuhang Ye, Yuansong Qiao, Brian Lee
> _IEEE DAPPS 2025 (Decentralized Applications and Infrastructures)_ > [PDF available](https://github.com/rarcifa/clm-rl-s2h)

---

## Project Highlights

- **End-to-end pipeline** for training, evaluation, and benchmarking CLM strategies.
- **Synthetic-to-Historical framework**: agents are trained on **synthetic** ETH/USDC scenarios and evaluated on **real historical Uniswap v3 data**.
- **Deterministic heuristic strategies**:

  - **HP**: triggers on ±5% price move.
  - **HV**: triggers when 7-day rolling volatility exceeds 3%.

- **Reinforcement Learning strategies**:

  - **PPO** (Stable-Baselines3)
  - **DQN** (Stable-Baselines3)

- **Per-seed logging and statistical analysis** across 50 seeds

---

## Directory Structure

```bash
.
├── core/
│   ├── train.py                  # Trains PPO and DQN on synthetic data
│   ├── evaluate.py               # Evaluates PPO, DQN, and heuristics on real data
├── envs/
│   ├── train_env.py              # UniswapV3TrainEnv class (gym.Env)
│   └── eval_env.py               # UniswapV3EvalEnv class (gym.Env)
├── strategies/
│   └── heuristic.py              # HP and HV heuristic strategies
├── utils/
│   ├── scenario.py               # Scenario generation (GBM)
│   ├── liquidity.py              # Position math, ticks, and balances
│   ├── callbacks.py              # Custom reward logging callback
├── results/
│   ├── train_logs/
│   │   └── synthetic_training_scenario.npz
│   ├── eval_logs/
│   │   ├── ppo_seed*.csv
│   │   ├── dqn_seed*.csv
│   │   ├── heuristic_price_seed*.csv
│   │   └── heuristic_vol_seed*.csv
│   └── seed_runs.csv             # Summary stats across seeds
├── models/
│   ├── uniswap_v3_ppo_model.zip
│   └── uniswap_v3_dqn_model.zip
├── data/
│   └── evaluation_scenario_data.csv  # Real ETH/USDC historical data (Mar 2024–Mar 2025)
├── main.py                       # Launches training and evaluation
├── LICENSE.md
└── README.md
```

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

**Python Requirements:**

- Python ≥ 3.8
- `stable-baselines3`
- `gym` or `gymnasium`
- `numpy`, `pandas`, `matplotlib`, `seaborn`

---

## Workflow: Training, Evaluation, and Multi-Seed Simulation

### 2. Train RL Agents (PPO & DQN)

```bash
python train.py
```

- Generates 1 synthetic scenario.
- Trains PPO and DQN agents on the same scenario.
- Saves models to `./models/` and training logs to `./results/train_logs/`.

### 3. Evaluate on Real Historical Scenario

```bash
python evaluate.py
```

- Loads real ETH/USDC data from `data/evaluation_scenario_data.csv`
- Evaluates:

  - PPO
  - DQN
  - Heuristic HP (price-based)
  - Heuristic HV (volatility-based)

- Logs per-seed results to `results/eval_logs/`
- Summary stats written to `results/seed_runs.csv`

### 4. Run Train and Evaluation as Subprocess

```bash
python main.py
```

- Runs 50 evaluations with different seeds.
- Each seed:

  - Generates a new synthetic scenario
  - Evaluates PPO, DQN, HP, HV

- Summary and per-seed logs written to `results/eval_logs/seeds/` and `results/seed_runs.csv`

---

## Reproducibility

- Each seed sets `np.random.seed()` and `random.seed()`.
- Trained PPO and DQN agents are evaluated under **the same market conditions**.
- Synthetic scenario used in training is saved to `.npz` for traceability.

---

## Evaluation Metrics

- **APR (%)**: Net return incl. LP fees – gas
- **Impermanent Loss (%)**: Relative to HODL
- **LP Fees (\$)**: Revenue from swaps
- **Gas Fees (\$)**: Cumulative gas for repositioning

---

## Example Output (Table V from paper)

| Strategy | APR (%)     | IL (%)     | LP Fees (\$K) | Gas Fees (\$K) |
| -------- | ----------- | ---------- | ------------- | -------------- |
| PPO      | 46 ± 40     | −66 ± 25   | 12.7 ± 3.7    | 0.7 ± 0.08     |
| DQN      | 18.6 ± 44.2 | −64.3 ± 12 | 9.9 ± 4.4     | 0.8 ± 0.1      |
| HP       | −30.6 ± 33  | −52.9 ± 26 | 6.7 ± 3.27    | 3.4 ± 0.75     |
| HV       | −82.4 ± 17  | −64.1 ± 10 | 7.7 ± 1.9     | 8.7 ± 0.2      |

---

## Troubleshooting

- `ModuleNotFoundError: No module named 'envs'`
  → Run from project root:

  ```bash
  python core/train.py
  python core/evaluate.py
  ```

- `np.save() got an unexpected keyword argument`
  → Use `np.savez(...)` to save multiple arrays.

- PPO or DQN performance is unstable
  → Check seeds, reward scale, and number of steps.

---

## Citation

```bibtex
@inproceedings{arcifa2025clm,
  title     = {Optimizing Concentrated Liquidity Management: A Synthetic-to-Historical Deep Reinforcement Learning Strategy},
  author    = {Ricardo Arcifa and Yuhang Ye and Yuansong Qiao and Brian Lee},
  booktitle = {IEEE International Conference on Decentralized Applications and Infrastructures (DAPPS)},
  year      = {2025}
}
```

---

## Contact

- Ricardo Arcifa — [a00279376@student.tus.ie](mailto:a00279376@student.tus.ie)
- GitHub — [github.com/rarcifa](https://github.com/rarcifa)

---

## License

MIT License — see `LICENSE.md` for details.
