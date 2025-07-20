# CLM-RL-S2H: Deep RL for Concentrated Liquidity Management on Uniswap v3

This repository implements a full synthetic-to-historical evaluation framework for **Concentrated Liquidity Management (CLM)** using deep reinforcement learning (RL). We benchmark **Proximal Policy Optimization (PPO)** and **Deep Q-Networks (DQN)** against two common heuristic strategies for managing Uniswap v3 LP positions.

**Reference**  
This codebase supports the experiments and analysis presented in the following paper:

> **Optimizing Concentrated Liquidity Management: A Synthetic-to-Historical Deep Reinforcement Learning Strategy**  
> Ricardo Arcifa, Yuhang Ye, Yuansong Qiao, Brian Lee  
> _IEEE DAPPS 2025 (Decentralized Applications and Infrastructures)_  
> [PDF available](https://github.com/rarcifa/clm-rl-s2h) :contentReference[oaicite:0]{index=0}

---

## Project Highlights

- **End-to-end pipeline** for training, evaluation, and comparison of CLM strategies.
- **Synthetic-to-Historical framework**: agents are trained on synthetic price/volume scenarios and evaluated on **real historical ETH/USDC Uniswap v3 data (Mar 2024 – Mar 2025)**.
- **Deterministic heuristic strategies**:
  - **HP**: rebalances on ±5% price deviation.
  - **HV**: rebalances when 7-day rolling volatility exceeds 3%.
- **Reinforcement Learning strategies**:
  - **PPO** (Stable-Baselines3)
  - **DQN** (Stable-Baselines3)

---

## Directory Structure

```bash
.
├── environments/              # Custom Gym-compatible environments
│   ├── train_env.py
│   └── eval_env.py
├── strategies/                # Heuristic rebalancing logic
│   └── heuristic.py
├── utils/                     # Core utility functions
│   ├── core.py
│   ├── liquidity.py
│   ├── callbacks.py
│   └── portfolio_base.py
├── results/                   # Output folder for logs, tables, plots
│   ├── eval_logs/             # Per-strategy evaluation logs (CSV)
│   ├── eval_logs/seeds/       # Per-seed evaluation logs (CSV)
│   └── seed_runs.csv          # Summary metrics across seeds
├── models/                    # Saved RL models (PPO, DQN)
├── data/
│   └── evaluation_scenario_data.csv  # 1-year ETH/USDC market data
├── train.py                   # Train PPO and DQN agents
├── eval.py                    # Evaluate PPO, DQN, and heuristics on a fixed scenario
├── seed_simulation.py         # Multi-seed evaluation for RL and heuristics
├── make_gas_sensitivity.py    # Generates Table IV: APR under 0.5×/1×/2× gas
├── make_portfolio_tables.py   # Generates median portfolio curves
├── make_tables.py             # Generates all APR visualizations
└── analyze_final_results.py   # Produces Table V–VII stats + Wilcoxon tests
```

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

**Dependencies**:

- Python ≥ 3.8
- `stable-baselines3`
- `gym`
- `seaborn`, `matplotlib`, `pandas`, `numpy`

---

## Workflow: Training, Evaluation, and Multi-Seed Simulation

### 2. Train RL Agents (PPO & DQN)

```bash
python train.py
```

- Trains PPO and DQN agents on synthetic data.
- Saves models to `models/uniswap_v3_ppo_model` and `models/uniswap_v3_dqn_model`.

### 3. Evaluate on a Fixed Scenario (Single Run)

```bash
python eval.py
```

- Loads the trained models.
- Evaluates PPO and DQN on a fixed historical scenario (`data/evaluation_scenario_data.csv`).
- Runs both heuristic strategies for comparison.
- Logs results to `results/eval_logs/` and summary to `results/seed_runs.csv`.

### 4. Multi-Seed Simulation (Robustness/Statistical Analysis)

```bash
python seed_simulation.py
```

- Loops over 50 random seeds (configurable in the script).
- For each seed:
  - Sets the random seed for reproducibility.
  - Generates a new synthetic scenario.
  - Evaluates PPO, DQN, and both heuristics on the scenario.
  - Logs per-seed results to `results/eval_logs/seeds/` and summary to `results/seed_runs.csv`.
- At the end, prints summary statistics for all strategies.

**Note:**

- Make sure you have trained models in `models/` before running `seed_simulation.py`.
- You can change the number of seeds by editing `N_SEEDS` in `seed_simulation.py`.
- The random seed ensures reproducibility of each scenario and evaluation.

---

## Reproducibility & Random Seeds

- Each run of `seed_simulation.py` sets the random seed for both `numpy` and `random` for full reproducibility.
- Per-seed logs and summary metrics allow for robust statistical analysis and fair comparison of strategies.

---

## Evaluation Metrics

- **APR (%)**: Net return including LP fees, minus gas costs
- **Impermanent Loss (IL)**: Relative to 50/50 buy-and-hold portfolio
- **Gas Fees**: USD cost of rebalancing (realistic gas per day via Etherscan)
- **LP Fees**: Gross fees accrued from swap volume

---

## Example Output (Table V)

| Strategy | APR (%)     | IL (%)     | LP Fees ($K) | Gas Fees ($K) |
| -------- | ----------- | ---------- | ------------ | ------------- |
| PPO      | 46 ± 40     | −66 ± 25   | 12.7 ± 3.7   | 0.7 ± 0.08    |
| DQN      | 18.6 ± 44.2 | −64.3 ± 12 | 9.9 ± 4.4    | 0.8 ± 0.1     |
| HP       | −30.6 ± 33  | −52.9 ± 26 | 6.7 ± 3.27   | 3.4 ± 0.75    |
| HV       | −82.4 ± 17  | −64.1 ± 10 | 7.7 ± 1.9    | 8.7 ± 0.2     |

---

## Troubleshooting

- If `results/seed_runs.csv` is missing after running `seed_simulation.py`, check for errors in the console output and ensure that the models exist in the `models/` directory.
- Make sure all dependencies are installed and the directory structure matches the above.

---

## Citation

If you use this framework in academic work, please cite:

```bibtex
@inproceedings{arcifa2025clm,
  title     = {Optimizing Concentrated Liquidity Management: A Synthetic-to-Historical Deep Reinforcement Learning Strategy},
  author    = {Ricardo Arcifa and Yuhang Ye and Yuansong Qiao and Brian Lee},
  booktitle = {IEEE International Conference on Decentralized Applications and Infrastructures (DAPPS)},
  year      = {2025}
}
```

---

## Acknowledgements

This work was supported by the President’s Doctoral Scholarship from the **Technological University of the Shannon (TUS), Ireland**, awarded in 2021.

---

## Contact

Feel free to reach out with questions or contributions:

- Ricardo Arcifa — [a00279376@student.tus.ie](mailto:a00279376@student.tus.ie)
- GitHub: [github.com/rarcifa](https://github.com/rarcifa)

---

## License

MIT License — see `LICENSE.md` for details.
