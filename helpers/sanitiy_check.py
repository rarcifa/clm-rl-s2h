"""
analyze_final_results.py — Descriptive Statistics and Wilcoxon Test for LP Strategy Evaluation

This script analyzes the final evaluation results stored in `seed_runs.csv`. It performs:

1. Per-strategy descriptive statistics (mean, std, min, max)
2. LaTeX-formatted summary lines for Table VII
3. Pairwise Wilcoxon signed-rank tests on APR distributions

The script assumes one row per (seed, strategy) and that all strategies share the same seeds.

Author: Ricardo Arcifa  
Affiliation: IEEE DAPPS 2025
"""

import pandas as pd
from scipy.stats import wilcoxon

# --- configuration ---
PATH = "seed_runs.csv"              # Path to input results CSV
STRATEGIES = ["ppo", "dqn", "heur_price", "heur_vol"]
METRICS = {
    "apr"              : "APR (\\%)",
    "impermanent_loss" : "IL (\\%)",
    "lp_fees"          : "LP Fees (\\$)",
    "gas_fees"         : "Gas Fees (\\$)",
}

# --- load and pivot data ---
df = pd.read_csv(PATH)
df = df.drop_duplicates(subset=["seed", "strategy"], keep="first")  # ensure 1 row per (seed, strategy)

pivot = df.pivot(index="seed", columns="strategy", values="apr").sort_index()

# --- 1. Per-strategy descriptive statistics ---
summary = (
    df.groupby("strategy")[list(METRICS.keys())]
      .agg(["mean", "std", "min", "max"])
      .round(2)
)

print("=== Per-strategy summary ===")
print(summary)
print()

# --- 2. Table VII LaTeX lines ---
agg = (
    df.groupby("strategy")[list(METRICS.keys())]
      .agg(["mean", "std"])
      .round(2)
)

print("=== Table VII (LaTeX format) ===")
for metric, label in METRICS.items():
    values = []
    for strat in STRATEGIES:
        mean = agg.loc[strat, (metric, "mean")]
        std = agg.loc[strat, (metric, "std")]
        values.append(f"{mean} \\pm {std}")
    print(f"{label} & $" + "$ & $".join(values) + "$ \\\\")
print()

# --- 3. Pairwise Wilcoxon tests (APR only) ---
pairs = [
    ("ppo", "dqn"),
    ("ppo", "heur_price"),
    ("ppo", "heur_vol"),
    ("dqn", "heur_price"),
    ("dqn", "heur_vol"),
]

print("=== Wilcoxon p-values (APR) ===")
for a, b in pairs:
    a_vals = pivot[a]
    b_vals = pivot[b]
    paired = pd.concat([a_vals, b_vals], axis=1).dropna()
    if not paired.empty:
        stat, pval = wilcoxon(paired[a], paired[b])
        print(f"{a:>9} vs {b:<10}:  p = {pval:.3g}")
    else:
        print(f"{a:>9} vs {b:<10}:  not enough data")
