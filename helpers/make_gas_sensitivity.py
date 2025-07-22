"""
make_gas_sensitivity.py — Gas Sensitivity Evaluation for LP Strategies

This script processes the evaluation results in `results/seed_runs.csv` and computes
the mean APR under three different gas-cost scenarios: 0.5×, 1×, and 2× gas fees.

The APR is adjusted linearly based on gas costs relative to the initial capital,
and the output is a CSV file with strategy-level average APRs across different
gas assumptions.

Author: Ricardo Arcifa  
Affiliation: IEEE DAPPS 2025
"""

import pandas as pd
from pathlib import Path

# configuration
CSV_MAIN = Path("../results/seed_runs.csv")          # Input file (per-seed evaluation results)
OUT_CSV  = Path("../results/gas_sensitivity.csv")    # Output file (mean APR by strategy)
INIT_USDC = 10_000                                    # Starting capital in USD

# required columns
REQ_COLS = {
    "eth_balance", "usdc_balance", "pool_eth", "pool_usdc",
    "price", "gas_fees", "apr"
}

# load and validate input data
df = pd.read_csv(CSV_MAIN)
df["strategy"] = df["strategy"].str.lower()

missing = REQ_COLS.difference(df.columns)
if missing:
    raise RuntimeError("CSV is missing required columns: " + ", ".join(sorted(missing)))

# compute APR under different gas-cost assumptions
# APR shift per $1 of gas spent = (100 / INIT_USDC)
factor = 100.0 / INIT_USDC

df["apr_gas_half"]   = df["apr"] + 0.5 * df["gas_fees"] * factor
df["apr_gas_double"] = df["apr"] - 1.0 * df["gas_fees"] * factor
df["apr_1x"]         = df["apr"]  # baseline (unadjusted)

# aggregate average APR across all seeds per strategy
summary = (
    df.groupby("strategy")[["apr_gas_half", "apr_1x", "apr_gas_double"]]
      .mean()
      .round(1)
      .rename(columns={
          "apr_gas_half"  : "0.5× gas",
          "apr_1x"        : "1× gas",
          "apr_gas_double": "2× gas"
      })
)

# save and print result
summary.to_csv(OUT_CSV)
print(f"wrote {OUT_CSV} ({len(summary)} strategies)")
print(summary)
