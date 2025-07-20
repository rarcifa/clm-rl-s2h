"""
make_portfolio_tables.py — Build Median Portfolio Trajectories and Benchmark Curve

This script reads per-seed evaluation logs from `results/eval_logs/*.csv` and computes:

1. The median net-of-gas portfolio value at each timestep per strategy
2. A benchmark "buy and hold" trajectory using the same ETH price path

The outputs are written to:
- `results/median_portfolio.csv`
- `results/holding_curve.csv`

Author: Ricardo Arcifa  
Affiliation: IEEE DAPPS 2025
"""

import pathlib
import re
import pandas as pd
import numpy as np

# --- configuration ---
LOG_DIR = pathlib.Path("../results/eval_logs")             # per-seed evaluation logs
OUT1    = pathlib.Path("../results/median_portfolio.csv")  # median portfolio values per step
OUT2    = pathlib.Path("../results/holding_curve.csv")     # passive holding benchmark

INIT_USDC = 5_000                                           # half of initial capital in USDC
INIT_ETH = (10_000 / 2) / 3_525.13                          # other half converted to ETH at day 0

STRATS = ["ppo", "dqn", "heur_price", "heur_vol"]          # known strategies, for output column order
pat = re.compile(rf"({'|'.join(STRATS)})_seed(\d+)\.csv", re.I)  # pattern to extract strategy and seed


def one_file(path: pathlib.Path) -> pd.DataFrame:
    """
    Parse a single per-seed evaluation log into [step, strategy, seed, net portfolio value].

    Args:
        path (Path): Path to the log CSV.

    Returns:
        pd.DataFrame: Long-format DataFrame with net-of-gas portfolio value.
    """
    m = pat.search(path.name)
    if not m:
        raise ValueError(f"Unexpected file format: {path.name}")

    strat, seed = m.group(1).lower(), int(m.group(2))
    df = pd.read_csv(path)

    gross_value = df["total_value"]
    if "total_gas_fees" in df:
        net_value = gross_value - df["total_gas_fees"]
    else:
        net_value = gross_value  # fallback (should not occur)

    return pd.DataFrame({
        "step": df["step"],
        "strategy": strat,
        "seed": seed,
        "portfolio": net_value
    })


# --- read and parse all per-seed logs ---
frames = [one_file(p) for p in LOG_DIR.glob("*.csv")]
if not frames:
    raise SystemExit("No per-seed logs found – did you write them?")

data = pd.concat(frames, ignore_index=True)

# --- compute median portfolio trajectory (per step, per strategy) ---
median = (
    data.groupby(["step", "strategy"])["portfolio"]
        .median()
        .unstack("strategy")
        .reindex(columns=STRATS)  # keep consistent column order
        .round(2)
)

median.to_csv(OUT1)
print("wrote", OUT1)

# --- generate buy-and-hold benchmark curve ---
# ETH price path is same for all seeds 
any_log = next(LOG_DIR.glob("ppo_seed*.csv"))
prices = pd.read_csv(any_log)["price"]
holding = INIT_USDC + INIT_ETH * prices

holding.to_frame(name="holding").reset_index(names="step").to_csv(OUT2, index=False)
print("wrote", OUT2)
