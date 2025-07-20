"""
make_daily_gas_stats.py — Etherscan Gas Cost Processor

This utility converts Etherscan's daily average gas CSV export into a clean,
analysis-ready CSV containing daily gas cost in Gwei and USD terms.

Input CSV (etherscan_gas_raw.csv) format:
  - Date(UTC)
  - UnixTimeStamp
  - Value (Wei): average gas price in Wei
  - token0Price: ETH price in USD

Output CSV (daily_gas_stats.csv) format:
  - date (YYYY-MM-DD)
  - gwei: average gas price
  - usd_tx: estimated USD cost of an average Uniswap v3 transaction

Author: Ricardo Arcifa  
Affiliation: IEEE DAPPS 2025
"""

import pandas as pd
from pathlib import Path

# Constants
RAW_CSV  = Path("etherscan_gas_raw.csv")     # Raw Etherscan gas CSV
GAS_UNITS = 360_000                          # Typical gas units for mint/swap in Uniswap v3
OUT_CSV  = Path("daily_gas_stats.csv")       # Output path

def gas_calculation() -> None:
    """
    Converts Etherscan raw gas data to daily Gwei + USD/tx estimates.

    Steps:
    1. Skip the banner row and load selected columns.
    2. Convert column names and types.
    3. Convert gas price from Wei → Gwei.
    4. Compute USD cost per transaction.
    5. Save cleaned output to disk.
    """
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"{RAW_CSV} not found")

    # ── 1. Load CSV, skipping banner row
    df = pd.read_csv(
        RAW_CSV,
        skiprows=1,
        usecols=["Date(UTC)", "Value (Wei)", "token0Price"]
    ).rename(columns={
        "Date(UTC)"   : "date",
        "Value (Wei)" : "wei",
        "token0Price" : "eth_usd"
    })

    # ── 2. Parse and convert column types
    df["date"]    = pd.to_datetime(df["date"]).dt.date.astype(str)
    df["wei"]     = df["wei"].astype(float)
    df["eth_usd"] = df["eth_usd"].astype(float)

    # ── 3. Convert Wei → Gwei
    df["gwei"] = df["wei"] / 1e9

    # ── 4. Estimate USD cost per transaction
    # gwei → ETH (× 1e-9), then × gas units, then × ETH/USD price
    df["usd_tx"] = df["gwei"] * 1e-9 * GAS_UNITS * df["eth_usd"]

    # ── 5. Output cleaned file
    df = df[["date", "gwei", "usd_tx"]]
    df.to_csv(OUT_CSV, index=False, float_format="%.2f")

    # Summary stats
    lo, hi = df["usd_tx"].min(), df["usd_tx"].max()
    print(f"wrote {OUT_CSV}  ({len(df)} rows, {lo:.2f}–{hi:.2f} USD/tx)")


