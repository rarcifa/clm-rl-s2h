"""
logging_helpers.py — Utilities for CSV-Based Result Logging

Provides helper functions to record simulation or evaluation summaries
(e.g., APR, LP fees, gas usage, IL) into CSV files. Automatically handles
header creation and appends structured result rows per run.

Used in:
- run_heuristic.py
- UniswapV3EvalEnv
- UniswapV3TrainEnv

Author: Ricardo Arcifa  
Affiliation: IEEE DAPPS 2025
"""

import csv
import os


def record_summary(path, rowdict):
    """
    Append a single summary row to a CSV file, creating header if needed.

    Args:
        path (str): Path to the CSV file (e.g., 'results/seed_runs.csv').
        rowdict (dict): Dictionary containing column values for this run.

    Behavior:
        - If the file doesn't exist yet, it creates it and writes the header.
        - If the file exists, appends the row directly.
        - Header is inferred from the keys of the input dictionary.

    Example:
        record_summary("results/seed_runs.csv", {
            "seed": 0,
            "strategy": "heur_price",
            "apr": 42.1,
            ...
        })
    """
    header = sorted(rowdict.keys())  # Alphabetical order ensures stable columns
    new_file = not os.path.isfile(path)

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if new_file:
            writer.writeheader()
        writer.writerow(rowdict)
