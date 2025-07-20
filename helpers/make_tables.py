"""
make_tables.py — Visualization of APR Distributions for LP Strategies

This script reads the final evaluation results from `results/seed_runs.csv`
and generates multiple PDF figures showing APR distributions for each strategy.

Figures produced:
    - fig_box_rl_apr.pdf     — box + scatter for RL strategies (ppo, dqn)
    - fig_box_hp_apr.pdf     — box + scatter for heur_price
    - fig_box_hv_apr.pdf     — box + scatter for heur_vol

These plots are designed for publication-quality rendering (tight layout, consistent color palette).

Author: Ricardo Arcifa  
Affiliation: IEEE DAPPS 2025
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- configuration ---
CSV = "../results/seed_runs.csv"     # Evaluation summary file
OUT_RL  = "fig_box_rl_apr.pdf"
OUT_HP  = "fig_box_hp_apr.pdf"
OUT_HV  = "fig_box_hv_apr.pdf"

# --- load and prepare data ---
df = pd.read_csv(CSV)
df["strategy"] = df["strategy"].str.lower()

# --- define strategy groups ---
rl_strats   = ["ppo", "dqn"]
heur_strats = ["heur_price", "heur_vol"]

df_rl   = df[df["strategy"].isin(rl_strats)]
df_hp   = df[df["strategy"] == "heur_price"]
df_hv   = df[df["strategy"] == "heur_vol"]

# --- consistent color palette across all figures ---
base_palette = sns.color_palette("Set2", n_colors=4)
pal_rl = {s: c for s, c in zip(sorted(rl_strats), base_palette[:2])}
pal_hp = {"heur_price": base_palette[2]}
pal_hv = {"heur_vol"  : base_palette[3]}


def make_single_box(data: pd.DataFrame, fname: str, palette: dict) -> None:
    """
    Draw a boxplot + scatterplot of APR values per strategy.

    Args:
        data (pd.DataFrame): Filtered data for one or more strategies.
        fname (str): Output filename for the figure.
        palette (dict): Mapping of strategy → color.
    """
    plt.figure(figsize=(3.8, 2.8))
    sns.boxplot(data=data, x="strategy", y="apr", palette=palette, width=0.6)
    sns.stripplot(data=data, x="strategy", y="apr",
                  color="k", size=3, alpha=0.4, jitter=True)
    plt.ylabel("APR (%)")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()


# --- generate figures ---
make_single_box(df_rl, OUT_RL, pal_rl)
make_single_box(df_hp, OUT_HP, pal_hp)
make_single_box(df_hv, OUT_HV, pal_hv)

print(f"wrote {OUT_RL}, {OUT_HP}, {OUT_HV}")
