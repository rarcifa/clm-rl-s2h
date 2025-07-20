"""
run_heuristic.py — Simulation of Heuristic LP Strategies on Uniswap V3

This module simulates a non-learning LP strategy (based on price or volatility triggers)
over a fixed ETH/USDC scenario. It logs daily portfolio metrics and records a summary
row including APR, gas cost, LP fees, and impermanent loss to `results/seed_runs.csv`.

Author: Ricardo Arcifa  
Affiliation: IEEE DAPPS 2025
"""

import csv
import random
import numpy as np
from utils.core import (
    tick_to_price,
    tick_to_sqrt_price,
    price_to_tick,
    nearest_tick,
    compute_liquidity_L,
    amounts_in_position,
    update_tick_liquidity,
    cross_tick_and_update_liquidity,
    record_summary
)
from utils.liquidity import get_initial_liquidity_setup
from utils.portfolio_base import UniswapV3PortfolioBase


def run_heuristic(
    shocks,
    prices_eval,
    sqrt_prices_eval,
    pool_liquidity_eval,
    volumes_eval,
    strategy_name,
    trigger="price",
    steps=365,
    initial_investment=10000,
    reposition_gas_cost_range=(50, 150),
    protocol_fee_fraction=0.1,
    tick_spacing=10,
    initial_price=2200.0,
    csv_filename="heuristic_simulation.csv",
):
    """
    Simulate a heuristic LP strategy over a fixed scenario using UniswapV3PortfolioBase for all state and operations.
    """
    # --- initialize portfolio ---
    portfolio = UniswapV3PortfolioBase(
        initial_investment=initial_investment,
        initial_price=initial_price,
        tick_spacing=tick_spacing,
        protocol_fee_fraction=protocol_fee_fraction,
        reposition_gas_cost_range=reposition_gas_cost_range
    )
    # --- volatility trigger tracking ---
    price_window = []
    VOL_WINDOW_LEN = 7
    SIGMA_THRESHOLD = 0.04
    # --- open output log file ---
    fieldnames = [
        "step", "price", "active", "liquidity", "eth_in_wallet", "usdc_in_wallet",
        "eth_in_pool", "usdc_in_pool", "total_value", "lp_fees", "protocol_fees",
        "cumulative_lp_fees", "cumulative_protocol_fees", "gas_fees_step",
        "total_gas_fees", "action"
    ]
    writer = csv.DictWriter(open(csv_filename, "w", newline=""), fieldnames)
    writer.writeheader()
    # --- main simulation loop ---
    for step_i in range(steps):
        price = float(prices_eval[step_i])
        volume = float(volumes_eval[step_i])
        shock = float(shocks[step_i])
        portfolio.pool_liquidity = float(pool_liquidity_eval[step_i])
        # Tick crossing
        portfolio.cross_tick(price, shock)
        # --- reposition triggers ---
        p_lower = tick_to_price(portfolio.position["lower_tick"])
        p_upper = tick_to_price(portfolio.position["upper_tick"])
        active_now = p_lower <= price <= p_upper
        price_trigger = portfolio.position["active"] and not active_now
        price_window.append(price)
        if len(price_window) > VOL_WINDOW_LEN:
            price_window.pop(0)
        vol_trigger = False
        if len(price_window) == VOL_WINDOW_LEN:
            sigma = np.std(price_window) / np.mean(price_window)
            vol_trigger = sigma > SIGMA_THRESHOLD
        if trigger == "price":
            need_reposition = price_trigger
        elif trigger == "vol":
            need_reposition = vol_trigger
        else:
            raise ValueError(f"Unknown trigger: {trigger}")
        gas_fee = 0.0
        actions = []
        if need_reposition:
            gas_fee = portfolio.reposition(price)
            actions.append("reposition")
        # --- LP fee tier determination and fee calculation ---
        lp_fees, protocol_fee, fee_tier = portfolio.compute_fees(volume, shock)
        # --- compute portfolio value ---
        total_val, pool0, pool1 = portfolio.compute_portfolio_value(price)
        writer.writerow({
            "step": step_i,
            "price": round(price, 2),
            "active": portfolio.position["active"],
            "liquidity": round(portfolio.position["liquidity"], 2),
            "eth_in_wallet": round(portfolio.eth_balance, 4),
            "usdc_in_wallet": round(portfolio.usdc_balance, 2),
            "eth_in_pool": round(pool0, 4),
            "usdc_in_pool": round(pool1, 2),
            "total_value": round(total_val, 2),
            "lp_fees": round(lp_fees, 2),
            "protocol_fees": round(protocol_fee, 2),
            "cumulative_lp_fees": round(portfolio.cumulative_lp_fees, 2),
            "cumulative_protocol_fees": round(portfolio.cumulative_protocol_fees, 2),
            "gas_fees_step": round(gas_fee, 2),
            "total_gas_fees": round(portfolio.position["total_gas_fees"], 2),
            "action": "; ".join(actions) if actions else "no action"
        })
        portfolio.position["active"] = active_now
    # --- compute summary metrics at the end ---
    final_price = prices_eval[-1]
    pool0, pool1 = amounts_in_position(
        portfolio.position["liquidity"], final_price,
        tick_to_price(portfolio.position["lower_tick"]),
        tick_to_price(portfolio.position["upper_tick"])
    )
    final_eth = portfolio.eth_balance + pool0
    final_usdc = portfolio.usdc_balance + pool1
    gross_val = final_eth * final_price + final_usdc
    price_val = gross_val - portfolio.cumulative_lp_fees
    hodl_eth = (initial_investment / 2) / initial_price
    hodl_val = hodl_eth * final_price + (initial_investment / 2)
    classic_il = ((price_val - hodl_val) / hodl_val) * 100
    net_profit = gross_val - portfolio.position["total_gas_fees"] - initial_investment
    apr = (net_profit / initial_investment) * 100
    record_summary("results/seed_runs.csv", {
        "strategy": strategy_name,
        "apr": round(apr, 4),
        "impermanent_loss": round(classic_il, 2),
        "lp_fees": round(portfolio.cumulative_lp_fees, 2),
        "gas_fees": round(portfolio.position["total_gas_fees"], 2),
        "eth_balance": round(final_eth, 6),
        "usdc_balance": round(final_usdc, 2),
        "price": round(final_price, 2),
    })

    print(f"[{strategy_name}] done – APR {apr:.2f} %, Net {net_profit:.2f} USDC, IL {classic_il:.2f} %")
