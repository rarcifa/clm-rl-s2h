"""
run_heuristic.py — Simulation of Heuristic LP Strategies on Uniswap V3

This script simulates deterministic liquidity provisioning strategies based on
price deviation or volatility conditions. It evaluates a 1-year ETH/USDC scenario
using historical prices and volumes and logs metrics for APR, impermanent loss,
LP fees, and gas costs.

Trigger strategies include:
- "price" : Reposition when price exits the current range
- "vol"   : Reposition when 7-day coefficient of variation σ/μ exceeds 0.04
- "both"  : Not supported in current implementation (but logic allows extension)

Final metrics are saved in `results/seed_runs.csv`.
Full step-wise simulation is logged in the specified `csv_filename`.

Reference:
- Eq. (11) — σ/μ trigger for volatility rebalancing
- Eq. (13) — APR and IL formulas
- Section IV of Arcifa et al., IEEE DAPPS 2025

Author: Ricardo Arcifa  
Affiliation: IEEE DAPPS 2025
"""

import numpy as np
import csv
import random

from utils.liquidity import (
    amounts_in_position,
    compute_liquidity_L,
    cross_tick_and_update_liquidity,
    get_initial_liquidity_setup,
    update_tick_liquidity,
)
from utils.logging_helpers import record_summary
from utils.uniswap_math import (
    nearest_tick,
    price_to_tick,
    tick_to_price,
    tick_to_sqrt_price,
)


def run_heuristic(
    shocks,
    prices_eval,
    sqrt_prices_eval,
    pool_liquidity_eval,
    volumes_eval,
    strategy_name,                 # e.g. "heur_price"
    seed,
    trigger="price",               # "price" | "vol"
    steps=365,
    initial_investment=10000,
    reposition_gas_cost_range=(50, 150),
    protocol_fee_fraction=0.1,
    tick_spacing=10,
    initial_price=2200.0,
    csv_filename="heuristic_simulation.csv",
):
    """
    Simulate a one-year run of a deterministic liquidity strategy (CLM heuristic).

    Args:
        shocks (list): Daily signed price shocks.
        prices_eval (list): Daily ETH/USDC prices.
        sqrt_prices_eval (list): Precomputed sqrt prices (unused here).
        pool_liquidity_eval (list): Daily pool liquidity snapshots.
        volumes_eval (list): Daily trading volumes (USD).
        strategy_name (str): Identifier for this strategy run.
        seed (int): RNG seed used for positioning and gas sampling.
        trigger (str): Rebalancing mode: "price" or "vol".
        steps (int): Simulation length (default 365).
        initial_investment (float): Total USDC invested at start.
        reposition_gas_cost_range (tuple): Range for gas cost sampling (in USD).
        protocol_fee_fraction (float): Portion of fees taken by protocol.
        tick_spacing (int): Granularity of tick movement.
        initial_price (float): Starting ETH/USDC price.
        csv_filename (str): Output file for full stepwise log.
    """
    # Initial setup (wallet, position, ticks)
    init = get_initial_liquidity_setup(initial_investment, initial_price, tick_spacing)
    eth_balance = init["eth_balance"]
    usdc_balance = init["usdc_balance"]

    tick_liquidity = {}
    update_tick_liquidity(tick_liquidity, init["lower_tick"], init["liquidity"])
    update_tick_liquidity(tick_liquidity, init["upper_tick"], -init["liquidity"])

    position = {
        "lower_tick": init["lower_tick"],
        "upper_tick": init["upper_tick"],
        "liquidity": init["liquidity"],
        "active": True,
        "total_repositions": 0,
        "total_gas_fees": 0.0,
        "out_of_range_steps": 0,
    }

    cumulative_lp_fees = 0.0
    cumulative_protocol_fees = 0.0

    # Volatility window for σ/μ trigger (Eq. 11)
    price_window = []
    VOL_WINDOW_LEN = 7
    SIGMA_THRESHOLD = 0.04  # 4% CV threshold for volatility trigger

    # CSV writer for per-day trajectory
    fieldnames = [
        "step", "price", "active", "liquidity",
        "eth_in_wallet", "usdc_in_wallet", "eth_in_pool", "usdc_in_pool",
        "total_value", "lp_fees", "protocol_fees",
        "cumulative_lp_fees", "cumulative_protocol_fees",
        "gas_fees_step", "total_gas_fees", "action",
    ]
    writer = csv.DictWriter(open(csv_filename, "w", newline=""), fieldnames)
    writer.writeheader()

    # Daily simulation loop
    for step_i in range(steps):
        price = float(prices_eval[step_i])
        volume = float(volumes_eval[step_i])
        shock = float(shocks[step_i])
        direction = "up" if shock > 0 else "down"

        # Tick crossing: adjust L if price hits range boundary
        current_tick = price_to_tick(price)
        if current_tick in tick_liquidity:
            dL = cross_tick_and_update_liquidity(tick_liquidity, current_tick, direction)
            position["liquidity"] += dL

        # Determine if current price is in range
        p_lower = tick_to_price(position["lower_tick"])
        p_upper = tick_to_price(position["upper_tick"])
        currently_active = p_lower <= price <= p_upper
        price_trigger = position["active"] and (not currently_active)

        # Maintain 7-day rolling window for volatility calculation
        price_window.append(price)
        if len(price_window) > VOL_WINDOW_LEN:
            price_window.pop(0)

        if len(price_window) == VOL_WINDOW_LEN:
            sigma = np.std(price_window) / np.mean(price_window)
            vol_trigger = sigma > SIGMA_THRESHOLD
        else:
            vol_trigger = False

        # Decide whether to reposition based on configured trigger
        if trigger == "price":
            need_reposition = price_trigger
        elif trigger == "vol":
            need_reposition = vol_trigger
        else:
            raise ValueError(f"Unknown trigger type: {trigger}")

        # Reposition logig
        gas_fees_step = 0.0
        actions = []

        if need_reposition:
            position["out_of_range_steps"] += 1
            actions.append("Reposition")

            gas_fees_step = random.uniform(*reposition_gas_cost_range)
            position["total_gas_fees"] += gas_fees_step

            # Withdraw old range
            old_L = position["liquidity"]
            out0, out1 = amounts_in_position(old_L, price, p_lower, p_upper)
            eth_balance += out0
            usdc_balance += out1
            update_tick_liquidity(tick_liquidity, position["lower_tick"], -old_L)
            update_tick_liquidity(tick_liquidity, position["upper_tick"], old_L)

            # Choose new ±4–8% range centered on current price
            band = random.uniform(0.04, 0.08) * price
            new_lower_tick = nearest_tick(price - band, tick_spacing)
            new_upper_tick = nearest_tick(price + band, tick_spacing)

            new_L = compute_liquidity_L(
                eth_balance,
                usdc_balance,
                tick_to_sqrt_price(new_lower_tick),
                tick_to_sqrt_price(new_upper_tick),
            )

            need0, need1 = amounts_in_position(
                new_L,
                price,
                tick_to_price(new_lower_tick),
                tick_to_price(new_upper_tick),
            )

            eth_balance -= need0
            usdc_balance -= need1

            update_tick_liquidity(tick_liquidity, new_lower_tick, new_L)
            update_tick_liquidity(tick_liquidity, new_upper_tick, -new_L)

            position.update(
                lower_tick=new_lower_tick,
                upper_tick=new_upper_tick,
                liquidity=new_L,
                total_repositions=position["total_repositions"] + 1,
            )

        # Fee logic: based on shock amplitude (Section IV-D)
        if abs(shock) < 0.0015:
            fee_tier = 0.0005
        elif abs(shock) < 0.003:
            fee_tier = 0.003
        else:
            fee_tier = 0.01

        total_fee = volume * fee_tier
        protocol_fee = total_fee * protocol_fee_fraction

        if position["active"]:
            pool_L = float(pool_liquidity_eval[step_i])
            share = position["liquidity"] / pool_L if pool_L > 0 else 0.0
            lp_fees = (total_fee - protocol_fee) * share
        else:
            lp_fees = 0.0

        usdc_balance += lp_fees
        cumulative_lp_fees += lp_fees
        cumulative_protocol_fees += protocol_fee

        # Portfolio mark-to-market
        pool0, pool1 = amounts_in_position(position["liquidity"], price, p_lower, p_upper)
        wallet_value = eth_balance * price + usdc_balance
        pool_value = pool0 * price + pool1
        total_value = wallet_value + pool_value

        # Log this timestep to CSV
        writer.writerow(
            dict(
                step=step_i,
                price=round(price, 2),
                active=position["active"],
                liquidity=round(position["liquidity"], 2),
                eth_in_wallet=round(eth_balance, 4),
                usdc_in_wallet=round(usdc_balance, 2),
                eth_in_pool=round(pool0, 4),
                usdc_in_pool=round(pool1, 2),
                total_value=round(total_value, 2),
                lp_fees=round(lp_fees, 2),
                protocol_fees=round(protocol_fee, 2),
                cumulative_lp_fees=round(cumulative_lp_fees, 2),
                cumulative_protocol_fees=round(cumulative_protocol_fees, 2),
                gas_fees_step=round(gas_fees_step, 2),
                total_gas_fees=round(position["total_gas_fees"], 2),
                action="; ".join(actions) if actions else "no action",
            )
        )

        # Update position's active flag for next iteration
        position["active"] = currently_active

    # Final metrics (APR, IL) → Eq. (13)
    final_price = prices_eval[-1]
    pool0, pool1 = amounts_in_position(
        position["liquidity"],
        final_price,
        tick_to_price(position["lower_tick"]),
        tick_to_price(position["upper_tick"]),
    )
    final_eth = eth_balance + pool0
    final_usdc = usdc_balance + pool1

    gross_final_val = final_eth * final_price + final_usdc
    price_only_val = gross_final_val - cumulative_lp_fees

    hodl_eth = (initial_investment / 2) / initial_price
    hodl_val = hodl_eth * final_price + (initial_investment / 2)

    classic_il_usd = price_only_val - hodl_val
    classic_il = (classic_il_usd / hodl_val) * 100

    net_profit = gross_final_val - position["total_gas_fees"] - initial_investment
    apr = (net_profit / initial_investment) * 100

    record_summary(
        "results/seed_runs.csv",
        dict(
            seed=seed,
            strategy=strategy_name,
            volatility_regime="real",
            apr=round(apr, 4),
            impermanent_loss=round(classic_il, 2),
            lp_fees=round(cumulative_lp_fees, 2),
            gas_fees=round(position["total_gas_fees"], 2),
            eth_balance=round(final_eth, 6),
            usdc_balance=round(final_usdc, 2),
            pool_eth=round(pool0, 6),
            pool_usdc=round(pool1, 2),
            price=round(final_price, 2),
        ),
    )

    print(
        f"[{strategy_name}] done – APR {apr:5.2f} %, "
        f"net {net_profit:,.2f} USDC, IL {classic_il:,.2f} USDC"
    )
