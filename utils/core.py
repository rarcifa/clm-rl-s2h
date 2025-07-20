"""
core.py — Core Utility Functions for Uniswap V3 LP Simulations

This module provides reusable helper functions for price/tick conversions,
liquidity math, tick state updates, and synthetic scenario generation.
It is used by both training and evaluation environments.

Author: Ricardo Arcifa  
Affiliation: IEEE DAPPS 2025
"""

import os
import csv
import numpy as np


# --- CSV Summary Logging ---

def record_summary(path: str, rowdict: dict) -> None:
    """
    Append a summary row to a CSV file, creating the file with a header if needed.

    Args:
        path (str): Output file path.
        rowdict (dict): Dictionary of column names and values.
    """
    header = sorted(rowdict.keys())
    new_file = not os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if new_file:
            writer.writeheader()
        writer.writerow(rowdict)


# --- Scenario Generator ---

def generate_scenario_arrays(
    steps: int = 365,
    initial_price: float = 2200.0,
    mu: float = 0.0001,
    sigma: float = 0.06
):
    """
    Generate a synthetic ETH/USDC scenario using geometric Brownian motion.

    Args:
        steps (int): Number of days to simulate.
        initial_price (float): Starting ETH price.
        mu (float): Drift coefficient.
        sigma (float): Volatility.

    Returns:
        tuple: (shocks, prices, sqrt_prices, volumes)
    """
    np.random.seed()
    shock_array = np.random.normal(mu, sigma, size=steps)
    prices_eval = np.zeros(steps)
    sqrt_prices_eval = np.zeros(steps)
    volumes_eval = np.zeros(steps)

    current_price = initial_price
    prices_eval[0] = current_price
    sqrt_prices_eval[0] = np.sqrt(current_price)
    volumes_eval[0] = np.random.uniform(1_000_000, 5_000_000)

    for i in range(1, steps):
        current_price *= np.exp(shock_array[i] / 2.0)
        prices_eval[i] = current_price
        sqrt_prices_eval[i] = np.sqrt(current_price)
        volumes_eval[i] = np.random.uniform(1_000_000, 5_000_000)

    return shock_array, prices_eval, sqrt_prices_eval, volumes_eval


# --- Price ↔ Tick Conversions ---

def price_to_tick(price: float) -> int:
    """
    Convert price to the nearest tick index.

    Args:
        price (float): Spot price.

    Returns:
        int: Uniswap v3 tick index.
    """
    return int(np.log(price) / np.log(1.0001))


def tick_to_price(tick: int) -> float:
    """
    Convert tick index to a price.

    Args:
        tick (int): Uniswap v3 tick.

    Returns:
        float: Corresponding price.
    """
    return 1.0001 ** tick


def tick_to_sqrt_price(tick: int) -> float:
    """
    Convert tick index to sqrt(price).

    Args:
        tick (int): Tick index.

    Returns:
        float: Square root of price.
    """
    return np.sqrt(tick_to_price(tick))


def nearest_tick(price: float, spacing: int) -> int:
    """
    Snap a price to the nearest valid tick index aligned to spacing.

    Args:
        price (float): Spot price.
        spacing (int): Tick spacing (e.g., 10).

    Returns:
        int: Nearest tick aligned to spacing.
    """
    tick = price_to_tick(price)
    return tick - (tick % spacing)


# --- Liquidity Computation ---

def compute_liquidity_L(
    eth_amount: float,
    usdc_amount: float,
    lower_sqrt_price: float,
    upper_sqrt_price: float
) -> float:
    """
    Calculate liquidity (L) based on ETH, USDC, and price range.

    Args:
        eth_amount (float): Available ETH.
        usdc_amount (float): Available USDC.
        lower_sqrt_price (float): Lower sqrt(P) boundary.
        upper_sqrt_price (float): Upper sqrt(P) boundary.

    Returns:
        float: Max liquidity that can be deployed.
    """
    L1 = eth_amount * (lower_sqrt_price * upper_sqrt_price) / (upper_sqrt_price - lower_sqrt_price)
    L2 = usdc_amount / (upper_sqrt_price - lower_sqrt_price)
    return min(L1, L2)


def amounts_in_position(
    L: float,
    p: float,
    p_lower: float,
    p_upper: float,
    token0_is_eth: bool = True
) -> tuple:
    """
    Compute token amounts in an LP position given price and range.

    Args:
        L (float): Liquidity.
        p (float): Spot price.
        p_lower (float): Lower bound of price range.
        p_upper (float): Upper bound of price range.
        token0_is_eth (bool): Whether token0 is ETH. Defaults to True.

    Returns:
        tuple: (amount_token0, amount_token1)
    """
    sqrt_p = np.sqrt(p)
    sqrt_lower = np.sqrt(p_lower)
    sqrt_upper = np.sqrt(p_upper)

    if p <= p_lower:
        amt0 = L * (1/sqrt_lower - 1/sqrt_upper)
        amt1 = 0.0
    elif p >= p_upper:
        amt0 = 0.0
        amt1 = L * (sqrt_upper - sqrt_lower)
    else:
        amt0 = L * (1/sqrt_p - 1/sqrt_upper)
        amt1 = L * (sqrt_p - sqrt_lower)

    return max(amt0, 0.0), max(amt1, 0.0)


# --- Tick Liquidity Management ---

def update_tick_liquidity(tick_dict: dict, tick: int, delta_L: float) -> None:
    """
    Update the net liquidity delta at a specific tick.

    Args:
        tick_dict (dict): Tick → liquidity delta mapping.
        tick (int): Tick index.
        delta_L (float): Liquidity change (positive or negative).
    """
    if tick not in tick_dict:
        tick_dict[tick] = 0.0
    tick_dict[tick] += delta_L
    if abs(tick_dict[tick]) < 1e-12:
        del tick_dict[tick]


def cross_tick_and_update_liquidity(tick_dict: dict, current_tick: int, direction: str) -> float:
    """
    Simulate liquidity crossing a tick, adjusting based on direction.

    Args:
        tick_dict (dict): Tick → liquidity delta mapping.
        current_tick (int): Tick being crossed.
        direction (str): 'up' or 'down'.

    Returns:
        float: Liquidity delta to apply to position.
    """
    delta_L = 0.0
    if current_tick in tick_dict:
        delta_L = tick_dict[current_tick]
        if direction == "up":
            tick_dict[current_tick] = 0.0
    return delta_L
