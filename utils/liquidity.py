"""
liquidity.py — Utilities for Uniswap v3 Liquidity Management

This module provides functions to compute liquidity amounts, track liquidity 
at ticks, and simulate liquidity transitions as the price crosses ticks. It 
is used across both heuristic and RL environments for concentrated liquidity 
management (CLM).

Key formulas and references:
- Eq. (5): Liquidity computation based on token amounts and sqrt prices
- Section II-B/C: Uniswap v3 tick mechanics
- Eq. (9): Reward indirectly depends on these liquidity movements

Author: Ricardo Arcifa  
Affiliation: IEEE DAPPS 2025
"""

import numpy as np

from utils.uniswap_math import nearest_tick, tick_to_price, tick_to_sqrt_price


def compute_liquidity_L(eth_amount, usdc_amount, lower_sqrt_price, upper_sqrt_price):
    """
    Compute the maximum liquidity L that can be added to a Uniswap v3 position
    given token amounts and a price range.

    This implements Eq. (5) from the paper:
        L₁ = ETH * (√P_lower * √P_upper) / (√P_upper - √P_lower)
        L₂ = USDC / (√P_upper - √P_lower)
        L  = min(L₁, L₂)

    Args:
        eth_amount (float): Available ETH tokens.
        usdc_amount (float): Available USDC tokens.
        lower_sqrt_price (float): √P_lower, lower bound of price range.
        upper_sqrt_price (float): √P_upper, upper bound of price range.

    Returns:
        float: Maximum liquidity L that fits within the available tokens.
    """
    L1 = eth_amount * (lower_sqrt_price * upper_sqrt_price) / (upper_sqrt_price - lower_sqrt_price)
    L2 = usdc_amount / (upper_sqrt_price - lower_sqrt_price)
    return min(L1, L2)


def amounts_in_position(L, p, p_lower, p_upper, token0_is_eth=True):
    """
    Compute the amount of token0 (ETH) and token1 (USDC) held in the LP position
    at a given price, given a price range and liquidity.

    This is derived from Uniswap v3 liquidity math and used to compute wallet/pool balances.

    Args:
        L (float): Liquidity in position.
        p (float): Current ETH/USDC price.
        p_lower (float): Lower bound of range.
        p_upper (float): Upper bound of range.
        token0_is_eth (bool): If True, token0 = ETH and token1 = USDC.

    Returns:
        (float, float): (ETH amount, USDC amount) in the position.
    """
    sqrt_p = np.sqrt(p)
    sqrt_lower = np.sqrt(p_lower)
    sqrt_upper = np.sqrt(p_upper)

    if p <= p_lower:
        amt0 = L * (1 / sqrt_lower - 1 / sqrt_upper)
        amt1 = 0.0
    elif p >= p_upper:
        amt0 = 0.0
        amt1 = L * (sqrt_upper - sqrt_lower)
    else:
        amt0 = L * (1 / sqrt_p - 1 / sqrt_upper)
        amt1 = L * (sqrt_p - sqrt_lower)

    return max(amt0, 0.0), max(amt1, 0.0)


def update_tick_liquidity(tick_dict, tick, delta_L):
    """
    Adjust liquidity at a tick by delta_L. If resulting value is ≈0, remove it.

    Args:
        tick (int): Tick index to update.
        delta_L (float): Amount of liquidity to add/remove.
    """
    if tick not in tick_dict:
        tick_dict[tick] = 0.0
    tick_dict[tick] += delta_L
    if abs(tick_dict[tick]) < 1e-12:
        del tick_dict[tick]


def cross_tick_and_update_liquidity(tick_dict, current_tick, direction):
    """
    Simulate price crossing a tick and return the liquidity change associated with it.

    If a tick is crossed:
        - "up": liquidity is removed from the previous lower range
        - "down": liquidity is added back to the lower range

    Args:
        tick_dict (dict): Mapping of tick index to net liquidity.
        current_tick (int): Tick being crossed.
        direction (str): "up" or "down".

    Returns:
        float: Liquidity to add/remove from active pool.
    """
    delta_L = 0
    if current_tick in tick_dict:
        delta_L = tick_dict[current_tick]
        if direction == "up":
            tick_dict[current_tick] = 0  # remove from below
        else:
            tick_dict[current_tick] = delta_L  # re-add to below
    return delta_L


def get_initial_liquidity_setup(initial_investment, initial_price, tick_spacing=10):
    """
    Set up initial LP position and wallet balances using a ±5% price band.

    Assumes 50/50 split between ETH and USDC at the start. This is used to create
    a consistent and valid initial range for deterministic or stochastic simulations.

    Args:
        initial_investment (float): Total capital in USD.
        initial_price (float): ETH/USDC price at t=0.
        tick_spacing (int): Spacing between ticks (Uniswap v3 constraint).

    Returns:
        dict: {
            'lower_tick': int,
            'upper_tick': int,
            'liquidity': float,
            'eth_balance': float,
            'usdc_balance': float
        }
    """
    eth_balance = (initial_investment / 2) / initial_price
    usdc_balance = initial_investment / 2

    r1 = 0.95  # 5% below
    r2 = 1.05  # 5% above

    lower_tick = nearest_tick(initial_price * r1, tick_spacing)
    upper_tick = nearest_tick(initial_price * r2, tick_spacing)
    lower_sqrt = tick_to_sqrt_price(lower_tick)
    upper_sqrt = tick_to_sqrt_price(upper_tick)

    L = compute_liquidity_L(eth_balance, usdc_balance, lower_sqrt, upper_sqrt)

    needed_eth, needed_usdc = amounts_in_position(
        L, initial_price, tick_to_price(lower_tick), tick_to_price(upper_tick)
    )

    eth_balance -= needed_eth
    usdc_balance -= needed_usdc

    if eth_balance < 0 or usdc_balance < 0:
        raise ValueError("Insufficient balance for initial liquidity.")

    return {
        "lower_tick": lower_tick,
        "upper_tick": upper_tick,
        "liquidity": L,
        "eth_balance": eth_balance,
        "usdc_balance": usdc_balance,
    }
