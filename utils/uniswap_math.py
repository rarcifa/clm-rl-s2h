"""
uniswap_math.py — Tick & Price Conversion Utilities for Uniswap v3

Provides helper functions to convert between prices, ticks, and square root prices
according to Uniswap v3’s formula:

    P = 1.0001^tick

These conversions are critical for defining liquidity ranges and interacting
with tick-based structures.

Used in:
- Liquidity management and rebalancing
- Initialization of LP positions
- Pricing logic in training and evaluation environments

Reference:
- Uniswap v3 whitepaper: https://uniswap.org/whitepaper-v3.pdf
- Section II-A of Arcifa et al., IEEE DAPPS 2025

Author: Ricardo Arcifa  
Affiliation: IEEE DAPPS 2025
"""

import numpy as np


def price_to_tick(price):
    """
    Convert a price (P) to the closest tick value using Uniswap v3 formula:
        tick = log_base(1.0001)(price)

    Args:
        price (float): ETH/USDC price

    Returns:
        int: Corresponding integer tick value
    """
    return int(np.log(price) / np.log(1.0001))


def tick_to_price(tick):
    """
    Convert a tick to its corresponding price using:
        price = 1.0001 ^ tick

    Args:
        tick (int): Tick index

    Returns:
        float: Price corresponding to that tick
    """
    return 1.0001 ** tick


def tick_to_sqrt_price(tick):
    """
    Compute square root of price at a given tick:
        √P = sqrt(1.0001 ^ tick)

    Used in Uniswap v3 liquidity equations (e.g., Eq. (5)).

    Args:
        tick (int): Tick index

    Returns:
        float: Square root of price at that tick
    """
    return np.sqrt(tick_to_price(tick))


def nearest_tick(price, spacing):
    """
    Round a price to the nearest tick that aligns with given tick spacing.

    Ensures resulting tick is aligned to the spacing constraint, e.g. spacing=10.

    Args:
        price (float): ETH/USDC price
        spacing (int): Tick spacing (e.g. 1, 10, 60)

    Returns:
        int: Nearest aligned tick
    """
    t = price_to_tick(price)
    return t - (t % spacing)
