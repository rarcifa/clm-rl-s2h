"""
liquidity.py — Initial Liquidity Setup for Uniswap V3 Simulations

This module provides a helper function for allocating an initial liquidity position
based on a fixed ETH/USDC split and a 5% tick band around the starting price.

Used by: Heuristic simulations, environment resets, and reproducible scenario seeds.

Author: Ricardo Arcifa  
Affiliation: IEEE DAPPS 2025
"""

from .core import (
    nearest_tick,
    tick_to_price,
    tick_to_sqrt_price,
    compute_liquidity_L,
    amounts_in_position
)


# --- Initial Liquidity Construction ---

def get_initial_liquidity_setup(
    initial_investment: float,
    initial_price: float,
    tick_spacing: int = 10
) -> dict:
    """
    Set up an LP position using 50/50 ETH and USDC and a ±5% tick band.

    Args:
        initial_investment (float): Total capital in USDC-equivalent.
        initial_price (float): ETH/USDC price at initialization.
        tick_spacing (int): Tick spacing to snap the band to (e.g. 10).

    Returns:
        dict: {
            'lower_tick': Lower tick index,
            'upper_tick': Upper tick index,
            'liquidity': Liquidity L placed,
            'eth_balance': Remaining ETH after deposit,
            'usdc_balance': Remaining USDC after deposit
        }
    """
    # Split capital into ETH and USDC at current price
    eth_balance = (initial_investment / 2) / initial_price
    usdc_balance = initial_investment / 2

    # Choose ±5% price range
    r1 = 0.95
    r2 = 1.05

    lower_tick = nearest_tick(initial_price * r1, tick_spacing)
    upper_tick = nearest_tick(initial_price * r2, tick_spacing)
    lower_sqrt = tick_to_sqrt_price(lower_tick)
    upper_sqrt = tick_to_sqrt_price(upper_tick)

    # Compute liquidity and required token amounts
    L = compute_liquidity_L(eth_balance, usdc_balance, lower_sqrt, upper_sqrt)
    needed_eth, needed_usdc = amounts_in_position(
        L, initial_price,
        tick_to_price(lower_tick),
        tick_to_price(upper_tick)
    )

    # Subtract tokens used in liquidity deposit
    eth_balance -= needed_eth
    usdc_balance -= needed_usdc

    if eth_balance < 0 or usdc_balance < 0:
        raise ValueError("Insufficient balance for initial liquidity.")

    return {
        'lower_tick': lower_tick,
        'upper_tick': upper_tick,
        'liquidity': L,
        'eth_balance': eth_balance,
        'usdc_balance': usdc_balance
    }
