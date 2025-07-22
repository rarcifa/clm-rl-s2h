"""
scenario.py — Synthetic Scenario Generator for RL Training

Generates synthetic price, volume, and volatility paths for training
concentrated liquidity management (CLM) agents on Uniswap v3. Follows a
geometric Brownian motion (GBM) process for price simulation.

Price evolution is based on:
    Eq. (4) from the paper:
        dPₜ = μPₜdt + σPₜdWₜ
    → Pₜ₊₁ = Pₜ * exp(εₜ / 2), where εₜ ~ N(μ, σ)

Volume is drawn daily from a uniform range.

Used in:
- UniswapV3TrainEnv

Author: Ricardo Arcifa  
Affiliation: IEEE DAPPS 2025
"""

import numpy as np


def generate_scenario_arrays(steps=365, initial_price=2200.0, mu=0.0001, sigma=0.06):
    """
    Generate a synthetic scenario using GBM price paths and random volume.

    Args:
        steps (int): Number of simulation steps (e.g., 365 days).
        initial_price (float): Starting ETH/USDC price.
        mu (float): Drift parameter of GBM (expected return).
        sigma (float): Volatility parameter of GBM.

    Returns:
        tuple:
            shock_array (np.ndarray): Daily GBM shocks εₜ ~ N(μ, σ)
            prices_eval (np.ndarray): Daily ETH/USDC prices simulated with GBM
            sqrt_prices_eval (np.ndarray): Square roots of each daily price
            volumes_eval (np.ndarray): Simulated daily trade volumes in USD
    """
    np.random.seed()  # ensure non-deterministic sampling across episodes

    shock_array = np.random.normal(mu, sigma, size=steps)  # εₜ shocks
    prices_eval = np.zeros(steps)
    sqrt_prices_eval = np.zeros(steps)
    volumes_eval = np.zeros(steps)

    # Initialize day 0
    current_price = initial_price
    prices_eval[0] = current_price
    sqrt_prices_eval[0] = np.sqrt(current_price)
    volumes_eval[0] = np.random.uniform(1_000_000, 5_000_000)

    # Simulate the next T days
    for i in range(1, steps):
        # Apply GBM: Pₜ = Pₜ₋₁ * exp(εₜ / 2)
        current_price *= np.exp(shock_array[i] / 2.0)
        prices_eval[i] = current_price
        sqrt_prices_eval[i] = np.sqrt(current_price)

        # Uniform random volume in USD
        volumes_eval[i] = np.random.uniform(1_000_000, 5_000_000)

    return shock_array, prices_eval, sqrt_prices_eval, volumes_eval
