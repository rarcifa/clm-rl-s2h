"""
portfolio_base.py — Shared Portfolio Logic for Uniswap V3 LP Simulations

This module defines a base class encapsulating all state and methods for managing
an LP portfolio, including balances, tick bands, liquidity, fee calculation, and
portfolio value computation. Used by both RL environments and heuristic simulations.

Author: Ricardo Arcifa  
Affiliation: IEEE DAPPS 2025
"""

from utils.core import (
    tick_to_price,
    tick_to_sqrt_price,
    price_to_tick,
    nearest_tick,
    compute_liquidity_L,
    amounts_in_position,
    update_tick_liquidity,
    cross_tick_and_update_liquidity
)
import numpy as np
import random

class UniswapV3PortfolioBase:
    """
    Base class for Uniswap V3 LP portfolio management.
    Encapsulates state and methods for balances, tick bands, liquidity, fee calculation, and value computation.
    """
    def __init__(self, initial_investment, initial_price, tick_spacing=10, protocol_fee_fraction=0.1, reposition_gas_cost_range=(50, 150)):
        self.initial_investment = initial_investment
        self.initial_price = initial_price
        self.tick_spacing = tick_spacing
        self.protocol_fee_fraction = protocol_fee_fraction
        self.reposition_gas_cost_range = reposition_gas_cost_range
        self.reset_portfolio()

    def reset_portfolio(self):
        """
        Initialize or reset portfolio state: balances, ticks, liquidity, and position.
        """
        self.eth_balance = (self.initial_investment / 2) / self.initial_price
        self.usdc_balance = self.initial_investment / 2
        self.tick_liquidity = {}
        self.cumulative_lp_fees = 0.0
        self.cumulative_protocol_fees = 0.0
        self.position = None
        self._init_position()

    def _init_position(self):
        """
        Initialize a symmetric tick band and deposit liquidity.
        """
        lower_tick = nearest_tick(self.initial_price * 0.95, self.tick_spacing)
        upper_tick = nearest_tick(self.initial_price * 1.05, self.tick_spacing)
        lower_sqrt = tick_to_sqrt_price(lower_tick)
        upper_sqrt = tick_to_sqrt_price(upper_tick)
        L = compute_liquidity_L(self.eth_balance, self.usdc_balance, lower_sqrt, upper_sqrt)
        need_eth, need_usdc = amounts_in_position(L, self.initial_price, tick_to_price(lower_tick), tick_to_price(upper_tick))
        self.eth_balance -= need_eth
        self.usdc_balance -= need_usdc
        update_tick_liquidity(self.tick_liquidity, lower_tick, L)
        update_tick_liquidity(self.tick_liquidity, upper_tick, -L)
        self.position = {
            "lower_tick": lower_tick,
            "upper_tick": upper_tick,
            "liquidity": L,
            "active": True,
            "out_of_range_steps": 0,
            "total_repositions": 0,
            "total_gas_fees": 0.0
        }

    def reposition(self, price, band=None):
        """
        Exit current position and enter a new tick band around the given price.
        Optionally specify band width (as a fraction of price).
        Returns the gas fee incurred.
        """
        if band is None:
            band = random.uniform(0.04, 0.08) * price
        gas_fee = random.uniform(*self.reposition_gas_cost_range)
        self.position["total_gas_fees"] += gas_fee
        # Exit old position
        old_L = self.position["liquidity"]
        p_lower = tick_to_price(self.position["lower_tick"])
        p_upper = tick_to_price(self.position["upper_tick"])
        out0, out1 = amounts_in_position(old_L, price, p_lower, p_upper)
        self.eth_balance += out0
        self.usdc_balance += out1
        update_tick_liquidity(self.tick_liquidity, self.position["lower_tick"], -old_L)
        update_tick_liquidity(self.tick_liquidity, self.position["upper_tick"], old_L)
        # Enter new band
        new_lower_tick = nearest_tick(price - band, self.tick_spacing)
        new_upper_tick = nearest_tick(price + band, self.tick_spacing)
        new_L = compute_liquidity_L(
            self.eth_balance, self.usdc_balance,
            tick_to_sqrt_price(new_lower_tick),
            tick_to_sqrt_price(new_upper_tick)
        )
        need0, need1 = amounts_in_position(
            new_L, price,
            tick_to_price(new_lower_tick),
            tick_to_price(new_upper_tick)
        )
        self.eth_balance -= need0
        self.usdc_balance -= need1
        update_tick_liquidity(self.tick_liquidity, new_lower_tick, new_L)
        update_tick_liquidity(self.tick_liquidity, new_upper_tick, -new_L)
        self.position.update({
            "lower_tick": new_lower_tick,
            "upper_tick": new_upper_tick,
            "liquidity": new_L,
            "total_repositions": self.position["total_repositions"] + 1
        })
        return gas_fee

    def cross_tick(self, price, shock):
        """
        Simulate tick crossing and update liquidity accordingly.
        """
        current_tick = price_to_tick(price)
        direction = "up" if shock > 0 else "down"
        dL = cross_tick_and_update_liquidity(self.tick_liquidity, current_tick, direction)
        self.position["liquidity"] += dL

    def compute_fees(self, volume, shock):
        """
        Compute LP and protocol fees for the current step, update balances.
        Returns (lp_fees, protocol_fee, fee_tier).
        """
        if abs(shock) < 0.0015:
            fee_tier = 0.0005
        elif abs(shock) < 0.003:
            fee_tier = 0.003
        else:
            fee_tier = 0.01
        total_fee = volume * fee_tier
        protocol_fee = total_fee * self.protocol_fee_fraction
        lp_fees = 0.0
        if self.position["active"]:
            # pool_L must be set externally before calling this
            pool_L = getattr(self, "pool_liquidity", 1.0)
            share = self.position["liquidity"] / pool_L if pool_L > 0 else 0.0
            lp_fees = (total_fee - protocol_fee) * share
        self.usdc_balance += lp_fees
        self.cumulative_lp_fees += lp_fees
        self.cumulative_protocol_fees += protocol_fee
        return lp_fees, protocol_fee, fee_tier

    def compute_portfolio_value(self, price):
        """
        Compute the total portfolio value (wallet + in-pool) at the given price.
        """
        p_lower = tick_to_price(self.position["lower_tick"])
        p_upper = tick_to_price(self.position["upper_tick"])
        pool0, pool1 = amounts_in_position(self.position["liquidity"], price, p_lower, p_upper)
        total_val = self.eth_balance * price + self.usdc_balance + pool0 * price + pool1
        return total_val, pool0, pool1 