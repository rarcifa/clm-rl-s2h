"""
UniswapV3TrainEnv - Training Environment for Reinforcement Learning on Uniswap v3 LP Management

This module defines a custom Gym-compatible environment that simulates synthetic ETH/USDC price scenarios
for training RL agents to manage liquidity positions in Uniswap v3. The agent interacts with tick-based
liquidity, gas costs, and fee mechanics to learn optimal LP repositioning behavior.

Author: Ricardo Arcifa  
Affiliation: IEEE DAPPS 2025
"""

import numpy as np
import csv
import random
from gym import Env, spaces

from utils.core import (
    generate_scenario_arrays,
)
from utils.portfolio_base import UniswapV3PortfolioBase


class UniswapV3TrainEnv(UniswapV3PortfolioBase, Env):
    """
    Gym environment for training reinforcement learning agents
    on Uniswap v3 LP strategies using synthetic ETH/USDC data.

    Attributes:
        initial_investment (float): Starting USDC-equivalent capital.
        reposition_gas_cost_range (tuple): (min, max) gas cost per reposition.
        protocol_fee_fraction (float): Portion of fees sent to protocol.
        tick_spacing (int): Minimum tick granularity.
        steps (int): Number of timesteps in the simulation.
        mu (float): Drift parameter of the geometric Brownian motion.
        sigma (float): Volatility parameter of the price process.
        enable_logging (bool): Whether to log step-level data.
        csv_filename (str): Output log file path.
    """

    def __init__(
        self,
        initial_investment=10_000,
        reposition_gas_cost_range=(10, 50),
        protocol_fee_fraction=0.1,
        tick_spacing=10,
        steps=365,
        initial_price=2200.0,
        enable_logging=False,
        csv_filename="ppo_train_log.csv",
        mu=0.0001,
        sigma=0.06
    ):
        UniswapV3PortfolioBase.__init__(
            self,
            initial_investment=initial_investment,
            initial_price=initial_price,
            tick_spacing=tick_spacing,
            protocol_fee_fraction=protocol_fee_fraction,
            reposition_gas_cost_range=reposition_gas_cost_range
        )
        Env.__init__(self)
        self.max_steps = steps
        self.mu = mu
        self.sigma = sigma
        self.enable_logging = enable_logging
        self.csv_filename = csv_filename
        self.csv_file = None
        self.csv_writer = None
        self.shocks = None
        self.prices = None
        self.sqrt_prices = None
        self.volumes = None
        self.current_step = 0
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([1e9, 365.0, 1e12, 1e9, 1e8]),
            shape=(5,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
        if self.enable_logging:
            self.csv_file = open(self.csv_filename, mode='w', newline='')
            fieldnames = [
                'step', 'price', 'liquidity_range', 'active', 'liquidity',
                'eth_balance', 'usdc_balance', 'pool_eth', 'pool_usdc',
                'total_value', 'lp_fees', 'protocol_fees',
                'cumulative_lp_fees', 'cumulative_protocol_fees',
                'gas_fees_step', 'total_gas_fees', 'action', 'reward',
                'out_of_range_steps'
            ]
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
            self.csv_writer.writeheader()

    def _initialize_episode_scenario(self):
        self.shocks, self.prices, self.sqrt_prices, self.volumes = generate_scenario_arrays(
            steps=self.max_steps,
            initial_price=self.initial_price,
            mu=self.mu,
            sigma=self.sigma
        )

    def reset(self):
        self._initialize_episode_scenario()
        self.current_step = 0
        self.reset_portfolio()
        return self._get_obs()

    def _get_obs(self):
        step = min(self.current_step, self.max_steps - 1)
        return np.array([
            float(self.prices[step]),
            float(self.position["out_of_range_steps"]),
            float(self.position["liquidity"]),
            float(self.volumes[step]),
            float(self.cumulative_lp_fees)
        ], dtype=np.float32)

    def close(self):
        if self.csv_file and not self.csv_file.closed:
            self.csv_file.close()
            print("Train CSV file closed.")
        super().close()

    def step(self, action):
        idx = self.current_step
        price = self.prices[idx]
        vol = self.volumes[idx]
        shock = self.shocks[idx]
        old_value, _, _ = self.compute_portfolio_value(price)
        self.cross_tick(price, shock)
        gas_fee = 0.0
        action_taken = "no-op"
        if action > 0:
            gas_fee = random.uniform(*self.reposition_gas_cost_range) * 0.5
            self.position["total_gas_fees"] += gas_fee
            self.reposition(price)
            action_taken = "reposition"
        # Fee calculation
        pool_L = 1.0  # Placeholder, should be set to current pool liquidity
        self.pool_liquidity = pool_L
        lp_fees, protocol_fee, _ = self.compute_fees(vol, shock)
        new_value, pool0, pool1 = self.compute_portfolio_value(price)
        reward = new_value - old_value - gas_fee
        self.current_step += 1
        done = self.current_step >= self.max_steps
        obs = self._get_obs()
        info = {}
        if self.enable_logging:
            self.csv_writer.writerow({
                'step': idx,
                'price': price,
                'liquidity_range': f"{self.position['lower_tick']}-{self.position['upper_tick']}",
                'active': self.position['active'],
                'liquidity': self.position['liquidity'],
                'eth_balance': self.eth_balance,
                'usdc_balance': self.usdc_balance,
                'pool_eth': pool0,
                'pool_usdc': pool1,
                'total_value': new_value,
                'lp_fees': lp_fees,
                'protocol_fees': protocol_fee,
                'cumulative_lp_fees': self.cumulative_lp_fees,
                'cumulative_protocol_fees': self.cumulative_protocol_fees,
                'gas_fees_step': gas_fee,
                'total_gas_fees': self.position['total_gas_fees'],
                'action': action_taken,
                'reward': reward,
                'out_of_range_steps': self.position['out_of_range_steps']
            })
        return obs, reward, done, info
