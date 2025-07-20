"""
UniswapV3EvalEnv - Evaluation Environment for RL Agents and Heuristic Strategies

This module defines a Gym-compatible environment for evaluating policy performance on a fixed
ETH/USDC scenario using historical price, volume, and liquidity time series data. It supports 
continuous tracking of LP rewards, impermanent loss, gas fees, and overall portfolio performance.

Author: Ricardo Arcifa  
Affiliation: IEEE DAPPS 2025
"""

import numpy as np  # Core numerical operations
import csv          # Step-by-step result logging
from gym import Env, spaces  # OpenAI Gym interface for custom environments
from utils.portfolio_base import UniswapV3PortfolioBase


class UniswapV3EvalEnv(UniswapV3PortfolioBase, Env):
    """
    Evaluation environment for Uniswap V3 LP strategies using ETH/USDC data.

    This Gym-compatible environment is used to evaluate reinforcement learning (PPO, DQN) 
    or heuristic LP strategies against a deterministic ETH/USDC market scenario. The 
    environment simulates LP fee collection, impermanent loss, gas cost, and capital 
    allocation dynamics with configurable tick spacing and logging.

    Attributes:
        shocks (np.ndarray): Daily price shocks (optional, unused directly here).
        prices (np.ndarray): Daily ETH/USDC prices.
        sqrt_prices (np.ndarray): sqrt(price) values for Uniswap math.
        volumes (np.ndarray): Daily trading volume data.
        pool_liquidity (np.ndarray): Daily total liquidity in the pool.
        strategy_name (str): Identifier of the strategy being evaluated.
        initial_investment (float): Capital (in USDC) to start with.
        reposition_gas_cost_range (tuple): (min, max) gas units required for reposition.
        protocol_fee_fraction (float): Portion of fee routed to the protocol treasury.
        tick_spacing (int): Tick granularity for Uniswap range positioning.
        enable_logging (bool): Whether to log each step into a CSV file.
        csv_filename (str): Name of the CSV file for evaluation logging.
        observation_space (spaces.Box): 5D continuous observation space.
        action_space (spaces.Discrete): Discrete action space (e.g., 3 choices).
    """

    def __init__(
        self,
        shocks_eval,
        prices_eval,
        sqrt_prices_eval,
        volumes_eval,
        pool_liquidity_eval,
        strategy_name,
        initial_investment=10000,
        reposition_gas_cost_range=(50, 150),
        protocol_fee_fraction=0.1,
        tick_spacing=10,
        enable_logging=False,
        csv_filename="ppo_eval_log.csv"
    ):
        UniswapV3PortfolioBase.__init__(
            self,
            initial_investment=initial_investment,
            initial_price=prices_eval[0],
            tick_spacing=tick_spacing,
            protocol_fee_fraction=protocol_fee_fraction,
            reposition_gas_cost_range=reposition_gas_cost_range
        )
        Env.__init__(self)
        self.shocks = shocks_eval
        self.prices = prices_eval
        self.sqrt_prices = sqrt_prices_eval
        self.volumes = volumes_eval
        self.pool_liquidity_arr = pool_liquidity_eval
        self.strategy_name = strategy_name
        self.enable_logging = enable_logging
        self.csv_filename = csv_filename
        self.csv_file = None
        self.csv_writer = None
        self.max_steps = len(self.prices)
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

    def _get_obs(self):
        idx = min(self.current_step, self.max_steps - 1)
        return np.array([
            float(self.prices[idx]),
            float(self.position["out_of_range_steps"]),
            float(self.position["liquidity"]),
            float(self.volumes[idx]),
            float(self.cumulative_lp_fees)
        ], dtype=np.float32)

    def close(self):
        if self.csv_file and not self.csv_file.closed:
            self.csv_file.close()
            print("Eval CSV file closed.")
        super().close()

    def reset(self):
        self.current_step = 0
        self.reset_portfolio()
        return self._get_obs()

    def step(self, action):
        idx = self.current_step
        price = self.prices[idx]
        volume = self.volumes[idx]
        shock = self.shocks[idx]
        self.pool_liquidity = self.pool_liquidity_arr[idx]
        old_value, _, _ = self.compute_portfolio_value(price)
        self.cross_tick(price, shock)
        gas_fee = 0.0
        action_taken = "no-op"
        if action > 0:
            gas_fee = self.reposition(price)
            action_taken = "reposition"
        lp_fees, protocol_fee, _ = self.compute_fees(volume, shock)
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
