"""
train_env.py — Training Environment for RL Agents on Uniswap V3

This module defines `UniswapV3TrainEnv`, an OpenAI Gym-compatible environment
that simulates synthetic ETH/USDC price paths using geometric Brownian motion (GBM)
to train reinforcement learning agents for concentrated liquidity management (CLM) 
on Uniswap v3.

Key features:
- Randomized episodes via GBM-based prices, volumes, and shocks
- Tick-level liquidity accounting and LP fee modeling
- Fee tiers and gas costs based on volatility conditions
- Penalties for staying out of range, rewards for LP fees and portfolio growth

The environment supports RL training by exposing a standard interface:
- Observation: [price, out_of_range_steps, liquidity, volume, cumulative_lp_fees]
- Action: Discrete(3) => 0 = hold, >0 = reposition

Author: Ricardo Arcifa  
Affiliation: IEEE DAPPS 2025
"""

import csv
import random
from gym import Env, spaces
import numpy as np

from utils.liquidity import (
    amounts_in_position,
    compute_liquidity_L,
    cross_tick_and_update_liquidity,
    update_tick_liquidity,
)
from utils.scenario import generate_scenario_arrays
from utils.uniswap_math import (
    nearest_tick,
    price_to_tick,
    tick_to_price,
    tick_to_sqrt_price,
)


class UniswapV3TrainEnv(Env):
    """
    Gym-style environment for deterministic evaluation of Uniswap v3 LP strategies.

    Simulates a portfolio using historical ETH/USDC data including prices, volume,
    volatility shocks, and pool liquidity. Used to benchmark reinforcement learning
    policies and heuristics with consistent trajectories across seeds.

    Attributes:
        - action_space: Discrete(3) → 0 = hold, 1+ = reposition
        - observation_space: Box with [price, out_of_range_steps, liquidity, volume, cum_fees]
        - Internal tick-based LP accounting and gas-cost simulation
    """

    def __init__(
        self,
        initial_investment=10000,
        reposition_gas_cost_range=(10, 50),
        protocol_fee_fraction=0.1,
        tick_spacing=10,
        steps=365,
        initial_price=2200.0,
        enable_logging=False,
        csv_filename="ppo_train_log.csv",
        mu=0.0001,
        sigma=0.06,
        shocks=None,
        prices=None,
        sqrt_prices=None,
        volumes=None,
    ):
        """
        Initialize the training environment.

        Args:
            initial_investment (float): Starting portfolio value in USD.
            reposition_gas_cost_range (tuple): Range of gas cost per reposition (USD).
            protocol_fee_fraction (float): Portion of fees taken by protocol (default: 10%).
            tick_spacing (int): Granularity of tick spacing.
            steps (int): Length of each episode in timesteps.
            initial_price (float): Starting ETH/USDC price.
            enable_logging (bool): If True, logs data to CSV.
            csv_filename (str): Filename for logging training metrics.
            mu (float): Drift term for GBM price simulation.
            sigma (float): Volatility term for GBM.
            shocks, prices, sqrt_prices, volumes (optional): Override synthetic data.
        """
        super().__init__()
        self.initial_investment = initial_investment
        self.reposition_gas_cost_range = reposition_gas_cost_range
        self.protocol_fee_fraction = protocol_fee_fraction
        self.tick_spacing = tick_spacing
        self.max_steps = steps
        self.initial_price = initial_price
        self.mu = mu
        self.sigma = sigma
        self.enable_logging = enable_logging
        self.csv_filename = csv_filename
        self.shocks = shocks
        self.prices = prices
        self.sqrt_prices = sqrt_prices
        self.volumes = volumes
        self.current_step = 0

        # Tick-based liquidity map and fee records (per tick)
        self.tick_liquidity = {}
        self.tick_fees = {}

        # Portfolio balances
        self.position = None
        self.eth_balance = None
        self.usdc_balance = None

        # Cumulative metrics
        self.cumulative_lp_fees = 0.0
        self.cumulative_protocol_fees = 0.0

        # State: [price, out_of_range_steps, liquidity, volume, cumulative_lp_fees]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([1e9, 365.0, 1e12, 1e9, 1e8]),
            shape=(5,),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(3)
        
        # Logging config
        self.csv_file = None
        self.csv_writer = None
        if self.enable_logging:
            self.csv_file = open(self.csv_filename, mode="w", newline="")
            fieldnames = [
                "step",
                "price",
                "liquidity_range",
                "active",
                "liquidity",
                "eth_balance",
                "usdc_balance",
                "pool_eth",
                "pool_usdc",
                "total_value",
                "lp_fees",
                "protocol_fees",
                "cumulative_lp_fees",
                "cumulative_protocol_fees",
                "gas_fees_step",
                "total_gas_fees",
                "action",
                "reward",
                "out_of_range_steps",
            ]
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
            self.csv_writer.writeheader()

    def _initialize_episode_scenario(self):
        """
        Generate synthetic GBM-based scenario for price, volume, and volatility.

        These synthetic paths are sampled from:
            dPₜ = μPₜdt + σPₜdWₜ
        where Wₜ is Brownian motion. Volumes are similarly randomized.

        Based on Section IV-A of the paper.
        """
        shock_array, prices_eval, sqrt_prices_eval, volumes_eval = generate_scenario_arrays(
            steps=self.max_steps,
            initial_price=self.initial_price,
            mu=self.mu,
            sigma=self.sigma,
        )
        self.shocks = shock_array
        self.prices = prices_eval
        self.sqrt_prices = sqrt_prices_eval
        self.volumes = volumes_eval

    def _setup_liquidity_and_wallet(self):
        """
        Initialize wallet and LP position for a new episode.

        - 50% of initial capital goes to ETH, 50% to USDC
        - A fixed range ±5% around initial price is chosen
        - Liquidity is computed and supplied
        - Tick liquidity maps are updated
        """
        self.tick_liquidity = {}
        self.tick_fees = {}
        self.cumulative_lp_fees = 0.0
        self.cumulative_protocol_fees = 0.0

        self.position = {
            "lower_tick": None,
            "upper_tick": None,
            "out_of_range_steps": 0,
            "total_repositions": 0,
            "total_gas_fees": 0,
            "active": True,
            "liquidity": 0.0,
        }

        init_price = self.prices[0]
        self.eth_balance = (self.initial_investment / 2) / init_price
        self.usdc_balance = self.initial_investment / 2

        r1 = 0.95
        r2 = 1.05

        lower_tick = nearest_tick(init_price * r1, self.tick_spacing)
        upper_tick = nearest_tick(init_price * r2, self.tick_spacing)
        lower_sqrt = tick_to_sqrt_price(lower_tick)
        upper_sqrt = tick_to_sqrt_price(upper_tick)

        L = compute_liquidity_L(
            self.eth_balance,
            self.usdc_balance,
            lower_sqrt,
            upper_sqrt,
        )

        needed_eth, needed_usdc = amounts_in_position(
            L,
            init_price,
            tick_to_price(lower_tick),
            tick_to_price(upper_tick),
        )

        self.eth_balance -= needed_eth
        self.usdc_balance -= needed_usdc

        if self.eth_balance < 0 or self.usdc_balance < 0:
            raise ValueError("Not enough wallet to supply initial liquidity")

        self.position["lower_tick"] = lower_tick
        self.position["upper_tick"] = upper_tick
        self.position["liquidity"] = L
        self.position["active"] = True

        update_tick_liquidity(self.tick_liquidity, lower_tick, L)
        update_tick_liquidity(self.tick_liquidity, upper_tick, -L)

    def reset(self):
        """
        Reset the environment for a new training episode.

        Each call generates a new synthetic market scenario and LP state.
        """
        self._initialize_episode_scenario()
        self.current_step = 0
        self._setup_liquidity_and_wallet()
        return self._get_obs()

    def step(self, action):
        """
        Perform one step in the synthetic Uniswap V3 training environment.

        The agent interacts with the environment by choosing one of:
            - action = 0: Do nothing
            - action > 0: Reposition liquidity around current price with a random range

        This method:
        - Calculates LP fees (based on trading volume and fee tier)
        - Applies gas costs for repositioning
        - Penalizes staying out of range for too long
        - Rewards increases in mark-to-market portfolio value

        Reward is based on Equation (9) from Arcifa et al. (IEEE DAPPS 2025):

            Rₜ = (Vₜ - Vₜ₋₁) - λ_gas × gas_costₜ + α × LP_feeₜ - λ_oor × 𝟙{oor > 5} × Vₜ₋₁

        Returns:
            obs (np.ndarray): Updated observation after action
            reward (float): Reward signal based on fee, value change, and gas
            done (bool): Whether the episode has ended
            info (dict): Additional data (unused here)
        """
        idx = self.current_step
        price = self.prices[idx]
        vol = self.volumes[idx]

        # Compute old portfolio value before action
        old_pool_eth, old_pool_usdc = amounts_in_position(
            self.position["liquidity"],
            price,
            tick_to_price(self.position["lower_tick"]),
            tick_to_price(self.position["upper_tick"]),
        )
        old_portfolio_value = (
            self.eth_balance * price
            + self.usdc_balance
            + old_pool_eth * price + old_pool_usdc
        )

        # Tick crossing logic (adjust liquidity if price crosses into/out of range)
        current_tick = price_to_tick(price)
        direction = "up" if self.shocks[idx] > 0 else "down"
        if current_tick in self.tick_liquidity:
            dL = cross_tick_and_update_liquidity(self.tick_liquidity, current_tick, direction)
            self.position["liquidity"] += dL

        gas_fees_step = 0.0
        actions_done = []

        # Action: 0 = hold, 1/2 = reposition with a new random width band
        if action > 0:
            gas_fees_step = random.uniform(*self.reposition_gas_cost_range) * 0.5
            self.position["total_gas_fees"] += gas_fees_step

            # Remove liquidity from current ticks
            old_L = self.position["liquidity"]
            update_tick_liquidity(self.tick_liquidity, self.position["lower_tick"], -old_L)
            update_tick_liquidity(self.tick_liquidity, self.position["upper_tick"], old_L)

            # Get tokens back from old position
            out0, out1 = amounts_in_position(
                old_L,
                price,
                tick_to_price(self.position["lower_tick"]),
                tick_to_price(self.position["upper_tick"]),
            )
            self.eth_balance += out0
            self.usdc_balance += out1

            # Generate new random band centered around current price
            range_adj = random.uniform(0.04, 0.08) * price
            new_lower_tick = nearest_tick(price - range_adj, self.tick_spacing)
            new_upper_tick = nearest_tick(price + range_adj, self.tick_spacing)

            new_lower_sqrt = tick_to_sqrt_price(new_lower_tick)
            new_upper_sqrt = tick_to_sqrt_price(new_upper_tick)

            new_L = compute_liquidity_L(
                self.eth_balance,
                self.usdc_balance,
                new_lower_sqrt,
                new_upper_sqrt,
            )

            # Deduct token amounts for new position
            need0, need1 = amounts_in_position(
                new_L,
                price,
                tick_to_price(new_lower_tick),
                tick_to_price(new_upper_tick),
            )

            # Apply new liquidity to wallet and tick map
            self.eth_balance -= need0
            self.usdc_balance -= need1

            update_tick_liquidity(self.tick_liquidity, new_lower_tick, new_L)
            update_tick_liquidity(self.tick_liquidity, new_upper_tick, -new_L)

            self.position.update({
                "lower_tick": new_lower_tick,
                "upper_tick": new_upper_tick,
                "liquidity": new_L,
                "total_repositions": self.position["total_repositions"] + 1,
            })

            actions_done.append(f"Reposition => ticks({new_lower_tick},{new_upper_tick})")

        # Check if current price is within the LP range
        p_lower = tick_to_price(self.position["lower_tick"])
        p_upper = tick_to_price(self.position["upper_tick"])
        now_active = p_lower <= price <= p_upper
        self.position["active"] = now_active

        if not now_active:
            self.position["out_of_range_steps"] += 1
        else:
            self.position["out_of_range_steps"] = 0

        # Determine fee tier from shock size (paper Section IV-D)
        shock = abs(self.shocks[idx])
        if shock < 0.0015:
            fee_tier = 0.0005
        elif shock < 0.003:
            fee_tier = 0.003
        else:
            fee_tier = 0.01

        # Compute total fee and protocol cut
        total_fee = vol * fee_tier
        protocol_fee = total_fee * self.protocol_fee_fraction

        # LP earns a share of fees only if position is active
        if now_active:
            # For training, pool liquidity is normalized (1e6)
            share = self.position["liquidity"] / 1e6
            lp_fees = (total_fee - protocol_fee) * share
        else:
            lp_fees = 0.0

        # Accumulate LP and protocol fees
        self.usdc_balance += lp_fees
        self.cumulative_lp_fees += lp_fees
        self.cumulative_protocol_fees += protocol_fee

        # Compute new portfolio value (wallet + in-range position)
        new_pool_eth, new_pool_usdc = amounts_in_position(
            self.position["liquidity"],
            price,
            p_lower,
            p_upper,
        )
        new_portfolio_value = (
            self.eth_balance * price
            + self.usdc_balance
            + new_pool_eth * price + new_pool_usdc
        )

        # Compute reward (based on Equation (9))
        reward = (new_portfolio_value - old_portfolio_value) - gas_fees_step  # ΔV - gas

        if self.position["out_of_range_steps"] > 5:
            reward -= old_portfolio_value * 0.01  # penalty for prolonged OOR

        reward += lp_fees * 10  # amplified reward for LP earnings

        # Logging
        if self.enable_logging and self.csv_writer:
            self.csv_writer.writerow({
                'step': self.current_step,
                'price': round(price, 2),
                'liquidity_range': f"{round(p_lower, 2)} - {round(p_upper, 2)}",
                'active': now_active,
                'liquidity': round(self.position["liquidity"], 2),
                'eth_balance': round(self.eth_balance, 4),
                'usdc_balance': round(self.usdc_balance, 2),
                'pool_eth': round(new_pool_eth, 4),
                'pool_usdc': round(new_pool_usdc, 2),
                'total_value': round(new_portfolio_value, 2),
                'lp_fees': round(lp_fees, 2),
                'protocol_fees': round(protocol_fee, 2),
                'cumulative_lp_fees': round(self.cumulative_lp_fees, 2),
                'cumulative_protocol_fees': round(self.cumulative_protocol_fees, 2),
                'gas_fees_step': round(gas_fees_step, 2),
                'total_gas_fees': round(self.position["total_gas_fees"], 2),
                'action': "; ".join(actions_done) if actions_done else "no action",
                'reward': reward,
                'out_of_range_steps': self.position["out_of_range_steps"],
            })

        # Advance simulation
        done = self.current_step >= self.max_steps - 1
        self.current_step += 1

        if done and self.enable_logging:
            final_pool0, final_pool1 = amounts_in_position(
                self.position["liquidity"],
                self.prices[-1],
                tick_to_price(self.position["lower_tick"]),
                tick_to_price(self.position["upper_tick"]),
            )
            final_eth = self.eth_balance + final_pool0
            final_usdc = self.usdc_balance + final_pool1
            final_value = final_eth * self.prices[-1] + final_usdc
            
            # final valuations 
            gross_final_val = final_value
            price_only_val = gross_final_val - self.cumulative_lp_fees

            hodl_eth = (self.initial_investment / 2) / self.prices[0]
            hodl_val = hodl_eth * self.prices[-1] + (self.initial_investment / 2)

            classic_il_usd = price_only_val - hodl_val
            classic_il = (classic_il_usd / hodl_val) * 100
            net_profit = gross_final_val - self.position["total_gas_fees"] - self.initial_investment
            apr = (net_profit / self.initial_investment) * 100

            print("\nTraining Env => Done.")
            print(f"  Holding Value:         {round(hodl_val,2)} USDC")
            print(f"  Impermanent Loss:      {round(classic_il,2)} USDC")
            print(f"  LP Fees:               {round(self.cumulative_lp_fees,2)} USDC")
            print(f"  Gas Fees:              {round(self.position['total_gas_fees'],2)} USDC")
            print(f"  Net Profit:            {round(net_profit,2)} USDC")
            print(f"  APR:                   {round(apr,2)}%")

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        """
        Generate the current observation vector.

        Observation vector includes:
            - Current ETH/USDC price
            - Number of steps the position has been out of range
            - Liquidity in current position
            - Volume on current day
            - Cumulative LP fees so far

        Returns:
            np.ndarray: 5-dimensional observation.
        """
        safe_step = min(self.current_step, self.max_steps - 1)
        return np.array(
            [
                float(self.prices[safe_step]),
                float(self.position["out_of_range_steps"]),
                float(self.position["liquidity"]),
                float(self.volumes[safe_step]),
                float(self.cumulative_lp_fees),
            ],
            dtype=np.float32,
        )

    def close(self):
        """
        Close the CSV log file (if logging was enabled).

        This is called at the end of a training run to ensure file handles
        are properly closed.
        """
        if self.csv_file and not self.csv_file.closed:
            self.csv_file.close()
            print("Train CSV file closed.")
        super().close()