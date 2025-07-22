"""
eval_env.py — Evaluation Environment for LP Strategies on Uniswap V3

This module defines the `UniswapV3EvalEnv` class, a Gym-compatible environment
used to evaluate liquidity provisioning (LP) strategies on Uniswap V3 using 
a fixed historical scenario. Unlike the training environment, this version
replays a fixed price/volume/shock sequence to ensure consistent policy evaluation.

It tracks portfolio value, LP and protocol fees, gas costs, and impermanent loss.
At the end of the run, it logs performance metrics including APR, LP fees, and
classic impermanent loss (relative to a buy-and-hold strategy) into a CSV summary.

Key features:
- Tick-based LP logic using Uniswap v3 formulas
- Repositioning logic with customizable gas cost range
- Fee computation with shock-dependent tiers (0.05%, 0.3%, 1%)
- Logging of daily state and summary outcomes
- Reward structure matching [Arcifa et al., IEEE DAPPS 2025]

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
from utils.logging_helpers import record_summary
from utils.uniswap_math import (
    nearest_tick,
    price_to_tick,
    tick_to_price,
    tick_to_sqrt_price,
)


class UniswapV3EvalEnv(Env):
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
        shocks_eval,
        prices_eval,
        sqrt_prices_eval,
        volumes_eval,
        pool_liquidity_eval,
        strategy_name,
        seed,
        steps=365,
        initial_investment=10000,
        reposition_gas_cost_range=(50, 150),
        protocol_fee_fraction=0.1,
        tick_spacing=10,
        enable_logging=False,
        csv_filename="ppo_eval_log.csv",
    ):
        """
        Initialize the evaluation environment with fixed scenario data.

        Args:
            shocks_eval (list[float]): List of daily price shocks (signed), 
                used to determine volatility tier and tick direction.
            prices_eval (list[float]): ETH/USDC prices for each step.
            sqrt_prices_eval (list[float]): Precomputed sqrt prices for Uniswap math (not used here).
            volumes_eval (list[float]): Daily trading volumes in USD.
            pool_liquidity_eval (list[float]): Total pool liquidity snapshots per day.
            strategy_name (str): Label of the strategy (used for logging and summary).
            seed (int): Evaluation seed (used for reproducibility in CSV output).
            initial_investment (float): Starting portfolio value (default = $10,000).
            reposition_gas_cost_range (tuple[float, float]): Range for random gas cost in USD for each reposition.
            protocol_fee_fraction (float): Portion of fees taken by Uniswap protocol (default = 10%).
            tick_spacing (int): Tick granularity for range positioning (default = 10).
            enable_logging (bool): If True, enables per-step CSV logging and final summary.
            csv_filename (str): Output filename for daily log rows.
        """
        super().__init__()
        self.shocks = shocks_eval
        self.prices = prices_eval
        self.sqrt_prices = sqrt_prices_eval
        self.volumes = volumes_eval
        self.pool_liquidity = pool_liquidity_eval
        self.strategy_name = strategy_name
        self.seed = seed
        self.enable_logging = enable_logging
        self.csv_filename = csv_filename
        self.initial_investment = initial_investment
        self.reposition_gas_cost_range = reposition_gas_cost_range
        self.protocol_fee_fraction = protocol_fee_fraction
        self.tick_spacing = tick_spacing
        self.max_steps = steps
        self.current_step = 0

        # Tick-based liquidity map and fee records (per tick)
        self.tick_liquidity = {}
        self.tick_fees = {}

        # Portfolio balances
        self.eth_balance = None
        self.usdc_balance = None
        self.position = None

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

    def reset(self):
        """
        Reset the environment to its initial portfolio state.

        Initializes the portfolio with a 50/50 split between ETH and USDC.
        A random initial price range is selected around the starting price,
        and liquidity is added accordingly. Liquidity ticks are updated to 
        reflect this new position.

        Returns:
            np.ndarray: Initial observation vector.
        """
        self.current_step = 0
        init_price = self.prices[0]

        # Convert initial capital: 50% into ETH, 50% into USDC
        self.eth_balance = (self.initial_investment / 2) / init_price
        self.usdc_balance = self.initial_investment / 2

        # Choose a price range for the initial LP position
        r1 = random.uniform(0.9, 0.95)
        r2 = random.uniform(1.05, 1.1)
        lower_tick = nearest_tick(init_price * r1, self.tick_spacing)
        upper_tick = nearest_tick(init_price * r2, self.tick_spacing)

        lower_sqrt = tick_to_sqrt_price(lower_tick)
        upper_sqrt = tick_to_sqrt_price(upper_tick)

        # Compute liquidity L based on price range and wallet balances
        L = compute_liquidity_L(self.eth_balance, self.usdc_balance, lower_sqrt, upper_sqrt)

        # Compute required token amounts to mint that liquidity
        needed_eth, needed_usdc = amounts_in_position(
            L, init_price, tick_to_price(lower_tick), tick_to_price(upper_tick)
        )

        # Deduct tokens from wallet
        self.eth_balance -= needed_eth
        self.usdc_balance -= needed_usdc

        if self.eth_balance < 0 or self.usdc_balance < 0:
            raise ValueError("Insufficient wallet funds for initial position")

        # Initialize tick liquidity map
        self.tick_liquidity = {}
        update_tick_liquidity(self.tick_liquidity, lower_tick, L)
        update_tick_liquidity(self.tick_liquidity, upper_tick, -L)

        # Save initial LP position state
        self.position = {
            "lower_tick": lower_tick,
            "upper_tick": upper_tick,
            "out_of_range_steps": 0,
            "total_repositions": 0,
            "total_gas_fees": 0,
            "active": True,
            "liquidity": L,
        }

        # Reset fee counters
        self.cumulative_lp_fees = 0.0
        self.cumulative_protocol_fees = 0.0

        return self._get_obs()

    def step(self, action):
        """
        Advance one time step.

        Executes the given action (0 = hold, >0 = reposition), calculates LP fees,
        gas costs, and reward, and updates portfolio state.

        Reward function follows Equation (9) from Arcifa et al. (IEEE DAPPS 2025):

            Rₜ = (Vₜ - Vₜ₋₁) - λ_gas × gas_cost + α × feeₜ - λ_oor × 𝟙{oor > 5} × Vₜ₋₁

        Where:
            - Vₜ = mark-to-market portfolio value at step t
            - gas_cost = gas spent if action was reposition
            - feeₜ = LP fees earned
            - 𝟙{oor > 5} = penalty if position is out of range for >5 steps
            - λ_gas, λ_oor, α are tunable coefficients
        """
        idx = self.current_step
        price = self.prices[idx]
        vol = self.volumes[idx]

        # Compute old portfolio value before action
        old_eth, old_usdc = amounts_in_position(
            self.position["liquidity"],
            price,
            tick_to_price(self.position["lower_tick"]),
            tick_to_price(self.position["upper_tick"]),
        )
        old_val = (
            self.eth_balance * price
            + self.usdc_balance
            + (old_eth * price + old_usdc)
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
            gas_fees_step = random.uniform(*self.reposition_gas_cost_range)
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
        now_active = (p_lower <= price <= p_upper)
        self.position["active"] = now_active

        if now_active:
            self.position["out_of_range_steps"] = 0
        else:
            self.position["out_of_range_steps"] += 1

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
        if now_active and float(self.pool_liquidity[idx]) > 0:
            liq_share = float(self.position["liquidity"]) / float(self.pool_liquidity[idx])
            lp_fees = (total_fee - protocol_fee) * liq_share
        else:
            lp_fees = 0.0

        # Accumulate LP and protocol fees
        self.usdc_balance += lp_fees
        self.cumulative_lp_fees += lp_fees
        self.cumulative_protocol_fees += protocol_fee

        # Compute new portfolio value (wallet + in-range position)
        new_eth, new_usdc = amounts_in_position(
            self.position["liquidity"],
            price,
            p_lower,
            p_upper,
        )
        new_val = (
            self.eth_balance * price
            + self.usdc_balance
            + (new_eth * price + new_usdc)
        )

        # Compute reward (based on Equation (9))
        reward = (new_val - old_val) - gas_fees_step  # ΔV - gas

        if self.position["out_of_range_steps"] > 5:
            reward -= old_val * 0.01  # penalty for prolonged OOR

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
                'pool_eth': round(new_eth, 4),
                'pool_usdc': round(new_usdc, 2),
                'total_value': round(new_val, 2),
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
                tick_to_price(self.position["upper_tick"])
            )
            final_eth  = self.eth_balance + final_pool0
            final_usdc = self.usdc_balance + final_pool1
            final_value= final_eth*self.prices[-1] + final_usdc

             # final valuations 
            gross_final_val = final_eth * self.prices[-1] + final_usdc     # includes LP fees
            price_only_val  = gross_final_val - self.cumulative_lp_fees    # strip fees → classic IL basis

            hodl_eth = (self.initial_investment/2)/self.prices[0]
            hodl_val = hodl_eth*self.prices[-1] + (self.initial_investment/2)
            
            classic_il_usd  = price_only_val - hodl_val   
            classic_il   = (classic_il_usd / hodl_val) * 100      # textbook IL  (usually ≤ 0)
            net_profit   = gross_final_val - self.position["total_gas_fees"] - self.initial_investment
            apr          = (net_profit / self.initial_investment)*100

            # build one-row dict
            summary = {
                "seed": self.seed,
                "strategy": self.strategy_name,           # or strategy_name
                "volatility_regime": "real",

                # performance
                "apr": round(apr, 4),
                "impermanent_loss": round(classic_il, 2),
                "lp_fees": round(self.cumulative_lp_fees, 2),
                "gas_fees": round(self.position["total_gas_fees"], 2),

                # final balances  ——> (needed by make_gas_sensitivity.py)
                "eth_balance":  round(final_eth, 6),      # wallet + pool, in ETH
                "usdc_balance": round(final_usdc, 2),     # wallet + pool, in USDC
                "pool_eth":     round(final_pool0, 6),
                "pool_usdc":    round(final_pool1, 2),
                "price":        round(self.prices[-1], 2) # last ETH price (USD)
            }

            record_summary("results/seed_runs.csv", summary)

            print("\nEval Env => Done.")
            print(f"  Strategy:               {self.strategy_name}")
            print(f"  Holding Value:          {round(hodl_val,2)} USDC")
            print(f"  Final Portfolio Value:  {round(final_value,2)} USDC")
            print(f"  Impermanent Loss:       {round(classic_il, 2)} USDC")
            print(f"  LP Fees:                {round(self.cumulative_lp_fees,2)} USDC")
            print(f"  Gas Fees:               {round(self.position['total_gas_fees'],2)} USDC")
            netp = (final_value 
                    - self.position["total_gas_fees"] 
                    - self.initial_investment)
            print(f"  Net Profit:             {round(netp,2)} USDC")
            print(f"  APR:                    {round(apr,2)}%")

        # Final summary is written in reset() if logging is enabled
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

        This is called at the end of an evaluation run to ensure file handles
        are properly closed.
        """
        if self.csv_file and not self.csv_file.closed:
            self.csv_file.close()
            print("Eval CSV file closed.")
        super().close()
