"""Constant-product AMM pool model.

Implements a simplified x*y=k automated market maker with configurable fee.
The pool is synthetically rebalanced each step to track the true mid-price
(this is a simulation simplification, not a realistic AMM dynamic).

Fee model: fee is charged on the input amount before the swap.
    effective_input = input_amount * (1 - fee_bps / 10000)

Pricing:
    Spot price (marginal, infinitesimal trade): P = reserve_y / reserve_x
    Execution price for buying dx of X:
        dy = reserve_y * dx / (reserve_x - dx)   (before fee adjustment on input)
    With fee on input:
        effective_dx = dx * (1 - fee_bps / 10000)
        dy = reserve_y * effective_dx / (reserve_x - effective_dx + effective_dx)

    Actually the standard constant-product swap with fee on input:
        For buying base (X) by paying quote (Y):
            dy_required = reserve_y * dx / (reserve_x - dx) with fee on dy input
        For selling base (X) to receive quote (Y):
            dy_out = reserve_y * effective_dx / (reserve_x + effective_dx)

This is a simulated pool. It is not connected to any blockchain.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class AMMPool:
    """Constant-product AMM pool state.

    Attributes:
        reserve_x: Reserve of base asset. Units: base units.
        reserve_y: Reserve of quote asset. Units: quote units.
        fee_bps: Swap fee in basis points, charged on input amount.
    """
    reserve_x: float
    reserve_y: float
    fee_bps: float

    def __post_init__(self):
        if self.reserve_x <= 0:
            raise ValueError(f"reserve_x must be > 0, got {self.reserve_x}")
        if self.reserve_y <= 0:
            raise ValueError(f"reserve_y must be > 0, got {self.reserve_y}")
        if self.fee_bps < 0:
            raise ValueError(f"fee_bps must be >= 0, got {self.fee_bps}")

    @property
    def spot_price(self) -> float:
        """Marginal price for infinitesimal trade.

        Units: quote currency per base unit.
        Formula: P = reserve_y / reserve_x
        """
        return self.reserve_y / self.reserve_x

    @property
    def k(self) -> float:
        """Invariant product k = reserve_x * reserve_y."""
        return self.reserve_x * self.reserve_y

    def quote_to_buy(self, base_amount: float) -> float:
        """Quote amount required to buy `base_amount` of base asset.

        This is a "buy base" swap: trader sends quote (Y), receives base (X).
        Fee is charged on the quote input.

        Args:
            base_amount: Amount of base to buy. Units: base units. Must be > 0
                and < reserve_x (cannot drain the pool).

        Returns:
            Amount of quote currency required (including fee). Units: quote units.

        Formula:
            Without fee: dy = reserve_y * dx / (reserve_x - dx)
            With fee on input: dy_with_fee = dy / (1 - fee_bps / 10000)
        """
        if base_amount <= 0:
            raise ValueError(f"base_amount must be > 0, got {base_amount}")
        if base_amount >= self.reserve_x:
            raise ValueError(
                f"base_amount ({base_amount}) must be < reserve_x ({self.reserve_x})"
            )

        dy_no_fee = self.reserve_y * base_amount / (self.reserve_x - base_amount)
        fee_multiplier = 1.0 - self.fee_bps / 10000.0
        if fee_multiplier <= 0:
            raise ValueError("fee_bps too high, effective multiplier <= 0")
        return dy_no_fee / fee_multiplier

    def exec_price_buy(self, base_amount: float) -> float:
        """Average execution price for buying base_amount of base.

        Args:
            base_amount: Amount of base to buy. Units: base units.

        Returns:
            Average price paid per base unit. Units: quote/base.

        Formula:
            exec_price = quote_to_buy(base_amount) / base_amount
        """
        return self.quote_to_buy(base_amount) / base_amount

    def quote_received_sell(self, base_amount: float) -> float:
        """Quote amount received for selling `base_amount` of base asset.

        This is a "sell base" swap: trader sends base (X), receives quote (Y).
        Fee is charged on the base input.

        Args:
            base_amount: Amount of base to sell. Units: base units. Must be > 0.

        Returns:
            Amount of quote currency received (after fee). Units: quote units.

        Formula:
            effective_dx = base_amount * (1 - fee_bps / 10000)
            dy = reserve_y * effective_dx / (reserve_x + effective_dx)
        """
        if base_amount <= 0:
            raise ValueError(f"base_amount must be > 0, got {base_amount}")

        effective_dx = base_amount * (1.0 - self.fee_bps / 10000.0)
        if effective_dx <= 0:
            raise ValueError("fee_bps too high, effective input <= 0")
        return self.reserve_y * effective_dx / (self.reserve_x + effective_dx)

    def exec_price_sell(self, base_amount: float) -> float:
        """Average execution price for selling base_amount of base.

        Args:
            base_amount: Amount of base to sell. Units: base units.

        Returns:
            Average price received per base unit. Units: quote/base.

        Formula:
            exec_price = quote_received_sell(base_amount) / base_amount
        """
        return self.quote_received_sell(base_amount) / base_amount


def create_pool(
    initial_price: float,
    reserve_x: float,
    fee_bps: float,
) -> AMMPool:
    """Create an AMM pool with reserves set to match an initial price.

    Args:
        initial_price: Target spot price P = reserve_y / reserve_x. Units: quote/base.
        reserve_x: Initial base reserve. Units: base units.
        fee_bps: Swap fee in basis points.

    Returns:
        AMMPool with reserve_y = initial_price * reserve_x.
    """
    if initial_price <= 0:
        raise ValueError(f"initial_price must be > 0, got {initial_price}")
    return AMMPool(
        reserve_x=reserve_x,
        reserve_y=initial_price * reserve_x,
        fee_bps=fee_bps,
    )


def rebalance_pool(
    pool: AMMPool,
    target_price: float,
    noise_std: float,
    rng: np.random.Generator,
) -> AMMPool:
    """Synthetically rebalance the pool to track a target price.

    This is a simulation simplification. In a real AMM, price changes come from
    arbitrageur trades. Here we directly adjust reserves to approximate the
    target price, with optional noise.

    Args:
        pool: Current pool state.
        target_price: Desired spot price. Units: quote/base.
        noise_std: Standard deviation of log-normal noise added to target price.
        rng: Seeded numpy random generator.

    Returns:
        New AMMPool with adjusted reserves. The product k is preserved.

    Method:
        noisy_price = target_price * exp(noise)
        new_reserve_x = sqrt(k / noisy_price)
        new_reserve_y = sqrt(k * noisy_price)
    """
    if target_price <= 0:
        raise ValueError(f"target_price must be > 0, got {target_price}")

    noise = rng.normal(0, noise_std) if noise_std > 0 else 0.0
    noisy_price = target_price * np.exp(noise)

    k = pool.k
    new_x = np.sqrt(k / noisy_price)
    new_y = np.sqrt(k * noisy_price)

    return AMMPool(reserve_x=new_x, reserve_y=new_y, fee_bps=pool.fee_bps)
