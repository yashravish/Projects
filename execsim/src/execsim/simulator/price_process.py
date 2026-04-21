"""GBM mid-price process for the simulation.

Generates a sequence of strictly positive prices using Geometric Brownian Motion.

Process:
    S(t+1) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z(t))

Where:
    S(t) : price at step t
    mu   : per-step drift (dimensionless)
    sigma: per-step volatility (dimensionless, not annualized)
    dt   : time step size in simulated seconds
    Z(t) : standard normal draw from seeded RNG

This is a simulated process, not calibrated to any real asset.
"""

import numpy as np


def generate_mid_prices(
    num_steps: int,
    initial_price: float,
    mu: float,
    sigma: float,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a GBM price path.

    Args:
        num_steps: Number of price steps to generate.
        initial_price: S(0), must be > 0. Units: quote currency per base unit.
        mu: Per-step drift. Dimensionless.
        sigma: Per-step volatility. Dimensionless (not annualized).
        dt: Time step in simulated seconds (used only to scale mu and sigma).
        rng: Seeded numpy random generator for reproducibility.

    Returns:
        1-D array of length num_steps with prices S(0), S(1), ..., S(num_steps-1).
        All values are strictly positive (guaranteed by GBM).

    Formula:
        S(t+1) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z(t))
        Reference: standard GBM discretization (Euler-Maruyama on log-price).
    """
    if initial_price <= 0:
        raise ValueError(f"initial_price must be > 0, got {initial_price}")
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")

    prices = np.empty(num_steps, dtype=np.float64)
    prices[0] = initial_price

    if num_steps == 1:
        return prices

    z = rng.standard_normal(num_steps - 1)
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    log_returns = drift + diffusion * z
    prices[1:] = initial_price * np.exp(np.cumsum(log_returns))

    return prices
