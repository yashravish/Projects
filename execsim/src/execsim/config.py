"""Application configuration via environment variables."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration for execsim. Read from environment variables."""

    model_config = SettingsConfigDict(env_file=".env")

    database_url: str = Field(
        default="postgresql://execsim:execsim@localhost:5432/execsim",
        description="PostgreSQL connection string.",
    )

    # --- Simulator ---
    sim_num_steps: int = Field(default=1000, ge=1)
    sim_dt: float = Field(default=0.1, gt=0, description="Time step in simulated seconds.")
    sim_initial_price: float = Field(default=100.0, gt=0, description="Initial mid-price.")
    sim_sigma: float = Field(
        default=0.02, gt=0,
        description="Per-step volatility (not annualized). See docs/decisions.md.",
    )
    sim_mu: float = Field(default=0.0, description="Per-step drift.")

    # --- Venues ---
    venue_a_half_spread_bps: float = Field(default=5.0, gt=0)
    venue_b_half_spread_bps: float = Field(default=8.0, gt=0)
    venue_b_lag_steps: int = Field(default=2, ge=0)
    venue_levels: int = Field(default=5, ge=1)
    venue_tick_size: float = Field(default=0.01, gt=0)
    venue_min_level_qty: float = Field(default=1.0, gt=0)
    venue_max_level_qty: float = Field(default=10.0, gt=0)

    # --- AMM ---
    amm_initial_x: float = Field(default=10000.0, gt=0)
    amm_fee_bps: float = Field(default=30.0, ge=0, description="AMM fee in basis points.")
    amm_tracking_noise_std: float = Field(default=0.001, ge=0)

    # --- Liquidation ---
    liq_probability: float = Field(default=0.005, ge=0, le=1)
    liq_qty: float = Field(default=50.0, gt=0)
    liq_impact_bps: float = Field(default=30.0, gt=0)

    # --- Detectors ---
    arb_threshold_bps: float = Field(default=2.0, ge=0)
    stale_threshold_bps: float = Field(default=3.0, ge=0)
    liq_threshold_bps: float = Field(default=10.0, ge=0)
    estimated_cost_bps: float = Field(default=1.0, ge=0)

    # --- Auction ---
    auction_n_bidders: int = Field(default=3, ge=1)
    auction_reserve_bps: float = Field(default=0.0, ge=0)
    calibration_floor: float = Field(default=0.5, ge=0, le=1)
    calibration_grid_max_bps: int = Field(default=50, ge=0)
    calibration_grid_step_bps: int = Field(default=1, ge=1)
