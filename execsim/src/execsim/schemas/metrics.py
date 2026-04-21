"""Schemas for execution metrics.

Metric definitions (used consistently in code and docs):
  - impl_shortfall_bps: (exec_price - arrival_mid) * signed_qty
      / (arrival_mid * |qty|) * 10000
  - realized_slippage_bps: (exec_price - decision_price)
      / decision_price * 10000
  - fill_quality: filled_qty / requested_qty, in [0, 1]
"""

from uuid import UUID

from pydantic import BaseModel, Field


class ExecutionMetricSchema(BaseModel):
    """Per-fill execution metric."""
    id: UUID
    fill_id: UUID
    impl_shortfall_bps: float
    realized_slippage_bps: float
    fill_quality: float = Field(ge=0, le=1)

    model_config = {"from_attributes": True}


class RunMetrics(BaseModel):
    """Aggregated metrics for a simulation run."""
    run_id: UUID
    num_opportunities: int = Field(ge=0)
    num_fills: int = Field(ge=0)
    mean_impl_shortfall_bps: float | None = None
    mean_realized_slippage_bps: float | None = None
    mean_fill_quality: float | None = Field(default=None, ge=0, le=1)
    total_edge_bps: float | None = None
