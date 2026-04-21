"""SQLAlchemy ORM models for execsim.

All models use SQLAlchemy 2.x Mapped[] style. Each model has:
  - Database-level CHECK constraints for data integrity.
  - Python-level @validates for early error detection on attribute set.
  - Relationships with cascade deletes where appropriate.
"""

import enum
import uuid
from datetime import datetime
from typing import Optional

import sqlalchemy as sa
from sqlalchemy import CheckConstraint, ForeignKey, Index, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    validates,
)


class Base(DeclarativeBase):
    """Base class for all ORM models."""


# ---------------------------------------------------------------------------
# Enums (native PostgreSQL enum types)
# ---------------------------------------------------------------------------

class RunStatus(enum.Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class OpportunityType(enum.Enum):
    cross_venue_arb = "cross_venue_arb"
    stale_quote = "stale_quote"
    liquidation = "liquidation"


class Side(enum.Enum):
    buy = "buy"
    sell = "sell"


class Venue(enum.Enum):
    venue_a = "venue_a"
    venue_b = "venue_b"
    amm = "amm"


class CheckType(enum.Enum):
    schema_check = "schema"
    temporal = "temporal"
    state = "state"
    calibration = "calibration"


class Severity(enum.Enum):
    info = "info"
    warning = "warning"
    error = "error"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class SimulationRun(Base):
    __tablename__ = "simulation_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    seed: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    config: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    started_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False,
    )
    finished_at: Mapped[Optional[datetime]] = mapped_column(
        sa.DateTime(timezone=True), nullable=True,
    )
    status: Mapped[RunStatus] = mapped_column(
        sa.Enum(RunStatus, name="runstatus", create_constraint=False),
        nullable=False,
        server_default="pending",
    )
    num_steps: Mapped[int] = mapped_column(sa.Integer, nullable=False)

    snapshots: Mapped[list["MarketSnapshot"]] = relationship(
        back_populates="run", cascade="all, delete-orphan",
    )
    opportunities: Mapped[list["Opportunity"]] = relationship(
        back_populates="run", cascade="all, delete-orphan",
    )
    auctions: Mapped[list["Auction"]] = relationship(
        back_populates="run", cascade="all, delete-orphan",
    )
    alerts: Mapped[list["ValidationAlert"]] = relationship(
        back_populates="run", cascade="all, delete-orphan",
    )

    __table_args__ = (
        CheckConstraint("seed >= 0", name="ck_run_seed_nonneg"),
        CheckConstraint("num_steps >= 1", name="ck_run_num_steps_pos"),
    )

    @validates("seed")
    def validate_seed(self, _key: str, value: int) -> int:
        if value < 0:
            raise ValueError("seed must be >= 0")
        return value

    @validates("num_steps")
    def validate_num_steps(self, _key: str, value: int) -> int:
        if value < 1:
            raise ValueError("num_steps must be >= 1")
        return value


class MarketSnapshot(Base):
    __tablename__ = "market_snapshots"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("simulation_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    step: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    venue_a_bid: Mapped[float] = mapped_column(sa.Float, nullable=False)
    venue_a_ask: Mapped[float] = mapped_column(sa.Float, nullable=False)
    venue_b_bid: Mapped[float] = mapped_column(sa.Float, nullable=False)
    venue_b_ask: Mapped[float] = mapped_column(sa.Float, nullable=False)
    amm_reserve_x: Mapped[float] = mapped_column(sa.Float, nullable=False)
    amm_reserve_y: Mapped[float] = mapped_column(sa.Float, nullable=False)
    amm_price: Mapped[float] = mapped_column(sa.Float, nullable=False)
    true_mid: Mapped[float] = mapped_column(sa.Float, nullable=False)
    ts: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False,
    )
    has_liquidation: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, server_default="false",
    )

    run: Mapped["SimulationRun"] = relationship(back_populates="snapshots")

    __table_args__ = (
        UniqueConstraint("run_id", "step", name="uq_snapshot_run_step"),
        Index("ix_snapshot_run_step", "run_id", "step"),
        CheckConstraint("step >= 0", name="ck_snapshot_step_nonneg"),
        CheckConstraint("venue_a_bid > 0", name="ck_snapshot_va_bid_pos"),
        CheckConstraint("venue_a_ask > 0", name="ck_snapshot_va_ask_pos"),
        CheckConstraint("venue_b_bid > 0", name="ck_snapshot_vb_bid_pos"),
        CheckConstraint("venue_b_ask > 0", name="ck_snapshot_vb_ask_pos"),
        CheckConstraint("amm_reserve_x > 0", name="ck_snapshot_amm_x_pos"),
        CheckConstraint("amm_reserve_y > 0", name="ck_snapshot_amm_y_pos"),
        CheckConstraint("amm_price > 0", name="ck_snapshot_amm_price_pos"),
        CheckConstraint("true_mid > 0", name="ck_snapshot_mid_pos"),
    )

    @validates(
        "venue_a_bid", "venue_a_ask", "venue_b_bid", "venue_b_ask",
        "amm_reserve_x", "amm_reserve_y", "amm_price", "true_mid",
    )
    def validate_positive(self, key: str, value: float) -> float:
        if value <= 0:
            raise ValueError(f"{key} must be > 0")
        return value

    @validates("step")
    def validate_step(self, _key: str, value: int) -> int:
        if value < 0:
            raise ValueError("step must be >= 0")
        return value


class Opportunity(Base):
    __tablename__ = "opportunities"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("simulation_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    step: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    type: Mapped[OpportunityType] = mapped_column(
        sa.Enum(OpportunityType, name="opportunitytype", create_constraint=False),
        nullable=False,
    )
    side: Mapped[Side] = mapped_column(
        sa.Enum(Side, name="side", create_constraint=False),
        nullable=False,
    )
    estimated_value_bps: Mapped[float] = mapped_column(sa.Float, nullable=False)
    arrival_mid: Mapped[float] = mapped_column(sa.Float, nullable=False)
    edge_bps: Mapped[float] = mapped_column(sa.Float, nullable=False)
    detail: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    detected_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False,
    )

    run: Mapped["SimulationRun"] = relationship(back_populates="opportunities")
    fill: Mapped[Optional["Fill"]] = relationship(
        back_populates="opportunity", cascade="all, delete-orphan", uselist=False,
    )
    auction_entries: Mapped[list["AuctionEntry"]] = relationship(
        back_populates="opportunity", cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_opportunity_run_id", "run_id"),
        Index("ix_opportunity_type", "type"),
        CheckConstraint("step >= 0", name="ck_opp_step_nonneg"),
        CheckConstraint("arrival_mid > 0", name="ck_opp_arrival_mid_pos"),
    )

    @validates("arrival_mid")
    def validate_arrival_mid(self, _key: str, value: float) -> float:
        if value <= 0:
            raise ValueError("arrival_mid must be > 0")
        return value

    @validates("step")
    def validate_step(self, _key: str, value: int) -> int:
        if value < 0:
            raise ValueError("step must be >= 0")
        return value


class Fill(Base):
    __tablename__ = "fills"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    opportunity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("opportunities.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    venue: Mapped[Venue] = mapped_column(
        sa.Enum(Venue, name="venue", create_constraint=False),
        nullable=False,
    )
    requested_qty: Mapped[float] = mapped_column(sa.Float, nullable=False)
    filled_qty: Mapped[float] = mapped_column(sa.Float, nullable=False)
    exec_price: Mapped[float] = mapped_column(sa.Float, nullable=False)
    decision_price: Mapped[float] = mapped_column(sa.Float, nullable=False)
    arrival_mid: Mapped[float] = mapped_column(sa.Float, nullable=False)
    latency_steps: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    executed_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False,
    )

    opportunity: Mapped["Opportunity"] = relationship(back_populates="fill")
    metric: Mapped[Optional["ExecutionMetric"]] = relationship(
        back_populates="fill", cascade="all, delete-orphan", uselist=False,
    )

    __table_args__ = (
        CheckConstraint("requested_qty > 0", name="ck_fill_req_qty_pos"),
        CheckConstraint("filled_qty > 0", name="ck_fill_filled_qty_pos"),
        CheckConstraint("filled_qty <= requested_qty", name="ck_fill_qty_le_req"),
        CheckConstraint("exec_price > 0", name="ck_fill_exec_price_pos"),
        CheckConstraint("decision_price > 0", name="ck_fill_dec_price_pos"),
        CheckConstraint("arrival_mid > 0", name="ck_fill_arrival_mid_pos"),
        CheckConstraint("latency_steps >= 0", name="ck_fill_latency_nonneg"),
    )

    @validates("requested_qty", "filled_qty", "exec_price", "decision_price", "arrival_mid")
    def validate_positive(self, key: str, value: float) -> float:
        if value <= 0:
            raise ValueError(f"{key} must be > 0")
        return value

    @validates("latency_steps")
    def validate_latency(self, _key: str, value: int) -> int:
        if value < 0:
            raise ValueError("latency_steps must be >= 0")
        return value


class ExecutionMetric(Base):
    __tablename__ = "execution_metrics"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    fill_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("fills.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    impl_shortfall_bps: Mapped[float] = mapped_column(sa.Float, nullable=False)
    realized_slippage_bps: Mapped[float] = mapped_column(sa.Float, nullable=False)
    fill_quality: Mapped[float] = mapped_column(sa.Float, nullable=False)

    fill: Mapped["Fill"] = relationship(back_populates="metric")

    __table_args__ = (
        CheckConstraint(
            "fill_quality >= 0 AND fill_quality <= 1",
            name="ck_metric_fill_quality_range",
        ),
    )

    @validates("fill_quality")
    def validate_fill_quality(self, _key: str, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("fill_quality must be in [0, 1]")
        return value


class Auction(Base):
    __tablename__ = "auctions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("simulation_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    reserve_price_bps: Mapped[float] = mapped_column(sa.Float, nullable=False)
    num_opportunities: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    num_allocated: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False,
    )

    run: Mapped["SimulationRun"] = relationship(back_populates="auctions")
    entries: Mapped[list["AuctionEntry"]] = relationship(
        back_populates="auction", cascade="all, delete-orphan",
    )
    result: Mapped[Optional["AuctionResult"]] = relationship(
        back_populates="auction", cascade="all, delete-orphan", uselist=False,
    )

    __table_args__ = (
        CheckConstraint("reserve_price_bps >= 0", name="ck_auction_reserve_nonneg"),
        CheckConstraint("num_opportunities >= 0", name="ck_auction_num_opp_nonneg"),
        CheckConstraint("num_allocated >= 0", name="ck_auction_num_alloc_nonneg"),
        CheckConstraint(
            "num_allocated <= num_opportunities", name="ck_auction_alloc_le_opp",
        ),
    )

    @validates("reserve_price_bps")
    def validate_reserve(self, _key: str, value: float) -> float:
        if value < 0:
            raise ValueError("reserve_price_bps must be >= 0")
        return value


class AuctionEntry(Base):
    __tablename__ = "auction_entries"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    auction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("auctions.id", ondelete="CASCADE"),
        nullable=False,
    )
    opportunity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("opportunities.id", ondelete="CASCADE"),
        nullable=False,
    )
    bidder_index: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    bid_value_bps: Mapped[float] = mapped_column(sa.Float, nullable=False)
    won: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, server_default="false",
    )
    payment_bps: Mapped[float] = mapped_column(
        sa.Float, nullable=False, server_default="0",
    )

    auction: Mapped["Auction"] = relationship(back_populates="entries")
    opportunity: Mapped["Opportunity"] = relationship(back_populates="auction_entries")

    __table_args__ = (
        CheckConstraint("payment_bps >= 0", name="ck_entry_payment_nonneg"),
        CheckConstraint("bidder_index >= 0", name="ck_entry_bidder_idx_nonneg"),
        Index("ix_entry_auction_id", "auction_id"),
    )


class AuctionResult(Base):
    __tablename__ = "auction_results"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    auction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("auctions.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    total_revenue_bps: Mapped[float] = mapped_column(sa.Float, nullable=False)
    allocation_rate: Mapped[float] = mapped_column(sa.Float, nullable=False)
    mean_payment_bps: Mapped[float] = mapped_column(sa.Float, nullable=False)

    auction: Mapped["Auction"] = relationship(back_populates="result")

    __table_args__ = (
        CheckConstraint(
            "allocation_rate >= 0 AND allocation_rate <= 1",
            name="ck_result_alloc_rate_range",
        ),
    )

    @validates("allocation_rate")
    def validate_allocation_rate(self, _key: str, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("allocation_rate must be in [0, 1]")
        return value


class ValidationAlert(Base):
    __tablename__ = "validation_alerts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("simulation_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    check_type: Mapped[CheckType] = mapped_column(
        sa.Enum(CheckType, name="checktype", create_constraint=False),
        nullable=False,
    )
    severity: Mapped[Severity] = mapped_column(
        sa.Enum(Severity, name="severity", create_constraint=False),
        nullable=False,
    )
    message: Mapped[str] = mapped_column(sa.Text, nullable=False)
    detail: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False,
    )

    run: Mapped["SimulationRun"] = relationship(back_populates="alerts")

    __table_args__ = (
        Index("ix_alert_run_id", "run_id"),
        Index("ix_alert_check_type", "check_type"),
        Index("ix_alert_severity", "severity"),
    )
