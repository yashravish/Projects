"""Initial schema: all tables and enum types.

Revision ID: 001_initial
Revises: None
Create Date: 2024-01-01
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- Enum types ---
    runstatus = sa.Enum(
        "pending", "running", "completed", "failed",
        name="runstatus",
    )
    runstatus.create(op.get_bind(), checkfirst=True)

    opportunitytype = sa.Enum(
        "cross_venue_arb", "stale_quote", "liquidation",
        name="opportunitytype",
    )
    opportunitytype.create(op.get_bind(), checkfirst=True)

    side = sa.Enum("buy", "sell", name="side")
    side.create(op.get_bind(), checkfirst=True)

    venue = sa.Enum("venue_a", "venue_b", "amm", name="venue")
    venue.create(op.get_bind(), checkfirst=True)

    checktype = sa.Enum(
        "schema", "temporal", "state", "calibration",
        name="checktype",
    )
    checktype.create(op.get_bind(), checkfirst=True)

    severity = sa.Enum("info", "warning", "error", name="severity")
    severity.create(op.get_bind(), checkfirst=True)

    # --- simulation_runs ---
    op.create_table(
        "simulation_runs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("seed", sa.Integer, nullable=False),
        sa.Column("config", JSONB, nullable=False, server_default="{}"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "status",
            sa.Enum(
                "pending", "running", "completed", "failed",
                name="runstatus", create_type=False,
            ),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("num_steps", sa.Integer, nullable=False),
        sa.CheckConstraint("seed >= 0", name="ck_run_seed_nonneg"),
        sa.CheckConstraint("num_steps >= 1", name="ck_run_num_steps_pos"),
    )

    # --- market_snapshots ---
    op.create_table(
        "market_snapshots",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "run_id", UUID(as_uuid=True),
            sa.ForeignKey("simulation_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("step", sa.Integer, nullable=False),
        sa.Column("venue_a_bid", sa.Float, nullable=False),
        sa.Column("venue_a_ask", sa.Float, nullable=False),
        sa.Column("venue_b_bid", sa.Float, nullable=False),
        sa.Column("venue_b_ask", sa.Float, nullable=False),
        sa.Column("amm_reserve_x", sa.Float, nullable=False),
        sa.Column("amm_reserve_y", sa.Float, nullable=False),
        sa.Column("amm_price", sa.Float, nullable=False),
        sa.Column("true_mid", sa.Float, nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("has_liquidation", sa.Boolean, nullable=False, server_default="false"),
        sa.UniqueConstraint("run_id", "step", name="uq_snapshot_run_step"),
        sa.CheckConstraint("step >= 0", name="ck_snapshot_step_nonneg"),
        sa.CheckConstraint("venue_a_bid > 0", name="ck_snapshot_va_bid_pos"),
        sa.CheckConstraint("venue_a_ask > 0", name="ck_snapshot_va_ask_pos"),
        sa.CheckConstraint("venue_b_bid > 0", name="ck_snapshot_vb_bid_pos"),
        sa.CheckConstraint("venue_b_ask > 0", name="ck_snapshot_vb_ask_pos"),
        sa.CheckConstraint("amm_reserve_x > 0", name="ck_snapshot_amm_x_pos"),
        sa.CheckConstraint("amm_reserve_y > 0", name="ck_snapshot_amm_y_pos"),
        sa.CheckConstraint("amm_price > 0", name="ck_snapshot_amm_price_pos"),
        sa.CheckConstraint("true_mid > 0", name="ck_snapshot_mid_pos"),
    )
    op.create_index("ix_snapshot_run_step", "market_snapshots", ["run_id", "step"])

    # --- opportunities ---
    op.create_table(
        "opportunities",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "run_id", UUID(as_uuid=True),
            sa.ForeignKey("simulation_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("step", sa.Integer, nullable=False),
        sa.Column(
            "type",
            sa.Enum(
                "cross_venue_arb", "stale_quote", "liquidation",
                name="opportunitytype", create_type=False,
            ),
            nullable=False,
        ),
        sa.Column(
            "side",
            sa.Enum("buy", "sell", name="side", create_type=False),
            nullable=False,
        ),
        sa.Column("estimated_value_bps", sa.Float, nullable=False),
        sa.Column("arrival_mid", sa.Float, nullable=False),
        sa.Column("edge_bps", sa.Float, nullable=False),
        sa.Column("detail", JSONB, nullable=False, server_default="{}"),
        sa.Column("detected_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("step >= 0", name="ck_opp_step_nonneg"),
        sa.CheckConstraint("arrival_mid > 0", name="ck_opp_arrival_mid_pos"),
    )
    op.create_index("ix_opportunity_run_id", "opportunities", ["run_id"])
    op.create_index("ix_opportunity_type", "opportunities", ["type"])

    # --- fills ---
    op.create_table(
        "fills",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "opportunity_id", UUID(as_uuid=True),
            sa.ForeignKey("opportunities.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column(
            "venue",
            sa.Enum("venue_a", "venue_b", "amm", name="venue", create_type=False),
            nullable=False,
        ),
        sa.Column("requested_qty", sa.Float, nullable=False),
        sa.Column("filled_qty", sa.Float, nullable=False),
        sa.Column("exec_price", sa.Float, nullable=False),
        sa.Column("decision_price", sa.Float, nullable=False),
        sa.Column("arrival_mid", sa.Float, nullable=False),
        sa.Column("latency_steps", sa.Integer, nullable=False),
        sa.Column("executed_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("requested_qty > 0", name="ck_fill_req_qty_pos"),
        sa.CheckConstraint("filled_qty > 0", name="ck_fill_filled_qty_pos"),
        sa.CheckConstraint("filled_qty <= requested_qty", name="ck_fill_qty_le_req"),
        sa.CheckConstraint("exec_price > 0", name="ck_fill_exec_price_pos"),
        sa.CheckConstraint("decision_price > 0", name="ck_fill_dec_price_pos"),
        sa.CheckConstraint("arrival_mid > 0", name="ck_fill_arrival_mid_pos"),
        sa.CheckConstraint("latency_steps >= 0", name="ck_fill_latency_nonneg"),
    )

    # --- execution_metrics ---
    op.create_table(
        "execution_metrics",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "fill_id", UUID(as_uuid=True),
            sa.ForeignKey("fills.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("impl_shortfall_bps", sa.Float, nullable=False),
        sa.Column("realized_slippage_bps", sa.Float, nullable=False),
        sa.Column("fill_quality", sa.Float, nullable=False),
        sa.CheckConstraint(
            "fill_quality >= 0 AND fill_quality <= 1",
            name="ck_metric_fill_quality_range",
        ),
    )

    # --- auctions ---
    op.create_table(
        "auctions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "run_id", UUID(as_uuid=True),
            sa.ForeignKey("simulation_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("reserve_price_bps", sa.Float, nullable=False),
        sa.Column("num_opportunities", sa.Integer, nullable=False),
        sa.Column("num_allocated", sa.Integer, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("reserve_price_bps >= 0", name="ck_auction_reserve_nonneg"),
        sa.CheckConstraint("num_opportunities >= 0", name="ck_auction_num_opp_nonneg"),
        sa.CheckConstraint("num_allocated >= 0", name="ck_auction_num_alloc_nonneg"),
        sa.CheckConstraint(
            "num_allocated <= num_opportunities", name="ck_auction_alloc_le_opp",
        ),
    )

    # --- auction_entries ---
    op.create_table(
        "auction_entries",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "auction_id", UUID(as_uuid=True),
            sa.ForeignKey("auctions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "opportunity_id", UUID(as_uuid=True),
            sa.ForeignKey("opportunities.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("bidder_index", sa.Integer, nullable=False),
        sa.Column("bid_value_bps", sa.Float, nullable=False),
        sa.Column("won", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("payment_bps", sa.Float, nullable=False, server_default="0"),
        sa.CheckConstraint("payment_bps >= 0", name="ck_entry_payment_nonneg"),
        sa.CheckConstraint("bidder_index >= 0", name="ck_entry_bidder_idx_nonneg"),
    )
    op.create_index("ix_entry_auction_id", "auction_entries", ["auction_id"])

    # --- auction_results ---
    op.create_table(
        "auction_results",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "auction_id", UUID(as_uuid=True),
            sa.ForeignKey("auctions.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("total_revenue_bps", sa.Float, nullable=False),
        sa.Column("allocation_rate", sa.Float, nullable=False),
        sa.Column("mean_payment_bps", sa.Float, nullable=False),
        sa.CheckConstraint(
            "allocation_rate >= 0 AND allocation_rate <= 1",
            name="ck_result_alloc_rate_range",
        ),
    )

    # --- validation_alerts ---
    op.create_table(
        "validation_alerts",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "run_id", UUID(as_uuid=True),
            sa.ForeignKey("simulation_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "check_type",
            sa.Enum(
                "schema", "temporal", "state", "calibration",
                name="checktype", create_type=False,
            ),
            nullable=False,
        ),
        sa.Column(
            "severity",
            sa.Enum("info", "warning", "error", name="severity", create_type=False),
            nullable=False,
        ),
        sa.Column("message", sa.Text, nullable=False),
        sa.Column("detail", JSONB, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_alert_run_id", "validation_alerts", ["run_id"])
    op.create_index("ix_alert_check_type", "validation_alerts", ["check_type"])
    op.create_index("ix_alert_severity", "validation_alerts", ["severity"])


def downgrade() -> None:
    op.drop_table("validation_alerts")
    op.drop_table("auction_results")
    op.drop_table("auction_entries")
    op.drop_table("auctions")
    op.drop_table("execution_metrics")
    op.drop_table("fills")
    op.drop_table("opportunities")
    op.drop_table("market_snapshots")
    op.drop_table("simulation_runs")

    for name in ("severity", "checktype", "venue", "side", "opportunitytype", "runstatus"):
        sa.Enum(name=name).drop(op.get_bind(), checkfirst=True)
