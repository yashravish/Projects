"""Shared schema types, enums, and base models."""

import enum

from pydantic import BaseModel


class RunStatusEnum(str, enum.Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class OpportunityTypeEnum(str, enum.Enum):
    cross_venue_arb = "cross_venue_arb"
    stale_quote = "stale_quote"
    liquidation = "liquidation"


class SideEnum(str, enum.Enum):
    buy = "buy"
    sell = "sell"


class VenueEnum(str, enum.Enum):
    venue_a = "venue_a"
    venue_b = "venue_b"
    amm = "amm"


class CheckTypeEnum(str, enum.Enum):
    schema_check = "schema"
    temporal = "temporal"
    state = "state"
    calibration = "calibration"


class SeverityEnum(str, enum.Enum):
    info = "info"
    warning = "warning"
    error = "error"


class ErrorResponse(BaseModel):
    """Standard error response body."""
    detail: str
