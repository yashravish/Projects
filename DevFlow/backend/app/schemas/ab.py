import datetime as dt
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ABMetricRollupRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    variant: str
    assignments: int
    conversion_count: int
    sum_latency_ms: float
    error_count: int
    updated_at: dt.datetime


class ABExperimentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    key: str
    name: str
    variant_a_name: str
    variant_b_name: str
    traffic_a_percent: int
    status: str
    key_metric: str
    notes: str | None
    created_at: dt.datetime
    updated_at: dt.datetime
    rollups: List[ABMetricRollupRead] = []


class ABExperimentCreate(BaseModel):
    project_id: int
    key: str = Field(min_length=1, max_length=120)
    name: str
    variant_a_name: str = "A"
    variant_b_name: str = "B"
    traffic_a_percent: int = Field(50, ge=0, le=100)
    key_metric: str = "conversion"
    notes: str | None = None


class ABAssignBody(BaseModel):
    experiment_id: int
    user_id: str


class ABMetricIngestBody(BaseModel):
    experiment_id: int
    variant: str
    user_id: str
    conversion: bool = False
    latency_ms: float = 0.0
    error: bool = False
