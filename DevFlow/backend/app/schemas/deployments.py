import datetime as dt
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class DeploymentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    version: str
    status: str
    environment: str
    canary_percent: int
    error_rate: float
    rolled_back_from_id: int | None
    created_at: dt.datetime
    updated_at: dt.datetime


class DeploymentCreateBody(BaseModel):
    version: str
    environment: str = "production"
    canary: bool = False
    canary_start_percent: int = Field(10, ge=10, le=100)


class CanaryBody(BaseModel):
    target_max_percent: int = Field(100, description="10, 25, 50, or 100 recommended")


class RollbackBody(BaseModel):
    reason: str = "user_request"
