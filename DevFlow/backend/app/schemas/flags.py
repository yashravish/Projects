import datetime as dt

from pydantic import BaseModel, ConfigDict, Field


class FeatureFlagRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str | None
    enabled: bool
    rollout_percentage: int
    environment: str
    created_at: dt.datetime
    updated_at: dt.datetime


class FeatureFlagCreate(BaseModel):
    name: str
    description: str | None = None
    enabled: bool = False
    rollout_percentage: int = Field(ge=0, le=100, default=0)
    environment: str = "default"


class FeatureFlagUpdate(BaseModel):
    description: str | None = None
    enabled: bool | None = None
    rollout_percentage: int | None = Field(default=None, ge=0, le=100)
    environment: str | None = None


class FlagEvaluateBody(BaseModel):
    flag_id: int
    user_id: str
