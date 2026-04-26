import datetime as dt
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class StageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    sort_order: int
    status: str
    duration_ms: int
    logs: str
    passed: bool


class TestResultRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    suite: str
    status: str
    duration_ms: int
    message: str | None
    created_at: dt.datetime


class PipelineRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    status: str
    branch: str
    commit_sha: str
    started_at: Optional[dt.datetime]
    finished_at: Optional[dt.datetime]
    total_duration_ms: int
    external_ref: str | None
    stages: List[StageRead] = []
    test_results: List[TestResultRead] = []


class TriggerPipelineBody(BaseModel):
    branch: str = "main"
    commit_sha: str = Field(min_length=4, default="a1b2c3d")


class RecordTestResultBody(BaseModel):
    name: str
    suite: str = "default"
    status: str
    duration_ms: int = 0
    message: str | None = None
