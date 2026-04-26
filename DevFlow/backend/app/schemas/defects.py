import datetime as dt
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class DefectRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    title: str
    description: str
    severity: str
    priority: str
    status: str
    owner: str | None
    root_cause: str | None
    suggested_fix: str | None
    linked_pipeline_run_id: int | None
    ai_report_id: int | None
    created_at: dt.datetime
    updated_at: dt.datetime
    resolved_at: dt.datetime | None
    linked_kb_article_ids: List[int] = []


class DefectCreate(BaseModel):
    project_id: int
    title: str
    description: str = ""
    severity: str = "medium"
    priority: str = "p2"
    status: str = "open"
    owner: str | None = None
    root_cause: str | None = None
    suggested_fix: str | None = None
    linked_pipeline_run_id: int | None = None
    ai_report_id: int | None = None
    linked_kb_article_ids: List[int] = []


class DefectUpdate(BaseModel):
    title: str | None = None
    description: str | None = None
    severity: str | None = None
    priority: str | None = None
    status: str | None = None
    owner: str | None = None
    root_cause: str | None = None
    suggested_fix: str | None = None
    linked_pipeline_run_id: int | None = None
    linked_kb_article_ids: List[int] | None = None


class DefectStatsRead(BaseModel):
    open: int
    resolved: int
    defect_rate: float
    by_severity: dict[str, int]
