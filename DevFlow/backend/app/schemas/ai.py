import datetime as dt
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class AIAnalyzeRequest(BaseModel):
    logs: str = Field(min_length=1)
    project_id: Optional[int] = None
    create_defect: bool = False
    link_kb_article_ids: List[int] = []


class AIAnalysisRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    log_hash: str
    root_cause_summary: str
    likely_file_or_component: str
    suggested_fix: str
    severity: str
    confidence_score: float
    project_id: int | None
    created_at: dt.datetime
    created_defect_id: int | None = None
    linked_kb_article_ids: List[int] = []


class AIAnalysisReadPublic(BaseModel):
    root_cause_summary: str
    likely_file_or_component: str
    suggested_fix: str
    severity: str
    confidence_score: float
