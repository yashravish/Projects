from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.ai_summary import AISummaryOut
from app.schemas.routing import RoutingDecisionOut


class RequestCreate(BaseModel):
    title: str
    description: str
    category: str
    urgency: int = Field(ge=1, le=5)
    business_impact: int = Field(ge=1, le=5)
    desired_completion_date: date | None = None


class RequestUpdateIn(BaseModel):
    status: str | None = None
    assigned_owner: str | None = None
    note: str | None = None
    resolution_summary: str | None = None


class RequestOut(BaseModel):
    id: int
    requester_id: int
    requester_name: str
    title: str
    description: str
    category: str
    urgency: int
    business_impact: int
    desired_completion_date: date | None
    status: str
    priority_score: float | None
    assigned_team: str | None
    assigned_owner: str | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class RequestUpdateOut(BaseModel):
    id: int
    author_name: str
    status_change: str | None
    note: str | None
    created_at: datetime


class RequestDetailOut(BaseModel):
    id: int
    requester_id: int
    requester_name: str
    title: str
    description: str
    category: str
    urgency: int
    business_impact: int
    desired_completion_date: date | None
    status: str
    priority_score: float | None
    assigned_team: str | None
    assigned_owner: str | None
    created_at: datetime
    updated_at: datetime
    routing_decision: RoutingDecisionOut | None = None
    ai_summary: AISummaryOut | None = None
    updates: list[RequestUpdateOut] = []

    model_config = ConfigDict(from_attributes=True)
