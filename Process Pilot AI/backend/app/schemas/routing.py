from datetime import datetime

from pydantic import BaseModel, ConfigDict


class RoutingDecisionOut(BaseModel):
    id: int
    suggested_team: str
    priority_score: float
    routing_explanation: str
    category_match: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
