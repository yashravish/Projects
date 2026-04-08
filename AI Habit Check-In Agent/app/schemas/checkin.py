from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from app.schemas.evaluation import EvaluationOutput


class CheckInRequest(BaseModel):
    """Incoming check-in submission from the user."""

    health_goal: str = Field(..., min_length=1, description="The user's health goal")
    todays_actions: str = Field(..., min_length=1, description="What the user did today")
    current_mood: str = Field(..., min_length=1, description="How the user feels right now")


class CoachOutput(BaseModel):
    """Structured coaching response from the coach agent."""

    summary: str = Field(..., description="Short personalized coaching response")
    habit_risk: str = Field(..., description="One habit risk or pattern identified")
    next_action: str = Field(..., description="One actionable next step suggestion")
    motivational_message: str = Field(..., description="Supportive motivational message")


class CheckInResponse(BaseModel):
    """Full response returned to the client after processing a check-in."""

    id: int
    health_goal: str
    todays_actions: str
    current_mood: str
    coach_output: CoachOutput
    evaluation: EvaluationOutput
    created_at: datetime


class CheckInListItem(BaseModel):
    """Abbreviated check-in for list endpoints."""

    id: int
    health_goal: str
    current_mood: str
    created_at: datetime
