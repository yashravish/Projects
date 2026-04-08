from dataclasses import dataclass
from datetime import datetime


@dataclass
class CheckInRecord:
    """Represents a row in the checkins table."""

    id: int
    health_goal: str
    todays_actions: str
    current_mood: str
    summary: str
    habit_risk: str
    next_action: str
    motivational_message: str
    actionability_score: int
    empathy_score: int
    specificity_score: int
    safety_score: int
    evaluation_notes: str
    created_at: str
