from pydantic import BaseModel, Field
from typing import Optional


class EvaluationOutput(BaseModel):
    """Scores from the evaluator agent assessing coaching quality."""

    actionability: int = Field(
        ..., ge=1, le=10, description="How actionable is the coaching advice (1-10)"
    )
    empathy: int = Field(
        ..., ge=1, le=10, description="How empathetic is the tone (1-10)"
    )
    specificity: int = Field(
        ..., ge=1, le=10, description="How specific and personalized is the response (1-10)"
    )
    safety: int = Field(
        ..., ge=1, le=10, description="How safe and responsible is the advice (1-10)"
    )
    overall_notes: str = Field(
        ..., description="Brief evaluator notes on the coaching quality"
    )
