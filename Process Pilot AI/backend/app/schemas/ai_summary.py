from datetime import datetime

from pydantic import BaseModel, ConfigDict


class AISummaryOut(BaseModel):
    id: int
    request_id: int
    summary: str
    business_impact_explanation: str
    recommended_action: str
    leadership_summary: str
    implementation_notes: str | None
    provider_used: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
