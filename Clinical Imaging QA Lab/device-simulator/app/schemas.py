from typing import Optional
from pydantic import BaseModel


class CaptureRequest(BaseModel):
    patient_id: str
    session_id: str
    image_type: str


class FailureModeRequest(BaseModel):
    mode: Optional[str] = None
