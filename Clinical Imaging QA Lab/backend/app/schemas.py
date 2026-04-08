from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class CaptureCreate(BaseModel):
    """Schema for creating a new capture request."""
    patient_id: str = Field(..., min_length=1, max_length=50, description="Patient identifier")
    session_id: str = Field(..., min_length=1, max_length=50, description="Session identifier")
    image_type: str = Field(..., min_length=1, max_length=50, description="Type of image")

    @field_validator("image_type")
    @classmethod
    def validate_image_type(cls, v: str) -> str:
        allowed = {"x-ray", "ct-scan", "mri", "ultrasound", "fluoroscopy"}
        if v.lower().strip() not in allowed:
            raise ValueError(f"image_type must be one of: {', '.join(sorted(allowed))}")
        return v.lower().strip()

    @field_validator("patient_id", "session_id")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class CaptureResponse(BaseModel):
    """Schema for capture response data."""
    id: int
    patient_id: str
    session_id: str
    image_type: str
    capture_status: str
    device_name: Optional[str] = None
    device_response_code: Optional[int] = None
    file_path: Optional[str] = None
    retry_count: int
    error_message: Optional[str] = None
    captured_at: Optional[datetime] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class DefectCreate(BaseModel):
    """Schema for creating a new defect report."""
    title: str = Field(..., min_length=1, max_length=200, description="Defect title")
    severity: str = Field(..., description="Severity level")
    priority: str = Field(..., description="Priority level")
    environment: Optional[str] = Field(None, max_length=100)
    steps_to_reproduce: Optional[str] = None
    expected_result: Optional[str] = None
    actual_result: Optional[str] = None

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        allowed = {"critical", "major", "minor", "trivial"}
        if v.lower().strip() not in allowed:
            raise ValueError(f"severity must be one of: {', '.join(sorted(allowed))}")
        return v.lower().strip()

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: str) -> str:
        allowed = {"high", "medium", "low"}
        if v.lower().strip() not in allowed:
            raise ValueError(f"priority must be one of: {', '.join(sorted(allowed))}")
        return v.lower().strip()


class DefectResponse(BaseModel):
    """Schema for defect response data."""
    id: int
    title: str
    severity: str
    priority: str
    environment: Optional[str] = None
    steps_to_reproduce: Optional[str] = None
    expected_result: Optional[str] = None
    actual_result: Optional[str] = None
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class DeviceStatusResponse(BaseModel):
    """Schema for device status from the simulator."""
    device_name: str
    status: str
    uptime_seconds: Optional[float] = None
    firmware_version: Optional[str] = None
    last_calibration: Optional[str] = None
    capture_count: int = 0
    failure_mode: Optional[str] = None


class DashboardSummary(BaseModel):
    """Schema for the dashboard summary stats."""
    total_captures: int
    successful_captures: int
    failed_captures: int
    pending_captures: int
    total_defects: int
    open_defects: int
    device_status: str
    recent_captures: list[CaptureResponse]
    recent_defects: list[DefectResponse]


class ErrorResponse(BaseModel):
    """Standard error response shape."""
    detail: str
