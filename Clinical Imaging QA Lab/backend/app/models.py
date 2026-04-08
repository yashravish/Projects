from sqlalchemy import Column, Integer, String, Text, DateTime, func
from app.database import Base


class Capture(Base):
    """Represents a single imaging capture attempt."""
    __tablename__ = "captures"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(50), nullable=False, index=True)
    session_id = Column(String(50), nullable=False)
    image_type = Column(String(50), nullable=False)
    capture_status = Column(String(20), nullable=False, default="pending")
    device_name = Column(String(100), default="SIMULATED_SCANNER_01")
    device_response_code = Column(Integer, nullable=True)
    file_path = Column(String(500), nullable=True)
    retry_count = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    captured_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Defect(Base):
    """Represents a QA defect/bug report."""
    __tablename__ = "defects"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    severity = Column(String(20), nullable=False)
    priority = Column(String(20), nullable=False)
    environment = Column(String(100), nullable=True)
    steps_to_reproduce = Column(Text, nullable=True)
    expected_result = Column(Text, nullable=True)
    actual_result = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, default="open")
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class DeviceEvent(Base):
    """Audit log for device interactions."""
    __tablename__ = "device_events"

    id = Column(Integer, primary_key=True, index=True)
    device_name = Column(String(100), nullable=False)
    event_type = Column(String(50), nullable=False)
    details = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
