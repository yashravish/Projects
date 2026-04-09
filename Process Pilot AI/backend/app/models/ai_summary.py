from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship

from app.database import Base


class AISummary(Base):
    __tablename__ = "ai_summaries"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(Integer, ForeignKey("requests.id"), unique=True, nullable=False)
    summary = Column(Text, nullable=False)
    business_impact_explanation = Column(Text, nullable=False)
    recommended_action = Column(Text, nullable=False)
    leadership_summary = Column(Text, nullable=False)
    implementation_notes = Column(Text, nullable=True)
    provider_used = Column(String(50), nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    request = relationship("Request", back_populates="ai_summary")
