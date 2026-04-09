from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship

from app.database import Base


class RoutingDecision(Base):
    __tablename__ = "routing_decisions"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(Integer, ForeignKey("requests.id"), unique=True, nullable=False)
    suggested_team = Column(String(100), nullable=False)
    priority_score = Column(Float, nullable=False)
    routing_explanation = Column(Text, nullable=False)
    category_match = Column(String(100), nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    request = relationship("Request", back_populates="routing_decision")
