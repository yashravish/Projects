from sqlalchemy import Column, Integer, String, Text, Float, Date, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship

from app.database import Base


class Request(Base):
    __tablename__ = "requests"

    id = Column(Integer, primary_key=True, index=True)
    requester_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    category = Column(String(50), nullable=False)
    urgency = Column(Integer, nullable=False)
    business_impact = Column(Integer, nullable=False)
    desired_completion_date = Column(Date, nullable=True)
    status = Column(String(30), nullable=False, default="submitted")
    priority_score = Column(Float, nullable=True)
    assigned_team = Column(String(100), nullable=True)
    assigned_owner = Column(String(255), nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    requester = relationship("User", back_populates="requests")
    routing_decision = relationship(
        "RoutingDecision", uselist=False, back_populates="request"
    )
    ai_summary = relationship("AISummary", uselist=False, back_populates="request")
    updates = relationship(
        "RequestUpdate",
        back_populates="request",
        order_by="desc(RequestUpdate.created_at)",
    )
