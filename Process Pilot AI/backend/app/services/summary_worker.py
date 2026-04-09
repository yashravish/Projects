import logging

from sqlalchemy.orm import Session

from app.models.ai_summary import AISummary
from app.models.request import Request
from app.services.ai_provider import get_ai_provider

logger = logging.getLogger(__name__)


def generate_request_summary(db: Session, request_id: int) -> AISummary:
    req = db.query(Request).filter(Request.id == request_id).first()
    if not req:
        raise ValueError(f"Request {request_id} not found")

    existing = db.query(AISummary).filter(AISummary.request_id == request_id).first()
    if existing:
        db.delete(existing)
        db.flush()

    provider = get_ai_provider()
    result = provider.generate_summary(
        title=req.title,
        description=req.description,
        category=req.category,
        urgency=req.urgency,
        business_impact=req.business_impact,
    )

    ai_summary = AISummary(
        request_id=request_id,
        summary=result["summary"],
        business_impact_explanation=result["business_impact_explanation"],
        recommended_action=result["recommended_action"],
        leadership_summary=result["leadership_summary"],
        implementation_notes=result.get("implementation_notes"),
        provider_used=provider.__class__.__name__,
    )
    db.add(ai_summary)
    db.commit()
    db.refresh(ai_summary)

    logger.info("Generated AI summary for request %d using %s", request_id, ai_summary.provider_used)
    return ai_summary
