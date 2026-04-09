from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session, joinedload

from app.auth.jwt_handler import get_current_user, require_manager
from app.database import get_db
from app.models.request import Request
from app.models.request_update import RequestUpdate
from app.models.routing_decision import RoutingDecision
from app.models.user import User
from app.schemas.ai_summary import AISummaryOut
from app.schemas.request import (
    RequestCreate,
    RequestDetailOut,
    RequestOut,
    RequestUpdateIn,
    RequestUpdateOut,
)
from app.schemas.routing import RoutingDecisionOut
from app.services.routing import route_request
from app.services.summary_worker import generate_request_summary

router = APIRouter(prefix="/api/requests", tags=["Requests"])


def _build_detail(req: Request) -> dict:
    """Build a RequestDetailOut-compatible dict from a Request ORM object."""
    data = {
        "id": req.id,
        "requester_id": req.requester_id,
        "requester_name": req.requester.full_name if req.requester else "Unknown",
        "title": req.title,
        "description": req.description,
        "category": req.category,
        "urgency": req.urgency,
        "business_impact": req.business_impact,
        "desired_completion_date": req.desired_completion_date,
        "status": req.status,
        "priority_score": req.priority_score,
        "assigned_team": req.assigned_team,
        "assigned_owner": req.assigned_owner,
        "created_at": req.created_at,
        "updated_at": req.updated_at,
        "routing_decision": None,
        "ai_summary": None,
        "updates": [],
    }
    if req.routing_decision:
        data["routing_decision"] = RoutingDecisionOut.model_validate(req.routing_decision)
    if req.ai_summary:
        data["ai_summary"] = AISummaryOut.model_validate(req.ai_summary)
    if req.updates:
        data["updates"] = [
            RequestUpdateOut(
                id=u.id,
                author_name=u.author.full_name if u.author else "Unknown",
                status_change=u.status_change,
                note=u.note,
                created_at=u.created_at,
            )
            for u in req.updates
        ]
    return data


@router.get("/", response_model=list[RequestOut])
def list_requests(
    department: str | None = None,
    category: str | None = None,
    status: str | None = None,
    priority_min: float | None = None,
    skip: int = 0,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(Request).join(User, Request.requester_id == User.id)

    if department:
        query = query.filter(User.department == department)
    if category:
        query = query.filter(Request.category == category)
    if status:
        query = query.filter(Request.status == status)
    if priority_min is not None:
        query = query.filter(Request.priority_score >= priority_min)

    requests = query.order_by(Request.created_at.desc()).offset(skip).limit(limit).all()

    return [
        RequestOut(
            id=r.id,
            requester_id=r.requester_id,
            requester_name=r.requester.full_name if r.requester else "Unknown",
            title=r.title,
            description=r.description,
            category=r.category,
            urgency=r.urgency,
            business_impact=r.business_impact,
            desired_completion_date=r.desired_completion_date,
            status=r.status,
            priority_score=r.priority_score,
            assigned_team=r.assigned_team,
            assigned_owner=r.assigned_owner,
            created_at=r.created_at,
            updated_at=r.updated_at,
        )
        for r in requests
    ]


@router.post("/", response_model=RequestDetailOut)
def create_request(
    body: RequestCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    new_request = Request(
        requester_id=current_user.id,
        title=body.title,
        description=body.description,
        category=body.category,
        urgency=body.urgency,
        business_impact=body.business_impact,
        desired_completion_date=body.desired_completion_date,
    )
    db.add(new_request)
    db.flush()

    routing_result = route_request(new_request)

    routing_decision = RoutingDecision(
        request_id=new_request.id,
        suggested_team=routing_result["suggested_team"],
        priority_score=routing_result["priority_score"],
        routing_explanation=routing_result["routing_explanation"],
        category_match=routing_result["category_match"],
    )
    db.add(routing_decision)

    new_request.priority_score = routing_result["priority_score"]
    new_request.assigned_team = routing_result["suggested_team"]

    db.commit()
    db.refresh(new_request)

    return RequestDetailOut.model_validate(_build_detail(new_request))


@router.get("/{request_id}", response_model=RequestDetailOut)
def get_request_detail(
    request_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    req = (
        db.query(Request)
        .options(
            joinedload(Request.requester),
            joinedload(Request.routing_decision),
            joinedload(Request.ai_summary),
            joinedload(Request.updates).joinedload(RequestUpdate.author),
        )
        .filter(Request.id == request_id)
        .first()
    )
    if not req:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Request not found")

    return RequestDetailOut.model_validate(_build_detail(req))


@router.patch("/{request_id}", response_model=RequestDetailOut)
def update_request(
    request_id: int,
    body: RequestUpdateIn,
    current_user: User = Depends(require_manager),
    db: Session = Depends(get_db),
):
    req = (
        db.query(Request)
        .options(
            joinedload(Request.requester),
            joinedload(Request.routing_decision),
            joinedload(Request.ai_summary),
            joinedload(Request.updates).joinedload(RequestUpdate.author),
        )
        .filter(Request.id == request_id)
        .first()
    )
    if not req:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Request not found")

    if body.status:
        req.status = body.status
    if body.assigned_owner:
        req.assigned_owner = body.assigned_owner

    update_record = RequestUpdate(
        request_id=request_id,
        author_id=current_user.id,
        status_change=body.status,
        note=body.note,
    )
    db.add(update_record)
    db.commit()
    db.refresh(req)

    req = (
        db.query(Request)
        .options(
            joinedload(Request.requester),
            joinedload(Request.routing_decision),
            joinedload(Request.ai_summary),
            joinedload(Request.updates).joinedload(RequestUpdate.author),
        )
        .filter(Request.id == request_id)
        .first()
    )

    return RequestDetailOut.model_validate(_build_detail(req))


@router.post("/{request_id}/summarize", response_model=AISummaryOut)
def summarize_request(
    request_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    req = db.query(Request).filter(Request.id == request_id).first()
    if not req:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Request not found")

    ai_summary = generate_request_summary(db, request_id)
    return AISummaryOut.model_validate(ai_summary)
