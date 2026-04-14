from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.auth import get_current_user
from backend.models import User, AssessmentTemplate, ControlDomain
from backend.schemas import ControlDomainOut

router = APIRouter(prefix="/api/templates", tags=["templates"])


@router.get("")
def list_templates(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    templates = db.query(AssessmentTemplate).filter(AssessmentTemplate.is_active == True).all()
    return [
        {
            "id": t.id,
            "name": t.name,
            "category": t.category,
            "description": t.description,
            "is_active": t.is_active,
            "created_at": t.created_at.isoformat() if t.created_at else None,
        }
        for t in templates
    ]


@router.get("/{template_id}")
def get_template(template_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    template = db.query(AssessmentTemplate).get(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    return {
        "id": template.id,
        "name": template.name,
        "category": template.category,
        "description": template.description,
        "questions": template.questions,
        "is_active": template.is_active,
    }


@router.get("/domains/list", response_model=list[ControlDomainOut])
def list_control_domains(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return db.query(ControlDomain).order_by(ControlDomain.code).all()
