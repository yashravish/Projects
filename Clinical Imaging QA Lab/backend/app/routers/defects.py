from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas import DefectCreate, DefectResponse
from app.services.defect_service import DefectService

router = APIRouter(prefix="/api/defects", tags=["defects"])


@router.post("", response_model=DefectResponse, status_code=201)
def create_defect(data: DefectCreate, db: Session = Depends(get_db)):
    """Log a new defect/bug report."""
    defect = DefectService.create_defect(db, data)
    return defect


@router.get("", response_model=list[DefectResponse])
def list_defects(
    limit: int = 100, offset: int = 0, db: Session = Depends(get_db)
):
    """Retrieve logged defects."""
    return DefectService.list_defects(db, limit=limit, offset=offset)


@router.get("/{defect_id}", response_model=DefectResponse)
def get_defect(defect_id: int, db: Session = Depends(get_db)):
    """Retrieve a single defect by ID."""
    defect = DefectService.get_defect(db, defect_id)
    if not defect:
        raise HTTPException(status_code=404, detail="Defect not found")
    return defect
