from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas import CaptureCreate, CaptureResponse
from app.services.capture_service import CaptureService

router = APIRouter(prefix="/api/captures", tags=["captures"])


@router.post("", response_model=CaptureResponse, status_code=201)
def create_capture(data: CaptureCreate, db: Session = Depends(get_db)):
    """Initiate a new imaging capture through the device simulator."""
    capture = CaptureService.create_capture(db, data)
    return capture


@router.get("", response_model=list[CaptureResponse])
def list_captures(
    limit: int = 100, offset: int = 0, db: Session = Depends(get_db)
):
    """Retrieve capture history records."""
    return CaptureService.list_captures(db, limit=limit, offset=offset)


@router.get("/{capture_id}", response_model=CaptureResponse)
def get_capture(capture_id: int, db: Session = Depends(get_db)):
    """Retrieve a single capture by ID."""
    capture = CaptureService.get_capture(db, capture_id)
    if not capture:
        raise HTTPException(status_code=404, detail="Capture not found")
    return capture


@router.post("/{capture_id}/retry", response_model=CaptureResponse)
def retry_capture(capture_id: int, db: Session = Depends(get_db)):
    """Retry a previously failed capture."""
    capture = CaptureService.retry_capture(db, capture_id)
    if not capture:
        raise HTTPException(status_code=404, detail="Capture not found")
    return capture
