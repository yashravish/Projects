from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas import DashboardSummary, CaptureResponse, DefectResponse
from app.services.capture_service import CaptureService
from app.services.defect_service import DefectService
from app.services.device_service import DeviceService

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/summary", response_model=DashboardSummary)
def get_dashboard_summary(db: Session = Depends(get_db)):
    """Return aggregated dashboard data including counts and recent records."""
    capture_stats = CaptureService.get_summary(db)
    defect_stats = DefectService.get_summary(db)

    device_data = DeviceService.get_status()
    device_status = device_data.get("status", "unknown")

    recent_captures_raw = CaptureService.list_captures(db, limit=5, offset=0)
    recent_defects_raw = DefectService.list_defects(db, limit=5, offset=0)

    recent_captures = [CaptureResponse.model_validate(c) for c in recent_captures_raw]
    recent_defects = [DefectResponse.model_validate(d) for d in recent_defects_raw]

    return DashboardSummary(
        total_captures=capture_stats["total_captures"],
        successful_captures=capture_stats["successful_captures"],
        failed_captures=capture_stats["failed_captures"],
        pending_captures=capture_stats["pending_captures"],
        total_defects=defect_stats["total_defects"],
        open_defects=defect_stats["open_defects"],
        device_status=device_status,
        recent_captures=recent_captures,
        recent_defects=recent_defects,
    )
