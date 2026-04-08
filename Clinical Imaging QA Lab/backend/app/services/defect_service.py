import logging
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models import Defect
from app.schemas import DefectCreate

logger = logging.getLogger(__name__)


class DefectService:
    """Business logic for defect/bug tracking operations."""

    @staticmethod
    def create_defect(db: Session, data: DefectCreate) -> Defect:
        """Create a new defect report."""
        defect = Defect(
            title=data.title,
            severity=data.severity,
            priority=data.priority,
            environment=data.environment,
            steps_to_reproduce=data.steps_to_reproduce,
            expected_result=data.expected_result,
            actual_result=data.actual_result,
            status="open",
        )
        db.add(defect)
        db.commit()
        db.refresh(defect)
        logger.info("Defect %d created: %s", defect.id, defect.title)
        return defect

    @staticmethod
    def list_defects(db: Session, limit: int = 100, offset: int = 0) -> list[Defect]:
        """Retrieve defect records ordered by most recent."""
        return (
            db.query(Defect)
            .order_by(Defect.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

    @staticmethod
    def get_defect(db: Session, defect_id: int) -> Defect | None:
        """Retrieve a single defect by ID."""
        return db.query(Defect).filter(Defect.id == defect_id).first()

    @staticmethod
    def get_summary(db: Session) -> dict:
        """Return aggregate defect counts."""
        total = db.query(func.count(Defect.id)).scalar() or 0
        open_count = (
            db.query(func.count(Defect.id))
            .filter(Defect.status == "open")
            .scalar() or 0
        )
        return {"total_defects": total, "open_defects": open_count}
