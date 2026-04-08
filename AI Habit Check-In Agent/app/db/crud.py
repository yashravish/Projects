from datetime import datetime, timezone
from typing import Optional
import aiosqlite
from app.db.database import get_db
from app.db.models import CheckInRecord
from app.schemas.checkin import CoachOutput, CheckInRequest
from app.schemas.evaluation import EvaluationOutput
from app.utils.logging import logger


async def create_checkin(
    request: CheckInRequest,
    coach: CoachOutput,
    evaluation: EvaluationOutput,
) -> CheckInRecord:
    """Insert a new check-in record and return the created record."""
    created_at = datetime.now(timezone.utc).isoformat()

    async with get_db() as db:
        cursor = await db.execute(
            """
            INSERT INTO checkins (
                health_goal, todays_actions, current_mood,
                summary, habit_risk, next_action, motivational_message,
                actionability_score, empathy_score, specificity_score, safety_score,
                evaluation_notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request.health_goal,
                request.todays_actions,
                request.current_mood,
                coach.summary,
                coach.habit_risk,
                coach.next_action,
                coach.motivational_message,
                evaluation.actionability,
                evaluation.empathy,
                evaluation.specificity,
                evaluation.safety,
                evaluation.overall_notes,
                created_at,
            ),
        )
        await db.commit()
        record_id = cursor.lastrowid

    logger.info(f"Created check-in record id={record_id}")
    return CheckInRecord(
        id=record_id,
        health_goal=request.health_goal,
        todays_actions=request.todays_actions,
        current_mood=request.current_mood,
        summary=coach.summary,
        habit_risk=coach.habit_risk,
        next_action=coach.next_action,
        motivational_message=coach.motivational_message,
        actionability_score=evaluation.actionability,
        empathy_score=evaluation.empathy,
        specificity_score=evaluation.specificity,
        safety_score=evaluation.safety,
        evaluation_notes=evaluation.overall_notes,
        created_at=created_at,
    )


async def get_checkin_by_id(checkin_id: int) -> Optional[CheckInRecord]:
    """Retrieve a single check-in by ID."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM checkins WHERE id = ?", (checkin_id,)
        )
        row = await cursor.fetchone()

    if row is None:
        return None

    return _row_to_record(row)


async def get_all_checkins() -> list[CheckInRecord]:
    """Retrieve all check-ins, ordered by newest first."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM checkins ORDER BY id DESC"
        )
        rows = await cursor.fetchall()

    return [_row_to_record(row) for row in rows]


def _row_to_record(row: aiosqlite.Row) -> CheckInRecord:
    """Convert a database row to a CheckInRecord dataclass."""
    return CheckInRecord(
        id=row["id"],
        health_goal=row["health_goal"],
        todays_actions=row["todays_actions"],
        current_mood=row["current_mood"],
        summary=row["summary"],
        habit_risk=row["habit_risk"],
        next_action=row["next_action"],
        motivational_message=row["motivational_message"],
        actionability_score=row["actionability_score"],
        empathy_score=row["empathy_score"],
        specificity_score=row["specificity_score"],
        safety_score=row["safety_score"],
        evaluation_notes=row["evaluation_notes"],
        created_at=row["created_at"],
    )
