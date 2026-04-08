from app.schemas.checkin import CheckInRequest, CheckInResponse, CoachOutput, CheckInListItem
from app.schemas.evaluation import EvaluationOutput
from app.agents.workflow import run_checkin_workflow
from app.db.crud import create_checkin, get_checkin_by_id, get_all_checkins
from app.db.models import CheckInRecord
from app.utils.logging import logger
from datetime import datetime
from typing import Optional


def _record_to_response(record: CheckInRecord) -> CheckInResponse:
    """Convert a database record into the API response schema."""
    return CheckInResponse(
        id=record.id,
        health_goal=record.health_goal,
        todays_actions=record.todays_actions,
        current_mood=record.current_mood,
        coach_output=CoachOutput(
            summary=record.summary,
            habit_risk=record.habit_risk,
            next_action=record.next_action,
            motivational_message=record.motivational_message,
        ),
        evaluation=EvaluationOutput(
            actionability=record.actionability_score,
            empathy=record.empathy_score,
            specificity=record.specificity_score,
            safety=record.safety_score,
            overall_notes=record.evaluation_notes,
        ),
        created_at=datetime.fromisoformat(record.created_at),
    )


async def process_checkin(request: CheckInRequest) -> CheckInResponse:
    """Orchestrate the full check-in flow: LLM workflow -> persist -> respond."""
    logger.info(f"Processing check-in for goal: {request.health_goal[:50]}...")

    # Run the 2-node LangGraph workflow
    coach_output, evaluation = await run_checkin_workflow(request)

    # Persist to database
    record = await create_checkin(request, coach_output, evaluation)

    logger.info(f"Check-in processed and stored with id={record.id}")
    return _record_to_response(record)


async def get_checkin(checkin_id: int) -> Optional[CheckInResponse]:
    """Retrieve a single check-in by ID."""
    record = await get_checkin_by_id(checkin_id)
    if record is None:
        return None
    return _record_to_response(record)


async def list_checkins() -> list[CheckInListItem]:
    """Retrieve all check-ins as abbreviated list items."""
    records = await get_all_checkins()
    return [
        CheckInListItem(
            id=r.id,
            health_goal=r.health_goal,
            current_mood=r.current_mood,
            created_at=datetime.fromisoformat(r.created_at),
        )
        for r in records
    ]
