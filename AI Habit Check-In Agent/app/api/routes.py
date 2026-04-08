from fastapi import APIRouter, HTTPException, status
from app.schemas.checkin import CheckInRequest, CheckInResponse, CheckInListItem
from app.services.checkin_service import process_checkin, get_checkin, list_checkins
from app.utils.logging import logger

router = APIRouter()


@router.get("/health", tags=["system"])
async def health_check() -> dict:
    """Health check endpoint to verify the API is running."""
    return {"status": "healthy", "service": "ai-habit-checkin-agent"}


@router.post(
    "/checkins",
    response_model=CheckInResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["checkins"],
)
async def create_checkin(request: CheckInRequest) -> CheckInResponse:
    """Submit a health check-in and receive AI coaching feedback with evaluation."""
    try:
        result = await process_checkin(request)
        return result
    except Exception as e:
        logger.error(f"Error processing check-in: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your check-in. Please try again.",
        )


@router.get(
    "/checkins/{checkin_id}",
    response_model=CheckInResponse,
    tags=["checkins"],
)
async def read_checkin(checkin_id: int) -> CheckInResponse:
    """Retrieve a specific check-in by its ID."""
    result = await get_checkin(checkin_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Check-in with id {checkin_id} not found",
        )
    return result


@router.get(
    "/checkins",
    response_model=list[CheckInListItem],
    tags=["checkins"],
)
async def read_all_checkins() -> list[CheckInListItem]:
    """Retrieve all check-ins, ordered by newest first."""
    return await list_checkins()
