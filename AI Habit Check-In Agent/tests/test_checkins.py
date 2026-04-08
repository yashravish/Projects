"""Tests for check-in submission and retrieval endpoints."""

import pytest
from unittest.mock import AsyncMock, patch
from app.schemas.checkin import CoachOutput
from app.schemas.evaluation import EvaluationOutput

MOCK_COACH_OUTPUT = CoachOutput(
    summary="Great job walking 5000 steps! Let's build on that momentum.",
    habit_risk="Skipping breakfast may lead to overeating later in the day.",
    next_action="Try preparing a simple overnight oats recipe tonight for tomorrow's breakfast.",
    motivational_message="Every healthy choice counts — you're doing better than you think!",
)

MOCK_EVALUATION = EvaluationOutput(
    actionability=8,
    empathy=9,
    specificity=7,
    safety=10,
    overall_notes="Solid personalized advice with strong empathy. Could add more detail on the breakfast suggestion.",
)

SAMPLE_CHECKIN = {
    "health_goal": "eat better and lose weight",
    "todays_actions": "I skipped breakfast, had fast food for lunch, and walked 5000 steps",
    "current_mood": "stressed and tired",
}


@pytest.mark.asyncio
@patch("app.services.checkin_service.run_checkin_workflow")
async def test_create_checkin_returns_201(mock_workflow, client):
    """POST /checkins should return 201 with structured coaching response."""
    mock_workflow.return_value = (MOCK_COACH_OUTPUT, MOCK_EVALUATION)

    response = await client.post("/checkins", json=SAMPLE_CHECKIN)
    assert response.status_code == 201

    data = response.json()
    assert data["health_goal"] == SAMPLE_CHECKIN["health_goal"]
    assert data["todays_actions"] == SAMPLE_CHECKIN["todays_actions"]
    assert data["current_mood"] == SAMPLE_CHECKIN["current_mood"]
    assert "id" in data
    assert "created_at" in data


@pytest.mark.asyncio
@patch("app.services.checkin_service.run_checkin_workflow")
async def test_create_checkin_has_coach_output(mock_workflow, client):
    """POST /checkins response should include all coach_output fields."""
    mock_workflow.return_value = (MOCK_COACH_OUTPUT, MOCK_EVALUATION)

    response = await client.post("/checkins", json=SAMPLE_CHECKIN)
    data = response.json()

    coach = data["coach_output"]
    assert "summary" in coach
    assert "habit_risk" in coach
    assert "next_action" in coach
    assert "motivational_message" in coach


@pytest.mark.asyncio
@patch("app.services.checkin_service.run_checkin_workflow")
async def test_create_checkin_has_evaluation(mock_workflow, client):
    """POST /checkins response should include all evaluation fields."""
    mock_workflow.return_value = (MOCK_COACH_OUTPUT, MOCK_EVALUATION)

    response = await client.post("/checkins", json=SAMPLE_CHECKIN)
    data = response.json()

    evaluation = data["evaluation"]
    assert "actionability" in evaluation
    assert "empathy" in evaluation
    assert "specificity" in evaluation
    assert "safety" in evaluation
    assert "overall_notes" in evaluation


@pytest.mark.asyncio
@patch("app.services.checkin_service.run_checkin_workflow")
async def test_get_checkin_by_id(mock_workflow, client):
    """GET /checkins/{id} should return the correct check-in."""
    mock_workflow.return_value = (MOCK_COACH_OUTPUT, MOCK_EVALUATION)

    # Create a check-in first
    create_response = await client.post("/checkins", json=SAMPLE_CHECKIN)
    created_id = create_response.json()["id"]

    # Retrieve it
    response = await client.get(f"/checkins/{created_id}")
    assert response.status_code == 200
    assert response.json()["id"] == created_id


@pytest.mark.asyncio
async def test_get_checkin_not_found(client):
    """GET /checkins/{id} should return 404 for non-existent ID."""
    response = await client.get("/checkins/99999")
    assert response.status_code == 404


@pytest.mark.asyncio
@patch("app.services.checkin_service.run_checkin_workflow")
async def test_list_checkins(mock_workflow, client):
    """GET /checkins should return a list of all check-ins."""
    mock_workflow.return_value = (MOCK_COACH_OUTPUT, MOCK_EVALUATION)

    # Create two check-ins
    await client.post("/checkins", json=SAMPLE_CHECKIN)
    await client.post("/checkins", json=SAMPLE_CHECKIN)

    response = await client.get("/checkins")
    assert response.status_code == 200

    data = response.json()
    assert len(data) == 2
    assert all("id" in item for item in data)


@pytest.mark.asyncio
async def test_create_checkin_validation_error(client):
    """POST /checkins with missing fields should return 422."""
    response = await client.post("/checkins", json={"health_goal": "test"})
    assert response.status_code == 422
