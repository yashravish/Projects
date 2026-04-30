"""Workflow / status transition tests."""
from __future__ import annotations

from app.models.task import TaskStatus
from app.services.workflow_service import VALID_TRANSITIONS, WorkflowService
from app.tests.conftest import auth_header


def _create_task(client, token, seeded_users):
    payload = {
        "title": "Audit accounts",
        "description": "",
        "priority": "high",
        "assigned_to_user_id": seeded_users["analyst"].id,
    }
    return client.post("/tasks", json=payload, headers=auth_header(token))


def test_valid_transitions_table_matches_spec():
    expected = {
        TaskStatus.PENDING: {TaskStatus.IN_PROGRESS, TaskStatus.SUBMITTED},
        TaskStatus.IN_PROGRESS: {TaskStatus.SUBMITTED},
        TaskStatus.SUBMITTED: {TaskStatus.APPROVED, TaskStatus.REJECTED},
        TaskStatus.APPROVED: {TaskStatus.COMPLETED},
        TaskStatus.REJECTED: {TaskStatus.IN_PROGRESS, TaskStatus.SUBMITTED},
        TaskStatus.COMPLETED: set(),
    }
    assert VALID_TRANSITIONS == expected


def test_is_valid_transition_helper():
    assert WorkflowService.is_valid_transition(TaskStatus.PENDING, TaskStatus.SUBMITTED)
    assert not WorkflowService.is_valid_transition(TaskStatus.PENDING, TaskStatus.COMPLETED)
    assert not WorkflowService.is_valid_transition(TaskStatus.COMPLETED, TaskStatus.PENDING)


def test_submit_then_approve_full_flow(client, seeded_users, analyst_token, manager_token):
    response = _create_task(client, analyst_token, seeded_users)
    task_id = response.json()["id"]
    submit = client.post(f"/tasks/{task_id}/submit", headers=auth_header(analyst_token))
    assert submit.status_code == 200
    assert submit.json()["status"] == "submitted"

    approve = client.post(
        f"/tasks/{task_id}/approve",
        json={"comment": "looks good"},
        headers=auth_header(manager_token),
    )
    assert approve.status_code == 200
    assert approve.json()["status"] == "approved"

    complete = client.post(
        f"/tasks/{task_id}/complete", headers=auth_header(manager_token)
    )
    assert complete.status_code == 200
    assert complete.json()["status"] == "completed"


def test_rejected_task_can_be_resubmitted(
    client, seeded_users, analyst_token, manager_token
):
    response = _create_task(client, analyst_token, seeded_users)
    task_id = response.json()["id"]
    client.post(f"/tasks/{task_id}/submit", headers=auth_header(analyst_token))
    reject = client.post(
        f"/tasks/{task_id}/reject",
        json={"comment": "Need more detail"},
        headers=auth_header(manager_token),
    )
    assert reject.status_code == 200
    assert reject.json()["status"] == "rejected"

    resubmit = client.post(
        f"/tasks/{task_id}/submit", headers=auth_header(analyst_token)
    )
    assert resubmit.status_code == 200
    assert resubmit.json()["status"] == "submitted"


def test_cannot_complete_pending_task(
    client, seeded_users, analyst_token, manager_token
):
    response = _create_task(client, analyst_token, seeded_users)
    task_id = response.json()["id"]
    complete = client.post(
        f"/tasks/{task_id}/complete", headers=auth_header(manager_token)
    )
    assert complete.status_code == 400


def test_cannot_approve_pending_task(client, seeded_users, analyst_token, manager_token):
    response = _create_task(client, analyst_token, seeded_users)
    task_id = response.json()["id"]
    approve = client.post(
        f"/tasks/{task_id}/approve", headers=auth_header(manager_token)
    )
    assert approve.status_code == 400


def test_cannot_edit_completed_task(
    client, seeded_users, analyst_token, manager_token
):
    response = _create_task(client, analyst_token, seeded_users)
    task_id = response.json()["id"]
    client.post(f"/tasks/{task_id}/submit", headers=auth_header(analyst_token))
    client.post(f"/tasks/{task_id}/approve", headers=auth_header(manager_token))
    client.post(f"/tasks/{task_id}/complete", headers=auth_header(manager_token))
    update = client.put(
        f"/tasks/{task_id}",
        json={"title": "should fail"},
        headers=auth_header(manager_token),
    )
    assert update.status_code == 400


def test_workflow_event_history_endpoint(
    client, seeded_users, analyst_token, manager_token
):
    response = _create_task(client, analyst_token, seeded_users)
    task_id = response.json()["id"]
    client.post(f"/tasks/{task_id}/submit", headers=auth_header(analyst_token))
    client.post(f"/tasks/{task_id}/approve", headers=auth_header(manager_token))

    history = client.get(
        f"/tasks/{task_id}/workflow-events", headers=auth_header(analyst_token)
    )
    assert history.status_code == 200
    events = history.json()
    transitions = [(e["from_status"], e["to_status"]) for e in events]
    assert transitions[0] == (None, "pending")
    assert (None, "pending") in transitions
    assert ("pending", "submitted") in transitions
    assert ("submitted", "approved") in transitions
