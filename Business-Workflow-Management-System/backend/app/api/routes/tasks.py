"""Task-related API routes - thin handlers that delegate to TaskService."""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, Query, Response, status as http_status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, require_manager
from app.db.session import get_db
from app.models.task import TaskPriority, TaskStatus
from app.models.user import User
from app.schemas.task import TaskActionRequest, TaskCreate, TaskRead, TaskUpdate
from app.schemas.workflow_event import WorkflowEventRead
from app.services.task_service import TaskService


router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("", response_model=List[TaskRead])
def list_tasks(
    status: Optional[TaskStatus] = Query(default=None),
    priority: Optional[TaskPriority] = Query(default=None),
    assigned_to_user_id: Optional[int] = Query(default=None),
    db: Session = Depends(get_db),
    _current_user: User = Depends(get_current_user),
) -> List[TaskRead]:
    service = TaskService(db)
    tasks = service.list_tasks(
        status=status,
        priority=priority,
        assigned_to_user_id=assigned_to_user_id,
    )
    return [TaskRead.model_validate(t) for t in tasks]


@router.post("", response_model=TaskRead, status_code=http_status.HTTP_201_CREATED)
def create_task(
    payload: TaskCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TaskRead:
    service = TaskService(db)
    task = service.create_task(payload, current_user)
    return TaskRead.model_validate(task)


@router.get("/{task_id}", response_model=TaskRead)
def get_task(
    task_id: int,
    db: Session = Depends(get_db),
    _current_user: User = Depends(get_current_user),
) -> TaskRead:
    service = TaskService(db)
    task = service.get_task_or_404(task_id)
    return TaskRead.model_validate(task)


@router.put("/{task_id}", response_model=TaskRead)
def update_task(
    task_id: int,
    payload: TaskUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TaskRead:
    service = TaskService(db)
    task = service.update_task(task_id, payload, current_user)
    return TaskRead.model_validate(task)


@router.delete("/{task_id}", status_code=http_status.HTTP_204_NO_CONTENT)
def delete_task(
    task_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Response:
    service = TaskService(db)
    service.delete_task(task_id, current_user)
    return Response(status_code=http_status.HTTP_204_NO_CONTENT)


@router.post("/{task_id}/submit", response_model=TaskRead)
def submit_task(
    task_id: int,
    payload: Optional[TaskActionRequest] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TaskRead:
    service = TaskService(db)
    task = service.submit_task(
        task_id,
        current_user,
        comment=payload.comment if payload else None,
    )
    return TaskRead.model_validate(task)


@router.post("/{task_id}/approve", response_model=TaskRead)
def approve_task(
    task_id: int,
    payload: Optional[TaskActionRequest] = None,
    db: Session = Depends(get_db),
    manager: User = Depends(require_manager),
) -> TaskRead:
    service = TaskService(db)
    task = service.approve_task(
        task_id,
        manager,
        comment=payload.comment if payload else None,
    )
    return TaskRead.model_validate(task)


@router.post("/{task_id}/reject", response_model=TaskRead)
def reject_task(
    task_id: int,
    payload: Optional[TaskActionRequest] = None,
    db: Session = Depends(get_db),
    manager: User = Depends(require_manager),
) -> TaskRead:
    service = TaskService(db)
    task = service.reject_task(
        task_id,
        manager,
        comment=payload.comment if payload else None,
    )
    return TaskRead.model_validate(task)


@router.post("/{task_id}/complete", response_model=TaskRead)
def complete_task(
    task_id: int,
    payload: Optional[TaskActionRequest] = None,
    db: Session = Depends(get_db),
    manager: User = Depends(require_manager),
) -> TaskRead:
    service = TaskService(db)
    task = service.complete_task(
        task_id,
        manager,
        comment=payload.comment if payload else None,
    )
    return TaskRead.model_validate(task)


@router.get("/{task_id}/workflow-events", response_model=List[WorkflowEventRead])
def list_workflow_events(
    task_id: int,
    db: Session = Depends(get_db),
    _current_user: User = Depends(get_current_user),
) -> List[WorkflowEventRead]:
    service = TaskService(db)
    service.get_task_or_404(task_id)
    events = service.workflow.get_task_history(task_id)
    return [WorkflowEventRead.model_validate(e) for e in events]
