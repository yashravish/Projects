"""Reporting schemas."""
from __future__ import annotations

from typing import List

from pydantic import BaseModel

from app.models.approval import ApprovalDecision
from app.models.task import TaskPriority, TaskStatus
from app.models.user import UserRole


class StatusCount(BaseModel):
    status: TaskStatus
    count: int


class PriorityCount(BaseModel):
    priority: TaskPriority
    count: int


class UserWorkloadEntry(BaseModel):
    user_id: int
    name: str
    email: str
    role: UserRole
    total: int
    pending: int
    in_progress: int
    submitted: int
    approved: int
    rejected: int
    completed: int


class ApprovalSummaryEntry(BaseModel):
    decision: ApprovalDecision
    count: int


class TasksByStatusReport(BaseModel):
    items: List[StatusCount]
    total: int


class TasksByPriorityReport(BaseModel):
    items: List[PriorityCount]
    total: int


class UserWorkloadReport(BaseModel):
    items: List[UserWorkloadEntry]


class ApprovalSummaryReport(BaseModel):
    items: List[ApprovalSummaryEntry]
    total: int
