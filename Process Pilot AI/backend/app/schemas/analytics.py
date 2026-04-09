from pydantic import BaseModel


class AnalyticsOverview(BaseModel):
    total_requests: int
    open_requests: int
    closed_requests: int
    avg_priority: float
    requests_this_week: int


class CategoryCount(BaseModel):
    category: str
    count: int


class DepartmentCount(BaseModel):
    department: str
    count: int


class PriorityCount(BaseModel):
    priority_range: str
    count: int


class StatusCount(BaseModel):
    status: str
    count: int


class PainPoint(BaseModel):
    description: str
    count: int
    category: str
