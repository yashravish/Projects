from pydantic import BaseModel


class DashboardMetrics(BaseModel):
    application: str = "devflow"
    from_metrics_state: dict
    from_metrics_events_sample: list
