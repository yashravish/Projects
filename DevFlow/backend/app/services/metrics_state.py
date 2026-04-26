from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from collections import deque


@dataclass
class MetricsState:
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    pipeline_success_total: int = 0
    pipeline_failure_total: int = 0
    deployment_success_total: int = 0
    deployment_failure_total: int = 0
    api_request_count: int = 0
    _pipeline_durations: deque[float] = field(
        default_factory=lambda: deque(maxlen=10_000)
    )

    def record_api_request(self) -> None:
        with self._lock:
            self.api_request_count += 1

    def record_pipeline(self, success: bool, duration_seconds: float) -> None:
        with self._lock:
            if success:
                self.pipeline_success_total += 1
            else:
                self.pipeline_failure_total += 1
            self._pipeline_durations.append(max(0.0, duration_seconds))

    def record_deployment(self, success: bool) -> None:
        with self._lock:
            if success:
                self.deployment_success_total += 1
            else:
                self.deployment_failure_total += 1

    @property
    def average_pipeline_duration_seconds(self) -> float:
        with self._lock:
            if not self._pipeline_durations:
                return 0.0
            return sum(self._pipeline_durations) / len(self._pipeline_durations)

    def to_prometheus_text(self) -> str:
        with self._lock:
            p_ok = self.pipeline_success_total
            p_fail = self.pipeline_failure_total
            d_ok = self.deployment_success_total
            d_fail = self.deployment_failure_total
            api = self.api_request_count
            avg = self.average_pipeline_duration_seconds
        lines = [
            "# HELP devflow_pipeline_success_total Count of successful pipeline runs.",
            "# TYPE devflow_pipeline_success_total counter",
            f"devflow_pipeline_success_total {p_ok}",
            "# HELP devflow_pipeline_failure_total Count of failed pipeline runs.",
            "# TYPE devflow_pipeline_failure_total counter",
            f"devflow_pipeline_failure_total {p_fail}",
            "# HELP devflow_deployment_success_total Count of successful deployments.",
            "# TYPE devflow_deployment_success_total counter",
            f"devflow_deployment_success_total {d_ok}",
            "# HELP devflow_deployment_failure_total Count of failed or rolled back deployments.",
            "# TYPE devflow_deployment_failure_total counter",
            f"devflow_deployment_failure_total {d_fail}",
            "# HELP devflow_api_request_count Total number of API requests (HTTP middleware).",
            "# TYPE devflow_api_request_count counter",
            f"devflow_api_request_count {api}",
            "# HELP devflow_average_pipeline_duration_seconds Rolling average of pipeline durations in seconds.",
            "# TYPE devflow_average_pipeline_duration_seconds gauge",
            f"devflow_average_pipeline_duration_seconds {avg:.6f}",
        ]
        return "\n".join(lines) + "\n"

    def to_dashboard(self) -> dict:
        with self._lock:
            return {
                "pipeline_success_total": self.pipeline_success_total,
                "pipeline_failure_total": self.pipeline_failure_total,
                "deployment_success_total": self.deployment_success_total,
                "deployment_failure_total": self.deployment_failure_total,
                "api_request_count": self.api_request_count,
                "average_pipeline_duration_seconds": round(
                    self.average_pipeline_duration_seconds, 4
                ),
            }


global_metrics = MetricsState()
