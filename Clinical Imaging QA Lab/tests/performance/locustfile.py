"""Locust load test for the Clinical Imaging QA Lab backend."""
from locust import HttpUser, task, between


class ClinicalImagingUser(HttpUser):
    """Simulates a user interacting with the clinical imaging backend."""

    wait_time = between(0.5, 2.0)
    host = "http://localhost:8000"

    @task(5)
    def view_dashboard(self):
        """Simulate dashboard summary requests (most frequent action)."""
        self.client.get("/api/dashboard/summary", name="/api/dashboard/summary")

    @task(3)
    def view_history(self):
        """Simulate capture history listing."""
        self.client.get("/api/captures", name="/api/captures")

    @task(2)
    def check_device_status(self):
        """Simulate device status polling."""
        self.client.get("/api/device/status", name="/api/device/status")

    @task(2)
    def submit_capture(self):
        """Simulate capture submission."""
        self.client.post(
            "/api/captures",
            json={
                "patient_id": "PAT-LOAD-001",
                "session_id": "SESS-LOAD-001",
                "image_type": "x-ray",
            },
            name="/api/captures [POST]",
        )

    @task(1)
    def view_defects(self):
        """Simulate defect listing."""
        self.client.get("/api/defects", name="/api/defects")

    @task(1)
    def submit_defect(self):
        """Simulate defect submission."""
        self.client.post(
            "/api/defects",
            json={
                "title": "Load test defect",
                "severity": "trivial",
                "priority": "low",
            },
            name="/api/defects [POST]",
        )

    @task(1)
    def check_health(self):
        """Simulate health check."""
        self.client.get("/api/health", name="/api/health")
