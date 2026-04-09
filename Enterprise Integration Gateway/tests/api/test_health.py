"""API tests for the health check endpoint."""


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_returns_healthy_status(self, client):
        data = response = client.get("/api/v1/health").json()
        assert data["status"] == "healthy"

    def test_health_includes_version(self, client):
        data = client.get("/api/v1/health").json()
        assert "version" in data

    def test_health_includes_database_check(self, client):
        data = client.get("/api/v1/health").json()
        assert "checks" in data
        assert "database" in data["checks"]

    def test_health_db_ok(self, client):
        data = client.get("/api/v1/health").json()
        assert data["checks"]["database"] == "ok"
