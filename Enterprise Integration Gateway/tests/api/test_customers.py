"""API tests for the /customers endpoints."""
from sqlalchemy.orm import sessionmaker

from app.models.customer import Customer


def _create_customer(db, external_id="EXT-001", source="crm", name="Alice", email="alice@test.com"):
    customer = Customer(
        external_id=external_id,
        source=source,
        name=name,
        email=email,
        status="active",
    )
    db.add(customer)
    db.commit()
    db.refresh(customer)
    return customer


class TestListCustomers:
    def test_empty_returns_list(self, client):
        response = client.get("/api/v1/customers")
        assert response.status_code == 200
        assert response.json() == []

    def test_returns_created_customer(self, client, engine):
        # Create a customer using a session from the same engine as the client
        Session = sessionmaker(bind=engine)
        db = Session()
        _create_customer(db)
        db.close()

        response = client.get("/api/v1/customers")
        assert response.status_code == 200
        assert len(response.json()) == 1

    def test_filter_by_source(self, client, engine):
        Session = sessionmaker(bind=engine)
        db = Session()
        _create_customer(db, external_id="CRM-1", source="crm")
        _create_customer(db, external_id="VND-1", source="vendor", name="Bob")
        db.close()

        response = client.get("/api/v1/customers?source=crm")
        data = response.json()
        assert all(c["source"] == "crm" for c in data)

    def test_filter_by_status(self, client, engine):
        Session = sessionmaker(bind=engine)
        db = Session()
        _create_customer(db, external_id="ACT-1", name="Active")
        customer = Customer(external_id="INA-1", source="crm", name="Inactive", status="inactive")
        db.add(customer)
        db.commit()
        db.close()

        response = client.get("/api/v1/customers?status=inactive")
        data = response.json()
        assert all(c["status"] == "inactive" for c in data)

    def test_pagination_limit(self, client, engine):
        Session = sessionmaker(bind=engine)
        db = Session()
        for i in range(5):
            _create_customer(db, external_id=f"CUST-{i}", name=f"Customer {i}", email=f"c{i}@test.com")
        db.close()

        response = client.get("/api/v1/customers?limit=2")
        assert len(response.json()) == 2


class TestGetCustomer:
    def test_existing_customer(self, client, engine):
        Session = sessionmaker(bind=engine)
        db = Session()
        customer = _create_customer(db)
        cid = customer.id
        db.close()

        response = client.get(f"/api/v1/customers/{cid}")
        assert response.status_code == 200
        assert response.json()["external_id"] == "EXT-001"

    def test_nonexistent_returns_404(self, client):
        response = client.get("/api/v1/customers/99999")
        assert response.status_code == 404

    def test_response_includes_timestamps(self, client, engine):
        Session = sessionmaker(bind=engine)
        db = Session()
        customer = _create_customer(db)
        cid = customer.id
        db.close()

        data = client.get(f"/api/v1/customers/{cid}").json()
        assert "created_at" in data
        assert "updated_at" in data
