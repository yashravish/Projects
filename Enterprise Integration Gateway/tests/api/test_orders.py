"""API tests for the /orders endpoints."""
from sqlalchemy.orm import sessionmaker

from app.models.order import Order


def _seed_order(db, external_id="ORD-001", source="crm", status="pending", customer_id=None):
    order = Order(
        external_id=external_id,
        source=source,
        order_number=external_id,
        status=status,
        total_amount=100.00,
        currency="USD",
        customer_id=customer_id,
    )
    db.add(order)
    db.commit()
    db.refresh(order)
    return order


class TestListOrders:
    def test_empty_returns_list(self, client):
        response = client.get("/api/v1/orders")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_returns_seeded_order(self, client, engine):
        Session = sessionmaker(bind=engine)
        db = Session()
        _seed_order(db)
        db.close()

        data = client.get("/api/v1/orders").json()
        assert len(data) == 1

    def test_filter_by_source(self, client, engine):
        Session = sessionmaker(bind=engine)
        db = Session()
        _seed_order(db, external_id="CRM-ORD-1", source="crm")
        _seed_order(db, external_id="VND-ORD-1", source="vendor")
        db.close()

        data = client.get("/api/v1/orders?source=vendor").json()
        assert all(o["source"] == "vendor" for o in data)

    def test_filter_by_status(self, client, engine):
        Session = sessionmaker(bind=engine)
        db = Session()
        _seed_order(db, external_id="SHP-1", status="shipped")
        _seed_order(db, external_id="PND-1", status="pending")
        db.close()

        data = client.get("/api/v1/orders?status=shipped").json()
        assert all(o["status"] == "shipped" for o in data)


class TestGetOrder:
    def test_existing_order(self, client, engine):
        Session = sessionmaker(bind=engine)
        db = Session()
        order = _seed_order(db)
        oid = order.id
        db.close()

        data = client.get(f"/api/v1/orders/{oid}").json()
        assert data["external_id"] == "ORD-001"

    def test_nonexistent_returns_404(self, client):
        response = client.get("/api/v1/orders/99999")
        assert response.status_code == 404
