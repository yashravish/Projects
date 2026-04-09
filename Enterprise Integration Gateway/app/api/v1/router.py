from fastapi import APIRouter

from app.api.v1 import (
    customers,
    failed_records,
    health,
    jobs,
    metrics,
    orders,
    shipments,
    sync,
)

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health"])
api_router.include_router(metrics.router, tags=["Metrics"])
api_router.include_router(customers.router, prefix="/customers", tags=["Customers"])
api_router.include_router(orders.router, prefix="/orders", tags=["Orders"])
api_router.include_router(shipments.router, prefix="/shipments", tags=["Shipments"])
api_router.include_router(sync.router, prefix="/sync", tags=["Sync"])
api_router.include_router(jobs.router, prefix="/integration-jobs", tags=["Integration Jobs"])
api_router.include_router(failed_records.router, prefix="/failed-records", tags=["Failed Records"])
