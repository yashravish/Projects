"""Mounts all v1 routers."""
from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import audit as audit_routes
from app.api.v1 import auth as auth_routes
from app.api.v1 import documents as document_routes
from app.api.v1 import evaluations as evaluation_routes
from app.api.v1 import health as health_routes
from app.api.v1 import query as query_routes
from app.api.v1 import training as training_routes

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(auth_routes.router)
api_v1_router.include_router(document_routes.router)
api_v1_router.include_router(query_routes.router)
api_v1_router.include_router(evaluation_routes.router)
api_v1_router.include_router(training_routes.router)
api_v1_router.include_router(audit_routes.router)

# Health is intentionally mounted at the app root (not under /api/v1) — it is
# wired in app.main so external orchestration can hit /health directly.
__all__ = ["api_v1_router", "health_routes"]
