"""
Mock External Providers Service.

Runs as a separate FastAPI application on port 8001.
Exposes:
  - /mock/crm/*   — CRM JSON endpoints
  - /mock/vendor/* — Vendor XML endpoints
  - /health       — health check

In Docker Compose the main app connects to this service at http://mock_providers:8001.
"""
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from mock_providers.crm.router import router as crm_router
from mock_providers.vendor.router import router as vendor_router

app = FastAPI(
    title="Mock External Providers",
    version="1.0.0",
    description="Simulated CRM (JSON) and Vendor (XML) services for integration testing.",
    docs_url="/docs",
)

app.include_router(crm_router, prefix="/mock/crm", tags=["Mock CRM"])
app.include_router(vendor_router, prefix="/mock/vendor", tags=["Mock Vendor"])


@app.get("/health", tags=["Health"])
def health():
    return JSONResponse(content={"status": "healthy", "service": "mock_providers"})
