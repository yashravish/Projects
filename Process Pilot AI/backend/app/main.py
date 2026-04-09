from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.logging_config import setup_logging
from app.routers import analytics, auth, requests

setup_logging(settings.log_level)

app = FastAPI(
    title="ProcessPilot AI",
    description="Business Process Modernization API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(requests.router)
app.include_router(analytics.router)


@app.get("/api/health")
def health_check():
    return {"status": "healthy", "service": "ProcessPilot AI"}
