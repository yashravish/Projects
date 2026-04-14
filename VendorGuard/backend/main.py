import structlog
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from backend.config import get_settings
from backend.routers import auth, vendors, assessments, findings, remediation, reports, dashboard, templates_router, pages

settings = get_settings()

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

app = FastAPI(
    title="VendorGuard",
    description="Enterprise Third-Party Security Assessment Platform",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(auth.router)
app.include_router(vendors.router)
app.include_router(assessments.router)
app.include_router(findings.router)
app.include_router(remediation.router)
app.include_router(reports.router)
app.include_router(dashboard.router)
app.include_router(templates_router.router)
app.include_router(pages.router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger = structlog.get_logger()
    logger.error("unhandled_exception", path=str(request.url), error=str(exc))
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/api/health")
def health_check():
    return {"status": "healthy", "version": "1.0.0"}
