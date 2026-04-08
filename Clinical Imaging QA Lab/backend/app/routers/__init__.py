from app.routers.captures import router as captures_router
from app.routers.defects import router as defects_router
from app.routers.device import router as device_router
from app.routers.dashboard import router as dashboard_router

__all__ = ["captures_router", "defects_router", "device_router", "dashboard_router"]
