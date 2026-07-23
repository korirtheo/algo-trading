"""System Health endpoint — exposes health check results for dashboard display."""
from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def get_health():
    from dashboard.backend.app import bridge
    checks = getattr(bridge, "system_health", [])
    healthy = all(c["status"] != "error" for c in checks) if checks else None
    return {"healthy": healthy, "checks": checks}
