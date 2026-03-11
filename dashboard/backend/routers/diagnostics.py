"""Diagnostics endpoint — why each ticker didn't trade."""
from fastapi import APIRouter

router = APIRouter(tags=["diagnostics"])


@router.get("/diagnostics")
async def get_diagnostics():
    from dashboard.backend.app import bridge
    return bridge.get_diagnostics()
