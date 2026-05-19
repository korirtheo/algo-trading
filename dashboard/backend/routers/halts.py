"""Halt-resume monitor endpoints."""
from fastapi import APIRouter

router = APIRouter(tags=["halts"])


@router.get("/halts/today")
async def get_halts_today():
    from dashboard.backend.app import bridge
    return bridge.get_halt_events()
