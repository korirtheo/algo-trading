"""Current positions endpoints."""
from fastapi import APIRouter

router = APIRouter(tags=["positions"])


@router.get("/positions")
async def get_positions():
    from dashboard.backend.app import bridge
    return bridge.get_positions()
