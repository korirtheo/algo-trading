"""Trade history endpoints."""
from fastapi import APIRouter

router = APIRouter(tags=["trades"])


@router.get("/trades/today")
async def get_trades_today():
    from dashboard.backend.app import bridge
    return bridge.get_trades_today()
