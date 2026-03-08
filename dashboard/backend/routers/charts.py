"""Chart data endpoints."""
from fastapi import APIRouter

router = APIRouter(tags=["charts"])


@router.get("/charts/{symbol}")
async def get_chart(symbol: str):
    from dashboard.backend.app import bridge
    return bridge.get_chart_data(symbol.upper())
