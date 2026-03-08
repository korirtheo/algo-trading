"""Watchlist endpoints."""
from fastapi import APIRouter

router = APIRouter(tags=["watchlist"])


@router.get("/watchlist")
async def get_watchlist():
    from dashboard.backend.app import bridge
    return bridge.get_watchlist()
