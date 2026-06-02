"""Slippage calibration endpoints — read logs/fills_calibration.csv."""
from fastapi import APIRouter

router = APIRouter(tags=["slippage"])


@router.get("/slippage/recent")
async def get_slippage_recent(limit: int = 100):
    """Return the most recent N fills with their slippage detail + aggregates."""
    from dashboard.backend.app import bridge
    return bridge.get_slippage_data(limit=limit)
