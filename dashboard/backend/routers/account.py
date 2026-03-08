"""Account & portfolio endpoints."""
from fastapi import APIRouter

router = APIRouter(tags=["account"])


@router.get("/account")
async def get_account():
    from dashboard.backend.app import bridge
    return bridge.get_account()


@router.get("/summary")
async def get_summary():
    from dashboard.backend.app import bridge
    return bridge.get_engine_summary()
