"""Daily reconcile endpoints — post-close SIP replay vs live comparison."""
from fastapi import APIRouter
from live.persistence_db import TradingDatabase

router = APIRouter(tags=["reconcile"])

db = TradingDatabase()


@router.get("/reconcile/daily")
async def get_reconcile():
    """Latest daily reconcile records (newest first), with parsed details."""
    try:
        rows = db.get_daily_reconcile(limit=30)
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["details"] = __import__("json").loads(d.get("details") or "{}")
            except Exception:
                d["details"] = {}
            out.append(d)
        return out
    except Exception as e:
        return {"error": str(e)}


@router.get("/reconcile/daily/{date}")
async def get_reconcile_by_date(date: str):
    """Reconcile record for a specific date."""
    try:
        rows = db.get_daily_reconcile(reconcile_date=date)
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["details"] = __import__("json").loads(d.get("details") or "{}")
            except Exception:
                d["details"] = {}
            out.append(d)
        return out
    except Exception as e:
        return {"error": str(e)}
