"""Trade history endpoints."""
from fastapi import APIRouter
from live.persistence_db import TradingDatabase

router = APIRouter(tags=["trades"])

# Initialize database connection
db = TradingDatabase()


@router.get("/trades/today")
async def get_trades_today():
    """Get today's trades from database."""
    try:
        trades = db.get_trades_today()
        # Format for frontend
        formatted = []
        for t in trades:
            formatted.append({
                "ticker": t["ticker"],
                "strategy": t["strategy"],
                "entry_price": t["entry_price"],
                "exit_price": t["exit_price"],
                "shares": t["shares"],
                "deployed_amount": t["shares"] * t["entry_price"],
                "pnl": t["pnl"],
                "pnl_pct": t["pnl_pct"],
                "reason": t["reason"],
                "entry_time": t["entry_time"],
                "exit_time": t["exit_time"],
            })
        return formatted
    except Exception as e:
        return []


@router.get("/trades/{date}")
async def get_trades_by_date(date: str):
    """Get trades for a specific date (YYYY-MM-DD format) from database."""
    try:
        trades = db.get_trades_by_date(date)
        # Format for frontend
        formatted = []
        for t in trades:
            formatted.append({
                "ticker": t["ticker"],
                "strategy": t["strategy"],
                "entry_price": t["entry_price"],
                "exit_price": t["exit_price"],
                "shares": t["shares"],
                "deployed_amount": t["shares"] * t["entry_price"],
                "pnl": t["pnl"],
                "pnl_pct": t["pnl_pct"],
                "reason": t["reason"],
                "entry_time": t["entry_time"],
                "exit_time": t["exit_time"],
            })
        return {"trades": formatted, "date": date, "found": len(trades) > 0}
    except Exception as e:
        return {"trades": [], "date": date, "found": False, "error": str(e)}
