"""Strategy performance endpoints."""
import json
import os
from fastapi import APIRouter

router = APIRouter(tags=["strategies"])

# Strategy metadata — add new strategies here as they're created
STRATEGY_INFO = {
    "H": {"name": "Candle 1 Breakout (High)", "description": "First green candle breaks premarket high immediately", "category": "momentum"},
    "G": {"name": "Candle 1 Breakout (Gap)", "description": "First green candle with gap confirmation", "category": "momentum"},
    "A": {"name": "Candle 1 Breakout (ATR)", "description": "First green candle with ATR-based targets", "category": "momentum"},
    "F": {"name": "Candle 1 Breakout (Follow)", "description": "Follow-through on first candle momentum", "category": "momentum"},
    "D": {"name": "Dip Buy", "description": "Gap up + spike + dip below VWAP = buy the dip", "category": "reversal"},
    "V": {"name": "VWAP Bounce", "description": "Price dips below VWAP then reclaims with volume", "category": "reversal"},
    "P": {"name": "Pullback + PM High Break", "description": "Confirm above PM high, pullback, then re-break", "category": "breakout"},
    "M": {"name": "Midday Consolidation", "description": "Morning spike + tight range = afternoon breakout", "category": "breakout"},
    "R": {"name": "Day 2 Runner", "description": "Multi-day momentum: day 1 runner, day 2 continuation", "category": "multiday"},
    "W": {"name": "Late Day Breakout", "description": "Afternoon HOD break with volume surge", "category": "breakout"},
    "O": {"name": "Opening Range Breakout", "description": "First 10-min range breakout with volume", "category": "breakout"},
    "B": {"name": "Red-to-Green (R2G)", "description": "Opens red, reclaims open price with volume surge", "category": "reversal"},
    "K": {"name": "First Pullback Buy", "description": "Morning run-up + orderly pullback + bounce", "category": "pullback"},
    "C": {"name": "Micro Flag / Base", "description": "Initial spike + tight consolidation + breakout", "category": "pattern"},
    "S": {"name": "Stuff-and-Break", "description": "Multiple HOD rejections then final breakout", "category": "breakout"},
    "E": {"name": "Gap-and-Go RelVol", "description": "Extreme premarket volume = immediate momentum entry", "category": "momentum"},
    "I": {"name": "PM High Immediate Break", "description": "Breaks premarket high within first few candles", "category": "breakout"},
    "J": {"name": "VWAP + PM High Break", "description": "Near VWAP + breaks premarket high = strong confirmation", "category": "breakout"},
    "N": {"name": "HOD Reclaim", "description": "Old HOD reclaim after pullback = continuation", "category": "breakout"},
    "L": {"name": "Low Float Squeeze", "description": "Low float + HOD break + volume surge = squeeze play", "category": "squeeze"},
}


@router.get("/strategies")
async def get_strategies():
    """Get strategy performance stats for today + metadata."""
    from dashboard.backend.app import bridge
    stats = bridge.get_strategy_stats()

    # Merge with metadata
    result = []
    for code, info in STRATEGY_INFO.items():
        s = stats.get(code, {"trades": 0, "wins": 0, "pnl": 0.0, "win_rate": 0, "best": 0, "worst": 0})
        result.append({
            "code": code,
            **info,
            **s,
        })
    return result


@router.get("/strategies/config")
async def get_strategy_config():
    """Get current strategy enable/disable and priority from trial params."""
    params_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        "config", "trial_432_params.json"
    )
    try:
        with open(params_path) as f:
            params = json.load(f)
    except FileNotFoundError:
        return {"error": "No trial params found"}

    config = []
    for code in STRATEGY_INFO:
        key = code.lower()
        config.append({
            "code": code,
            "name": STRATEGY_INFO[code]["name"],
            "enabled": params.get(f"enable_{key}", False),
            "priority": params.get(f"priority_{key}", 99),
        })
    # Sort by priority
    config.sort(key=lambda x: x["priority"])
    return config
