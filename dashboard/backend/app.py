"""
FastAPI Dashboard Server.
Serves the trading dashboard API and WebSocket endpoints.
Embeds alongside the live trading engine.
"""
import os
import json
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from dashboard.backend.services.ws_manager import ws_manager
from dashboard.backend.services.engine_bridge import EngineBridge
from dashboard.backend.routers import account, positions, watchlist, trades, charts, strategies

log = logging.getLogger(__name__)

# Global bridge — set by live/main.py when starting the dashboard
bridge = EngineBridge()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Dashboard server starting")
    yield
    log.info("Dashboard server shutting down")


app = FastAPI(title="AlgoTrader Dashboard", lifespan=lifespan)

# CORS for local development (React dev server on port 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(account.router, prefix="/api")
app.include_router(positions.router, prefix="/api")
app.include_router(watchlist.router, prefix="/api")
app.include_router(trades.router, prefix="/api")
app.include_router(charts.router, prefix="/api")
app.include_router(strategies.router, prefix="/api")


@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    """WebSocket endpoint for real-time dashboard updates."""
    await ws_manager.connect(ws)
    try:
        while True:
            # Keep connection alive, handle client messages
            data = await ws.receive_text()
            # Client can request specific data
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await ws.send_json({"type": "pong"})
                elif msg.get("type") == "subscribe_chart":
                    symbol = msg.get("symbol", "")
                    chart_data = bridge.get_chart_data(symbol)
                    await ws.send_json({"type": "chart_data", "data": chart_data})
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)
    except Exception:
        ws_manager.disconnect(ws)


# Serve frontend static files (production build)
_frontend_dist = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dist")
if os.path.isdir(_frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(_frontend_dist, "assets")), name="assets")

    @app.get("/{path:path}")
    async def serve_frontend(path: str):
        """Serve React SPA — all non-API routes go to index.html."""
        file_path = os.path.join(_frontend_dist, path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(_frontend_dist, "index.html"))


def get_bridge() -> EngineBridge:
    """Dependency injection for routers."""
    return bridge
