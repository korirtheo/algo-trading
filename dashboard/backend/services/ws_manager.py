"""
WebSocket connection manager.
Broadcasts real-time updates to all connected dashboard clients.
"""
import json
import asyncio
import logging
from typing import Dict, Any
from fastapi import WebSocket

log = logging.getLogger(__name__)


class WSManager:
    """Manages WebSocket connections and broadcasts updates."""

    def __init__(self):
        self.connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)
        log.info(f"Dashboard client connected ({len(self.connections)} total)")

    def disconnect(self, ws: WebSocket):
        if ws in self.connections:
            self.connections.remove(ws)
        log.info(f"Dashboard client disconnected ({len(self.connections)} total)")

    async def broadcast(self, message: Dict[str, Any]):
        """Send a message to all connected clients."""
        if not self.connections:
            return
        data = json.dumps(message, default=str)
        disconnected = []
        for ws in self.connections:
            try:
                await ws.send_text(data)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self.disconnect(ws)

    def broadcast_sync(self, message: Dict[str, Any]):
        """Non-async broadcast — schedules into the event loop.
        Call this from the engine's bar callback (runs in streamer thread).
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(self.broadcast(message), loop)
            else:
                loop.run_until_complete(self.broadcast(message))
        except RuntimeError:
            pass  # No event loop available yet


# Singleton
ws_manager = WSManager()
