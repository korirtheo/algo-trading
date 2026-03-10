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

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        """Store the uvicorn event loop so broadcast_sync can schedule into it."""
        self._loop = loop

    def broadcast_sync(self, message: Dict[str, Any]):
        """Non-async broadcast — schedules into the uvicorn event loop.
        Call this from the engine's bar callback (runs in streamer thread).
        """
        loop = getattr(self, '_loop', None)
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(self.broadcast(message), loop)


# Singleton
ws_manager = WSManager()
