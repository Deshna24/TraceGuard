"""WebSocket connection manager with structured event envelope broadcasting.

Uses the EventEnvelope schema for every message. No security decisions are
made here — this is an observation/control channel only.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from fastapi import WebSocket

from web.schemas.api_models import EventEnvelope, EventType

logger = logging.getLogger("traceguard.web.websocket")


class ConnectionManager:
    """Manages active WebSocket connections and broadcasts structured events."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(
            "WebSocket connected. Total connections: %d",
            len(self.active_connections),
        )

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(
            "WebSocket disconnected. Total connections: %d",
            len(self.active_connections),
        )

    async def broadcast_event(
        self,
        run_id: str,
        event_type: EventType,
        payload: dict[str, Any] | None = None,
        step: int | None = None,
    ) -> EventEnvelope:
        """Create an EventEnvelope and send it to every connected client.

        Returns the envelope that was broadcast so callers can log or store it.
        """
        envelope = EventEnvelope(
            run_id=run_id,
            event_type=event_type,
            step=step,
            payload=payload or {},
        )
        message = json.dumps(envelope.to_json_dict())
        disconnected: list[WebSocket] = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(connection)
        for ws in disconnected:
            self.disconnect(ws)
        return envelope

    @property
    def connection_count(self) -> int:
        return len(self.active_connections)


# Module-level singleton used across the application.
manager = ConnectionManager()


__all__ = ["ConnectionManager", "manager"]
