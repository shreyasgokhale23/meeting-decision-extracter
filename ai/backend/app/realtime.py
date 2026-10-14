import asyncio
import logging
from collections import defaultdict

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class MeetingSockets:
    """In-process WebSocket fanout; MongoDB remains the durable source of truth."""

    def __init__(self) -> None:
        self._clients: dict[str, set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def connect(self, meeting_id: str, socket: WebSocket) -> None:
        await socket.accept()
        async with self._lock:
            self._clients[meeting_id].add(socket)

    async def disconnect(self, meeting_id: str, socket: WebSocket) -> None:
        async with self._lock:
            clients = self._clients.get(meeting_id)
            if clients:
                clients.discard(socket)
                if not clients:
                    self._clients.pop(meeting_id, None)

    async def publish(self, meeting_id: str, event: dict) -> None:
        async with self._lock:
            clients = tuple(self._clients.get(meeting_id, ()))
        stale: list[WebSocket] = []
        for socket in clients:
            try:
                await socket.send_json(event)
            except Exception:
                logger.debug("Removing disconnected WebSocket", exc_info=True)
                stale.append(socket)
        for socket in stale:
            await self.disconnect(meeting_id, socket)


sockets = MeetingSockets()

