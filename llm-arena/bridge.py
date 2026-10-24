"""Async client for the in-game LlmBridgeBotModule (newline-delimited JSON over loopback TCP)."""

from __future__ import annotations

import asyncio
import itertools
import json
from typing import Any

# Game state snapshots can be large; raise asyncio's default 64 KiB line limit.
_LINE_LIMIT = 16 * 1024 * 1024
_READ_METHODS = {"get_state", "get_game_info"}


class BridgeError(RuntimeError):
    """Raised when the game is unreachable or rejects a request."""


class GameBridge:
    def __init__(self, port: int, host: str = "127.0.0.1", timeout: float = 15.0) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self._ids = itertools.count(1)
        self._lock = asyncio.Lock()
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._game_id: str | None = None

    async def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Retry reads on disconnect; never replay an order whose outcome is unknown."""
        async with self._lock:
            payload = dict(params or {})
            if self._game_id and method != "get_state":
                payload["game_id"] = self._game_id
            for attempt in range(2):
                try:
                    result = await self._roundtrip(method, payload)
                    if method == "get_state" and isinstance(result, dict):
                        self._game_id = result.get("game_id")
                    return result
                except (ConnectionError, OSError, asyncio.IncompleteReadError, asyncio.TimeoutError) as e:
                    await self._close()
                    if attempt == 1 or method not in _READ_METHODS:
                        raise BridgeError(
                            f"OpenRA bridge on {self.host}:{self.port} is not reachable ({e!r}). "
                            "Read a fresh state before retrying. An interrupted order may already have executed."
                        ) from e
        raise AssertionError("unreachable")

    async def _roundtrip(self, method: str, params: dict[str, Any]) -> Any:
        if self._writer is None:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port, limit=_LINE_LIMIT), self.timeout
            )

        request_id = next(self._ids)
        payload = json.dumps({"id": request_id, "method": method, "params": params})
        self._writer.write(payload.encode("utf-8") + b"\n")
        await self._writer.drain()

        line = await asyncio.wait_for(self._reader.readline(), self.timeout)
        if not line:
            raise ConnectionResetError("bridge closed the connection")

        response = json.loads(line)
        if response.get("error"):
            raise BridgeError(response["error"])
        return response.get("result")

    async def _close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except OSError:
                pass
        self._reader = self._writer = None
