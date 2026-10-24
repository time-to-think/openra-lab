"""One LLM commander: an Ollama chat model driving an OpenRA player through MCP tools."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import ollama
from mcp import Client, StdioServerParameters

from prompts import NO_BRIEFING, SYSTEM_PROMPT, TURN_PROMPT, format_briefing

HERE = Path(__file__).resolve().parent


@dataclass
class AgentConfig:
    name: str
    model: str
    port: int
    interval: float = 5.0
    max_tool_rounds: int = 8
    temperature: float = 0.4


class Commander:
    def __init__(self, config: AgentConfig, llm: ollama.AsyncClient, log_dir: Path) -> None:
        self.cfg = config
        self.llm = llm
        self.log_path = log_dir / f"{config.name.lower()}.jsonl"
        self.notes = "(none yet - this is the first turn)"
        self.briefing = NO_BRIEFING
        self.turn = 0
        self.game_id: str | None = None

    async def run(self, keep_alive: bool = False) -> str:
        """Play until the game reports a result; returns the final win state."""
        server = StdioServerParameters(
            command=sys.executable, args=[str(HERE / "mcp_server.py"), "--port", str(self.cfg.port)], cwd=str(HERE)
        )
        async with Client(server) as mcp:
            tools = [_to_ollama_tool(t) for t in (await mcp.list_tools()).tools]
            self._log("start", model=self.cfg.model, tools=[t["function"]["name"] for t in tools])
            state = await self._wait_for_game(mcp)
            await self._begin_game(mcp, state)
            while True:
                if state.get("game_id") != self.game_id:
                    await self._begin_game(mcp, state)
                if state.get("win_state") != "Undefined":
                    await self._finish_game(state)
                    if not keep_alive:
                        return state.get("win_state", "Unknown")
                    self._say("match finished; waiting for game restart")
                    state = await self._wait_for_game(mcp, active_only=True)
                    continue
                await self._play_turn(mcp, tools, state)
                await asyncio.sleep(self.cfg.interval)
                state = await self._wait_for_game(mcp)

    async def _wait_for_game(self, mcp: Client, active_only: bool = False) -> dict[str, Any]:
        while True:
            try:
                state = await self._get_state(mcp)
                if not active_only or state.get("win_state") == "Undefined":
                    return state
            except RuntimeError as e:
                self._say(f"waiting for game on port {self.cfg.port}: {e}")
            await asyncio.sleep(3)

    async def _begin_game(self, mcp: Client, state: dict[str, Any]) -> None:
        self.game_id = state.get("game_id")
        self.turn = 0
        self.notes = "(none yet - this is the first turn)"
        self.briefing = NO_BRIEFING
        self._log("game", game_id=self.game_id)
        self._say("new game detected; resetting plans and loading the briefing")
        await self._load_briefing(mcp)

    async def _finish_game(self, state: dict[str, Any]) -> None:
        self._log("finished", win_state=state.get("win_state"))

    async def _load_briefing(self, mcp: Client) -> None:
        """Fetch the static map and tech tree once; the game is playable without it."""
        text, is_error = await self._call_tool(mcp, "get_game_info", {})
        try:
            if is_error:
                raise ValueError(text)
            self.briefing = format_briefing(json.loads(text))
        except (ValueError, KeyError, TypeError) as e:
            self._say(f"game briefing unavailable: {e}")
        self._log("briefing", text=self.briefing)

    async def _get_state(self, mcp: Client) -> dict[str, Any]:
        text, is_error = await self._call_tool(mcp, "get_state", {})
        if is_error:
            raise RuntimeError(text)
        return json.loads(text)

    async def _play_turn(self, mcp: Client, tools: list[dict[str, Any]], state: dict[str, Any]) -> None:
        self.turn += 1
        messages: list[Any] = [
            {"role": "system", "content": SYSTEM_PROMPT.format(name=self.cfg.name, briefing=self.briefing)},
            {"role": "user", "content": TURN_PROMPT.format(
                turn=self.turn, tick=state.get("tick"), notes=self.notes,
                state=json.dumps(state, separators=(",", ":")))},
        ]

        for _ in range(self.cfg.max_tool_rounds):
            started = time.monotonic()
            try:
                response = await self.llm.chat(
                    model=self.cfg.model, messages=messages, tools=tools,
                    options={"temperature": self.cfg.temperature},
                )
            except (ollama.ResponseError, httpx.RequestError, ConnectionError, TimeoutError) as e:
                self._say(f"LLM request failed: {e}")
                self._log("llm_error", error=str(e))
                await asyncio.sleep(10)
                return

            message = response.message
            messages.append(message)
            self._log("llm", seconds=round(time.monotonic() - started, 1), content=message.content,
                      tool_calls=[{"name": c.function.name, "args": c.function.arguments} for c in message.tool_calls or []])

            # A slow Cloud response can arrive after the player has restarted the match.
            try:
                fresh = await self._get_state(mcp)
            except RuntimeError:
                self._say("waiting for game: connection interrupted; discarding this response")
                return
            if fresh.get("game_id") != self.game_id or fresh.get("win_state") != "Undefined":
                self._say("game changed or finished; discarding this response")
                return

            if not message.tool_calls:
                self.notes = (message.content or "").strip()[-1500:] or self.notes
                self._say(f"turn {self.turn} plan: {self.notes.splitlines()[0] if self.notes else ''}")
                return

            for call in message.tool_calls:
                name, args = call.function.name, dict(call.function.arguments or {})
                text, is_error = await self._call_tool(mcp, name, args)
                self._say(f"{name}({json.dumps(args)}) -> {'ERROR ' if is_error else ''}{text[:160]}")
                messages.append({"role": "tool", "tool_name": name, "content": text})
                if name == "get_state" and not is_error:
                    if json.loads(text).get("game_id") != self.game_id:
                        return
                if is_error and any(reason in text for reason in ("Game changed", "not reachable", "in time", "shutting down")):
                    return

    async def _call_tool(self, mcp: Client, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        try:
            result = await mcp.call_tool(name, args)
        except Exception as e:  # Tool failures are reported back to the model instead of crashing the match.
            return f"{type(e).__name__}: {e}", True
        text = "\n".join(c.text for c in result.content if getattr(c, "text", None))
        return text, bool(result.is_error)

    def _say(self, text: str) -> None:
        print(f"[{self.cfg.name}] {text}", flush=True)

    def _log(self, event: str, **data: Any) -> None:
        record = {"time": time.time(), "agent": self.cfg.name, "turn": self.turn,
                  "game_id": self.game_id, "event": event, **data}
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _to_ollama_tool(tool: Any) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": tool.name, "description": tool.description or "", "parameters": tool.input_schema},
    }
