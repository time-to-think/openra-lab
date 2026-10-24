"""MCP server exposing one OpenRA LLM bot player as tools.

Run one instance per player, pointing at that bot's bridge port:
    python mcp_server.py --port 7701
Any MCP client (the arena runner, Claude Code, ...) can then drive the player over stdio.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from bridge import BridgeError, GameBridge

_args = argparse.ArgumentParser(description=__doc__)
_args.add_argument("--port", type=int, required=True, help="LlmBridgeBotModule port of the player to control")
_args.add_argument("--host", default="127.0.0.1")
ARGS = _args.parse_args()

bridge = GameBridge(ARGS.port, ARGS.host)
server = MCPServer(
    name=f"openra-player-{ARGS.port}",
    instructions=(
        "Controls one Red Alert player in a running OpenRA skirmish. Coordinates are map cells (x, y). "
        "Call get_state first; ids in it are used by the other tools. Finished structures are placed automatically."
    ),
)


async def _call(method: str, **params: Any) -> str:
    try:
        result = await bridge.call(method, {k: v for k, v in params.items() if v is not None})
    except BridgeError as e:
        raise ToolError(str(e)) from e
    return result if isinstance(result, str) else json.dumps(result, separators=(",", ":"))


@server.tool()
async def get_state() -> str:
    """Snapshot of your player: cash, power, buildings, units (with ids, cell positions, hp %),
    visible enemies, and per-queue production status with buildable item names and costs."""
    return await _call("get_state")


@server.tool()
async def get_game_info() -> str:
    """Static briefing for this game: map title and bounds, your spawn, possible enemy spawns, ore fields,
    a coarse terrain grid, and your faction's full tech tree (costs, requirements, unit roles)."""
    return await _call("get_game_info")


@server.tool()
async def build(item: str, count: int = 1, near_x: int | None = None, near_y: int | None = None) -> str:
    """Queue production of a buildable item (structure or unit) by its internal name, e.g. 'powr', 'e1'.
    Structures are placed automatically when finished; near_x/near_y optionally bias placement toward a cell."""
    return await _call("build", item=item, count=count, near_x=near_x, near_y=near_y)


@server.tool()
async def cancel_build(item: str) -> str:
    """Cancel one queued or in-progress item and refund its cost."""
    return await _call("cancel_build", item=item)


@server.tool()
async def move(unit_ids: list[int], x: int, y: int, queued: bool = False) -> str:
    """Move your units to cell (x, y) without stopping to fight."""
    return await _call("move", unit_ids=unit_ids, x=x, y=y, queued=queued)


@server.tool()
async def attack_move(unit_ids: list[int], x: int, y: int, queued: bool = False) -> str:
    """Move your units toward cell (x, y), engaging any enemies met on the way. Best for assaults."""
    return await _call("attack_move", unit_ids=unit_ids, x=x, y=y, queued=queued)


@server.tool()
async def attack(unit_ids: list[int], target_id: int) -> str:
    """Order your units to attack one currently visible enemy actor by id."""
    return await _call("attack", unit_ids=unit_ids, target_id=target_id)


@server.tool()
async def stop(unit_ids: list[int]) -> str:
    """Stop your units' current activity."""
    return await _call("stop", unit_ids=unit_ids)


@server.tool()
async def deploy(unit_ids: list[int]) -> str:
    """Deploy units that transform in place, e.g. an 'mcv' into a construction yard."""
    return await _call("deploy", unit_ids=unit_ids)


@server.tool()
async def sell(building_ids: list[int]) -> str:
    """Sell your own buildings for a partial refund."""
    return await _call("sell", unit_ids=building_ids)


if __name__ == "__main__":
    server.run("stdio")
