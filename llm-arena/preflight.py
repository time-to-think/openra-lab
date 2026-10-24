"""Check the arena's MCP subprocess without needing a running game."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters

from cloud import CloudError

HERE = Path(__file__).resolve().parent
REQUIRED_TOOLS = {"get_state", "get_game_info", "build", "cancel_build", "move", "attack_move", "attack", "stop", "deploy", "sell"}


async def check_mcp(port: int) -> None:
    server = StdioServerParameters(
        command=sys.executable, args=[str(HERE / "mcp_server.py"), "--port", str(port)], cwd=str(HERE)
    )
    try:
        async with asyncio.timeout(30):
            async with Client(server) as client:
                names = {tool.name for tool in (await client.list_tools()).tools}
                missing = REQUIRED_TOOLS - names
                if missing:
                    raise CloudError(f"MCP tools missing: {', '.join(sorted(missing))}")
    except CloudError:
        raise
    except Exception as error:
        raise CloudError("MCP startup failed. Run setup.ps1 -SkipApiKey to check dependencies.") from error
    print(f"[OK] MCP subprocess on player port {port}: all {len(REQUIRED_TOOLS)} tools registered", flush=True)
