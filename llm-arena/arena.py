"""Run two Ollama Cloud models against each other in an OpenRA skirmish.

Start a Red Alert skirmish with the "GLM (LLM)" and "DeepSeek (LLM)" bots first, then:
    python arena.py
The API key is read from the OLLAMA_API_KEY environment variable.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

from agent import AgentConfig, Commander
from cloud import CLOUD_HOST, CloudError, check_inference, check_models, list_models, make_client
from preflight import check_mcp

HERE = Path(__file__).resolve().parent
PLAYERS = {"glm": "GLM", "deepseek": "DeepSeek"}


async def run_commanders(commanders: list[Commander], keep_alive: bool) -> list:
    """Release both MCP sessions if either commander fails, so Retry can start cleanly."""
    tasks = [asyncio.create_task(commander.run(keep_alive=keep_alive)) for commander in commanders]
    try:
        _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for task in pending:
            task.cancel()
        return await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glm-model", default=os.environ.get("ARENA_GLM_MODEL", "glm-5.3"))
    p.add_argument("--deepseek-model", default=os.environ.get("ARENA_DEEPSEEK_MODEL", "deepseek-v4-pro:0813"))
    p.add_argument("--players", default="glm,deepseek",
                   help="Comma-separated LLM players to run (glm, deepseek); use one to face a built-in OpenRA bot")
    p.add_argument("--glm-port", type=int, default=7701)
    p.add_argument("--deepseek-port", type=int, default=7702)
    p.add_argument("--host", default=os.environ.get("OLLAMA_HOST_URL", CLOUD_HOST),
                   help="Ollama API host. Use http://localhost:11434 to go through a signed-in local Ollama.")
    p.add_argument("--interval", type=float, default=5.0, help="Seconds to wait between turns")
    p.add_argument("--max-tool-rounds", type=int, default=8, help="LLM/tool round trips per turn")
    p.add_argument("--list-models", action="store_true", help="List models available on the host and exit")
    p.add_argument("--check-cloud", action="store_true", help="Check MCP, authentication and model tool calls, then exit")
    p.add_argument("--check-mcp", action="store_true", help="Check MCP locally without an API key or a running game")
    p.add_argument("--watch-games", action="store_true", help="Wait for subsequent matches until stopped")
    args = p.parse_args()
    args.players = [name.strip().lower() for name in args.players.split(",") if name.strip()]
    if not args.players or any(name not in PLAYERS for name in args.players) or len(set(args.players)) != len(args.players):
        p.error("--players must list glm and/or deepseek once each")
    if args.interval <= 0 or args.max_tool_rounds <= 0:
        p.error("--interval and --max-tool-rounds must be positive")
    if not 1 <= args.glm_port <= 65535 or not 1 <= args.deepseek_port <= 65535:
        p.error("Player ports must be between 1 and 65535")
    if len(args.players) > 1 and args.glm_port == args.deepseek_port:
        p.error("Each player needs a different bridge port")
    return args


async def main() -> None:
    args = parse_args()
    players = [(PLAYERS[key], getattr(args, f"{key}_model"), getattr(args, f"{key}_port")) for key in args.players]
    if args.check_mcp:
        for _, _, port in players:
            await check_mcp(port)
        return
    llm = make_client(args.host)

    if args.list_models:
        for model in await list_models(llm):
            print(model)
        return

    print(f"Checking Ollama at {args.host}...", flush=True)
    for _, _, port in players:
        await check_mcp(port)
    models = list(dict.fromkeys(model for _, model, _ in players))
    await check_models(llm, models)
    await check_inference(llm, models)
    if args.check_cloud:
        print("Cloud and MCP checks passed. Start the OpenRA skirmish, then run without --check-cloud.")
        return

    log_dir = HERE / "logs" / time.strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    commanders = [Commander(AgentConfig(name, model, port, args.interval, args.max_tool_rounds), llm, log_dir)
                  for name, model, port in players]

    print(" vs ".join(f"{name}={model} (port {port})" for name, model, port in players))
    print(f"Logs: {log_dir}")
    results = await run_commanders(commanders, args.watch_games)
    for commander, result in zip(commanders, results):
        print(f"{commander.cfg.name}: {result!r}")
    if any(isinstance(result, BaseException) for result in results):
        raise CloudError("An arena commander failed. See the result above and the match logs.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except CloudError as error:
        print(f"Setup error: {error}", file=sys.stderr)
        sys.exit(1)
