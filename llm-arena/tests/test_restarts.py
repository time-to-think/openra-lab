"""Regression checks for match changes during Cloud inference and bridge outages."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import ollama

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import AgentConfig, Commander


class RestartTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, *, change_during_chat=False, recover=False):
        game, requests, chats, notes = "first", [], [], []
        changed, played_again = asyncio.Event(), asyncio.Event()
        outage = recover

        async def bridge(reader, writer):
            nonlocal game, outage
            try:
                while line := await reader.readline():
                    request = json.loads(line)
                    requests.append(request)
                    method = request["method"]
                    error = None
                    if method == "get_state":
                        if game == "first" and changed.is_set():
                            game = "second"
                        result = {"game_id": game, "tick": 1, "win_state": "Undefined"}
                        if outage and len(chats) >= 1:
                            outage = False
                            error = "The game did not process the request in time (is the game paused?)."
                            changed.set()
                    elif method == "get_game_info":
                        result = {"game_id": game, "map": {"title": game}, "faction": "soviet"}
                    else:
                        self.assertEqual(request["params"]["game_id"], game)
                        result = {"ok": True}
                        if game == "second":
                            played_again.set()
                    writer.write((json.dumps({"id": request["id"], "error": error, "result": result}) + "\n").encode())
                    await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        def respond(request):
            body = json.loads(request.content)
            chats.append(body)
            if 'Map "first"' in body["messages"][0]["content"] and not change_during_chat and not recover:
                changed.set()
                message = {"role": "assistant", "content": "Old attack plan"}
            else:
                message = {"role": "assistant", "content": "", "tool_calls": [{"function": {
                    "name": "move", "arguments": {"unit_ids": [1], "x": 10, "y": 20},
                }}]}
                if change_during_chat and len(chats) == 1:
                    changed.set()
            return httpx.Response(200, json={"message": message, "done": True})

        server = await asyncio.start_server(bridge, "127.0.0.1", 0)
        client = ollama.AsyncClient(host="https://ollama.com", transport=httpx.MockTransport(respond))
        self.addAsyncCleanup(client._client.aclose)
        async with server:
            with tempfile.TemporaryDirectory() as directory:
                commander = Commander(AgentConfig("Test", "fake-model", server.sockets[0].getsockname()[1],
                                                  interval=0.001, max_tool_rounds=1), client, Path(directory))
                original_begin = commander._begin_game

                async def begin(mcp, state):
                    await original_begin(mcp, state)
                    notes.append((commander.game_id, commander.notes, commander.turn))
                    if commander.game_id == "first":
                        commander.notes = "Old attack plan"

                with patch.object(commander, "_begin_game", begin):
                    task = asyncio.create_task(commander.run(keep_alive=True))
                    try:
                        await asyncio.wait_for(played_again.wait(), 20)
                        self.assertFalse(task.done(), "Agents must stay alive across a match restart")
                    finally:
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                records = [json.loads(line) for line in commander.log_path.read_text().splitlines()]
        self.assertEqual([entry[0] for entry in notes], ["first", "second"])
        self.assertTrue(all(entry[1:] == ("(none yet - this is the first turn)", 0) for entry in notes))
        self.assertEqual([entry["text"] for entry in records if entry["event"] == "briefing"][1].splitlines()[1],
                         'Map "second": x None..None, y None..None.')
        return requests, chats

    async def test_restart_resets_briefing_and_plan(self):
        requests, chats = await self.exercise()
        second = next(body for body in chats if 'Map "second"' in body["messages"][0]["content"])
        self.assertNotIn("Old attack plan", second["messages"][1]["content"])

    async def test_cloud_response_from_previous_game_is_discarded(self):
        requests, _ = await self.exercise(change_during_chat=True)
        moves = [entry for entry in requests if entry["method"] == "move"]
        self.assertTrue(moves)
        self.assertTrue(all(entry["params"]["game_id"] == "second" for entry in moves))

    async def test_state_timeout_recovers_instead_of_exiting(self):
        await self.exercise(recover=True)

    async def test_finished_game_waits_for_a_new_match(self):
        with tempfile.TemporaryDirectory() as directory:
            commander = Commander(AgentConfig("Test", "fake", 1), None, Path(directory))
            with patch.object(commander, "_get_state", AsyncMock(side_effect=[
                {"game_id": "old", "win_state": "Won"}, {"game_id": "new", "win_state": "Undefined"},
            ])), patch("agent.asyncio.sleep", AsyncMock()):
                self.assertEqual((await commander._wait_for_game(None, active_only=True))["game_id"], "new")


if __name__ == "__main__":
    unittest.main()
