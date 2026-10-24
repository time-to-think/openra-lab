"""Bridge reconnection must not duplicate orders or target a different match."""

import asyncio
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge import BridgeError, GameBridge


class BridgeRestartTests(unittest.IsolatedAsyncioTestCase):
    async def start_bridge(self, handler):
        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        self.addAsyncCleanup(server.wait_closed)
        self.addCleanup(server.close)
        bridge = GameBridge(server.sockets[0].getsockname()[1], timeout=1)
        self.addAsyncCleanup(bridge._close)
        return bridge

    async def test_interrupted_order_is_not_automatically_replayed(self):
        calls = []

        async def disconnect(reader, writer):
            calls.append(json.loads(await reader.readline()))
            writer.close()
            await writer.wait_closed()

        bridge = await self.start_bridge(disconnect)
        with self.assertRaisesRegex(BridgeError, "may already have executed"):
            await bridge.call("build", {"item": "powr"})
        self.assertEqual(len(calls), 1)

    async def test_reads_reconnect_after_socket_disconnect(self):
        calls = []

        async def respond(reader, writer):
            request = json.loads(await reader.readline())
            calls.append(request)
            if len(calls) == 2:
                writer.write((json.dumps({"id": request["id"], "result": {"game_id": "new"}}) + "\n").encode())
                await writer.drain()
            writer.close()
            await writer.wait_closed()

        bridge = await self.start_bridge(respond)
        self.assertEqual(await bridge.call("get_state"), {"game_id": "new"})
        self.assertEqual(len(calls), 2)

    async def test_commands_remain_bound_to_the_last_observed_game(self):
        seen = []

        async def respond(reader, writer):
            try:
                while line := await reader.readline():
                    request = json.loads(line)
                    seen.append(request)
                    response = {"id": request["id"], "result": {"game_id": "old"}}
                    if request["method"] != "get_state":
                        response = {"id": request["id"], "error": "Game changed"}
                    writer.write((json.dumps(response) + "\n").encode())
                    await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        bridge = await self.start_bridge(respond)
        await bridge.call("get_state")
        with self.assertRaisesRegex(BridgeError, "Game changed"):
            await bridge.call("move", {"unit_ids": [1], "x": 1, "y": 1, "game_id": "new"})
        self.assertEqual(seen[-1]["params"]["game_id"], "old")


if __name__ == "__main__":
    unittest.main()
