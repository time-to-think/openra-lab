"""Offline checks for authentication, model validation, and the real MCP transport."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import ollama

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import AgentConfig, Commander
from cloud import CloudError, check_inference, check_models, list_models, make_client


class CloudTests(unittest.IsolatedAsyncioTestCase):
    def client(self, handler):
        client = ollama.AsyncClient(host="https://ollama.com", transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(client._client.aclose)
        return client

    def test_key_required_before_network_access(self):
        with patch.dict(os.environ, {"OLLAMA_API_KEY": "  "}):
            with self.assertRaisesRegex(CloudError, "setup.ps1"):
                make_client("https://ollama.com/")

    async def test_cloud_authorization_header(self):
        with patch.dict(os.environ, {"OLLAMA_API_KEY": "offline-test-key"}):
            client = make_client("https://ollama.com/")
        self.addAsyncCleanup(client._client.aclose)
        self.assertEqual(client._client.headers["Authorization"], "Bearer offline-test-key")

    async def test_missing_model_detected_before_chat(self):
        client = self.client(lambda request: httpx.Response(200, json={"models": [{"model": "glm-5.3"}]}))
        with self.assertRaisesRegex(CloudError, "Models unavailable: missing-model"):
            await check_models(client, ["glm-5.3", "missing-model"])

    async def test_authentication_and_quota_errors(self):
        for status, message in [(401, "authentication"), (403, "authentication"), (429, "usage limit")]:
            with self.subTest(status=status):
                client = self.client(lambda request: httpx.Response(status, json={"error": "test failure"}))
                with self.assertRaisesRegex(CloudError, message):
                    await list_models(client)

    async def test_network_error_is_actionable(self):
        def offline(request):
            raise httpx.ConnectError("offline", request=request)
        with self.assertRaisesRegex(CloudError, "Cannot reach Ollama"):
            await list_models(self.client(offline))

    async def test_inference_rejects_model_without_tool_calls(self):
        client = self.client(lambda request: httpx.Response(200, json={
            "message": {"role": "assistant", "content": "Hello"}, "done": True,
        }))
        with self.assertRaisesRegex(CloudError, "did not call the test tool"):
            await check_inference(client, ["glm-5.3"])

    async def test_inference_checks_tool_result_round_trip(self):
        requests = []

        def respond(request):
            body = json.loads(request.content)
            requests.append(body)
            message = {"role": "assistant", "content": ""}
            if len(requests) == 1:
                message["tool_calls"] = [{"function": {"name": "cloud_probe", "arguments": {}}}]
            else:
                message["content"] = "Ready"
            return httpx.Response(200, json={"message": message, "done": True})

        await check_inference(self.client(respond), ["glm-5.3", "glm-5.3"])
        self.assertEqual(len(requests), 2)
        self.assertIn({"role": "tool", "tool_name": "cloud_probe", "content": "ready"}, requests[1]["messages"])

    async def test_commander_calls_real_mcp_and_bridge(self):
        bridge_calls, chat_calls = [], []
        moved = False

        async def bridge(reader, writer):
            nonlocal moved
            try:
                while line := await reader.readline():
                    request = json.loads(line)
                    bridge_calls.append(request)
                    if request["method"] == "get_state":
                        result = {"tick": 1, "win_state": "Won" if moved else "Undefined"}
                    elif request["method"] == "get_game_info":
                        result = {"faction": "soviet", "map": {"title": "Bridge Test Map"}}
                    else:
                        moved = request["method"] == "move"
                        result = {"ok": True}
                    writer.write((json.dumps({"id": request["id"], "result": result}) + "\n").encode())
                    await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        def llm(request):
            body = json.loads(request.content)
            chat_calls.append(body)
            message = {"role": "assistant", "content": "Move toward the enemy."}
            if len(chat_calls) == 1:
                message["tool_calls"] = [{"function": {
                    "name": "move", "arguments": {"unit_ids": [1], "x": 10, "y": 20},
                }}]
            return httpx.Response(200, json={"message": message, "done": True})

        server = await asyncio.start_server(bridge, "127.0.0.1", 0)
        async with server:
            port = server.sockets[0].getsockname()[1]
            with tempfile.TemporaryDirectory() as logs:
                commander = Commander(AgentConfig("Test", "glm-5.3", port, interval=0.001),
                                      self.client(llm), Path(logs))
                self.assertEqual(await asyncio.wait_for(commander.run(), 30), "Won")
                records = [json.loads(line) for line in commander.log_path.read_text().splitlines()]
                self.assertEqual(records[-1]["event"], "finished")
        move = next(call for call in bridge_calls if call["method"] == "move")
        self.assertEqual(move["params"], {"unit_ids": [1], "x": 10, "y": 20, "queued": False})
        self.assertEqual(len(chat_calls[0]["tools"]), 10)
        self.assertIn('Map "Bridge Test Map"', chat_calls[0]["messages"][0]["content"])
        self.assertTrue(any(message["role"] == "tool" for message in chat_calls[1]["messages"]))


if __name__ == "__main__":
    unittest.main()
