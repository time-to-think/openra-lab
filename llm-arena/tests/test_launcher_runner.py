"""Verify desktop-launcher shutdown releases real MCP child processes."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mcp import Client, StdioServerParameters

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import launcher_runner
from cloud import CloudError


@unittest.skipUnless(sys.platform == "win32", "The desktop launcher requires Windows")
class LauncherRunnerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        previous_args = sys.argv[:]
        self.addCleanup(setattr, sys, "argv", previous_args)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.stop_file = Path(directory.name) / "stop-agents"

    async def test_preexisting_stop_request_avoids_starting_agents(self):
        self.stop_file.touch()
        with patch.object(launcher_runner.arena, "main") as main:
            await launcher_runner.supervise(self.stop_file, os.getpid(), [])
            main.assert_not_called()

    async def test_stop_request_closes_real_mcp_session(self):
        started, closed = asyncio.Event(), asyncio.Event()
        server = StdioServerParameters(command=sys.executable, args=[
            str(launcher_runner.arena.HERE / "mcp_server.py"), "--port", "7701",
        ])

        async def run_agents():
            try:
                async with Client(server) as client:
                    self.assertEqual(len((await client.list_tools()).tools), 10)
                    started.set()
                    await asyncio.Event().wait()
            finally:
                closed.set()

        with patch.object(launcher_runner.arena, "main", run_agents):
            task = asyncio.create_task(launcher_runner.supervise(self.stop_file, os.getpid(), []))
            try:
                await asyncio.wait_for(started.wait(), 15)
                self.stop_file.touch()
                await asyncio.wait_for(task, 10)
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
        self.assertTrue(closed.is_set())

    async def test_parent_process_exit_triggers_shutdown(self):
        parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.5)"],
                                  creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            await asyncio.wait_for(launcher_runner.wait_for_stop(self.stop_file, parent.pid), 5)
            await asyncio.to_thread(parent.wait, timeout=5)
            self.assertEqual(parent.returncode, 0)
        finally:
            if parent.poll() is None:
                parent.terminate()
                parent.wait(timeout=5)

    async def test_completed_match_exits_without_waiting_for_parent(self):
        async def completed_match():
            return
        with patch.object(launcher_runner.arena, "main", completed_match):
            await asyncio.wait_for(launcher_runner.supervise(self.stop_file, os.getpid(), []), 5)

    async def test_cloud_failure_reaches_launcher(self):
        async def failed_match():
            raise CloudError("Authentication rejected")
        with patch.object(launcher_runner.arena, "main", failed_match):
            with self.assertRaisesRegex(CloudError, "Authentication rejected"):
                await launcher_runner.supervise(self.stop_file, os.getpid(), [])

    async def test_training_mode_runs_local_learner_with_its_arguments(self):
        import train
        observed = []

        async def run_learner():
            observed.extend(sys.argv[1:])

        with patch.object(train, "main", run_learner):
            await launcher_runner.supervise(self.stop_file, os.getpid(),
                                            ["--agents", "alpha:7701"], training=True)
        self.assertEqual(observed, ["--agents", "alpha:7701"])


if __name__ == "__main__":
    unittest.main()
