"""A failed commander must release its still-running peer before a retry."""

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena import run_commanders


class ArenaLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_commander_cancels_its_peer(self):
        closed = asyncio.Event()

        async def waiting(**kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()

        commanders = [SimpleNamespace(run=waiting), SimpleNamespace(run=AsyncMock(side_effect=RuntimeError("failed")))]
        results = await asyncio.wait_for(run_commanders(commanders, True), 2)
        self.assertTrue(closed.is_set())
        self.assertIsInstance(results[0], asyncio.CancelledError)
        self.assertIsInstance(results[1], RuntimeError)

    async def test_normal_results_wait_for_both_commanders(self):
        commanders = [SimpleNamespace(run=AsyncMock(return_value=result)) for result in ("Won", "Lost")]
        self.assertEqual(await run_commanders(commanders, False), ["Won", "Lost"])


if __name__ == "__main__":
    unittest.main()
