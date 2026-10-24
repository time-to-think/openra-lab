"""Offline checks for the RL action parser, rewards, returns and the learner's game loop."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import AgentConfig
from rl.buffer import ExperienceBuffer, Transition
from rl.candidates import candidates, choice
from rl.learner import LearnerCommander
from rl.observation import turn_prompt
from rl.reward import RewardConfig, turn_reward


def state(tick=1, win="Undefined", **stats):
    return {"game_id": "g", "tick": tick, "win_state": win, "cash": 1000, "power": {"provided": 100, "drained": 50},
            "stats": stats, "units": [{"id": 1, "type": "mcv", "x": 3, "y": 4, "hp": 100, "idle": True}],
            "buildings": [], "visible_enemies": [], "production": []}


class ActionTests(unittest.TestCase):
    def test_candidates_use_observed_ids_and_available_builds(self):
        current = {**state(), "cash": 500, "production": [{"queue": "Building", "queued": [],
                   "buildable": [{"name": "powr", "cost": 300}, {"name": "proc", "cost": 1500}]}],
                   "visible_enemies": [{"id": 17, "type": "e1"}], "map": {"left": 0, "right": 20,
                   "top": 0, "bottom": 20}}
        options = candidates(current, [(10, 10), (30, 30)])
        self.assertEqual(choice("1", options), (("deploy", {"unit_ids": [1]}), True))
        self.assertEqual(choice("1. Deploy MCV 1", options), (("deploy", {"unit_ids": [1]}), True))
        self.assertIn(("Build powr", ("build", {"item": "powr"})), options)
        self.assertFalse(any("proc" in label or "30,30" in label for label, _ in options))
        self.assertEqual(choice("deploy 1", options), (None, False))


class RewardTests(unittest.TestCase):
    def test_reward_components(self):
        cfg = RewardConfig()
        before = state(kills_cost=0, deaths_cost=0, earned=0)
        after = state(kills_cost=2000, deaths_cost=500, earned=1000)
        reward, parts = turn_reward(before, after, parsed=True, invalid=1, errors=0, cfg=cfg)
        self.assertEqual(parts, {"kills": 2.0, "deaths": -0.5, "earned": 0.5, "format": -0.1})
        self.assertAlmostEqual(reward, 1.9)
        won, parts = turn_reward(before, state(win="Won"), parsed=False, invalid=0, errors=0, cfg=cfg)
        self.assertEqual(parts["outcome"], cfg.win)
        self.assertEqual(parts["format"], cfg.unparsable)


class BufferTests(unittest.TestCase):
    def test_truncated_returns_and_normalized_advantages(self):
        buffer = ExperienceBuffer(gamma=0.5, horizon=2)
        for reward in (1.0, 2.0, 4.0):
            buffer.add(Transition("s", "p", "r"))
            buffer.reward_pending(reward)
        self.assertEqual([t.ret for t in buffer.ready], [2.0, 4.0])
        buffer.add(Transition("s", "p", "r"))
        buffer.end_episode()  # The unscored last action is dropped; the final scored one gets a short window.
        self.assertEqual([t.ret for t in buffer.ready], [2.0, 4.0, 4.0])
        batch = buffer.take(3)
        self.assertAlmostEqual(sum(t.advantage for t in batch), 0.0)
        self.assertLess(batch[0].advantage, batch[1].advantage)
        self.assertIsNone(buffer.take(1))


class FakePolicy:
    def __init__(self):
        self.prompts, self.batches = [], []

    def generate(self, adapter, system, prompt):
        self.prompts.append(prompt)
        return "1", [1, 2, 3]

    def update(self, adapter, batch):
        self.batches.append(batch)
        return {"update": len(self.batches), "loss": 0.0, "kl": 0.0}


class LearnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_learner_scores_turns_trains_and_records_the_outcome(self):
        requests, ticks = [], iter(range(1, 1000))
        finished = asyncio.Event()

        class RecordedLearner(LearnerCommander):
            async def _finish_game(self, final_state):
                await super()._finish_game(final_state)
                finished.set()

        async def bridge(reader, writer):
            while line := await reader.readline():
                request = json.loads(line)
                requests.append(request["method"])
                if request["method"] == "get_state":
                    tick = next(ticks)
                    result = state(tick, "Won" if tick >= 40 else "Undefined", earned=tick * 100)
                elif request["method"] == "get_game_info":
                    result = {"map": {"title": "Arena", "your_spawn": [3, 4]}}
                else:
                    result = {"ok": True}
                writer.write((json.dumps({"id": request["id"], "result": result}) + "\n").encode())
                await writer.drain()
            writer.close()

        server = await asyncio.start_server(bridge, "127.0.0.1", 0)
        policy = FakePolicy()
        async with server:
            with tempfile.TemporaryDirectory() as directory:
                learner = RecordedLearner(AgentConfig("alpha", "base", server.sockets[0].getsockname()[1],
                                                       interval=0.001), policy, Path(directory), Path(directory),
                                           buffer=ExperienceBuffer(0.9, 2), batch_size=4)
                task = asyncio.create_task(learner.run(keep_alive=True))
                try:
                    await asyncio.wait_for(finished.wait(), 30)
                finally:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                metrics = [json.loads(line) for line in learner.metrics_path.read_text().splitlines()]

        self.assertIn("deploy", requests)
        self.assertIn("Choose one action. Reply with its number only.", policy.prompts[-1])
        self.assertTrue(policy.batches and all(len(b) == 4 for b in policy.batches))
        self.assertTrue(all(t.response_ids == [1, 2, 3] for b in policy.batches for t in b))
        episode = next(m for m in metrics if m["event"] == "episode")
        self.assertEqual(episode["result"], "Won")
        self.assertGreater(episode["episode_return"], RewardConfig().win)
        self.assertTrue(any(m["event"] == "update" for m in metrics))

    def test_turn_prompt_is_compact(self):
        units = [{"id": i, "type": "e1", "x": i, "y": i, "hp": 100, "idle": False} for i in range(100)]
        text = turn_prompt({**state(), "units": units}, "")
        self.assertIn("... 60 more", text)
        self.assertLess(len(text), 2500)


if __name__ == "__main__":
    unittest.main()
