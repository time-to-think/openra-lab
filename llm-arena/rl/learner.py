"""A commander whose policy is a trainable local model that improves from game rewards.

Each turn is one RL step: observe the state, sample one legal action choice from the policy, execute it through MCP,
and score it when the next state arrives. The adapter is updated whenever a batch of scored turns is ready,
so the commander keeps improving across turns and matches; adapters are saved after every update.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Protocol

from mcp import Client

from agent import AgentConfig, Commander
from rl.buffer import ExperienceBuffer, Transition
from rl.candidates import candidates, choice, choice_prompt
from rl.observation import system_prompt, turn_prompt
from rl.reward import RewardConfig, outcome_reward, turn_reward

_STALE = ("Game changed", "not reachable", "in time", "shutting down")


class Policy(Protocol):
    def generate(self, adapter: str, system: str, prompt: str) -> tuple[str, list[int]]: ...

    def update(self, adapter: str, batch: list[Transition]) -> dict[str, float]: ...


class LearnerCommander(Commander):
    def __init__(self, config: AgentConfig, policy: Policy, log_dir: Path, metrics_dir: Path, *,
                 buffer: ExperienceBuffer | None = None, batch_size: int = 16,
                 rewards: RewardConfig | None = None) -> None:
        super().__init__(config, None, log_dir)
        self.policy = policy
        self.buffer = buffer or ExperienceBuffer()
        self.batch_size = batch_size
        self.rewards = rewards or RewardConfig()
        self.metrics_path = metrics_dir / config.name / "metrics.jsonl"
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.system = system_prompt(None)
        self.enemy_spawns: list[tuple[int, int]] = []
        self.last_state: dict[str, Any] | None = None
        self.feedback = (True, 0, 0)
        self.episode_return = 0.0

    async def _begin_game(self, mcp: Client, state: dict[str, Any]) -> None:
        if self.buffer.episode:  # The previous match was abandoned or restarted before it finished.
            self._close_episode("Abandoned", None)
        await super()._begin_game(mcp, state)
        self.notes = ""

    async def _load_briefing(self, mcp: Client) -> None:
        text, is_error = await self._call_tool(mcp, "get_game_info", {})
        try:
            info = None if is_error else json.loads(text)
        except ValueError:
            info = None
        self.system = system_prompt(info)
        self.enemy_spawns = [tuple(spawn["cell"]) for spawn in ((info or {}).get("map") or {}).get("other_spawns", [])
                             if isinstance(spawn.get("cell"), list) and len(spawn["cell"]) == 2
                             and all(isinstance(value, int) for value in spawn["cell"])]
        self._log("briefing", text=self.system)

    async def _finish_game(self, state: dict[str, Any]) -> None:
        await super()._finish_game(state)
        reward = None
        if self.buffer.pending is not None:
            reward = self._score(state)
        else:  # The last action was discarded; the outcome still belongs to this episode.
            outcome = outcome_reward(state, self.rewards)
            self.buffer.add_to_last(outcome)
            self.episode_return += outcome
        self._close_episode(state.get("win_state", "Unknown"), reward)
        await self._train_ready()

    async def _play_turn(self, mcp: Client, tools: list[dict[str, Any]], state: dict[str, Any]) -> None:
        self.turn += 1
        if self.buffer.pending is not None:
            self.buffer.reward_pending(self._score(state))
        await self._train_ready()

        options = candidates(state, self.enemy_spawns)
        prompt = turn_prompt(state, self.notes) + "\n\n" + choice_prompt(options)
        started = time.monotonic()
        response, tokens = await asyncio.to_thread(self.policy.generate, self.cfg.name, self.system, prompt)
        call, parsed = choice(response, options)
        self._log("policy", seconds=round(time.monotonic() - started, 1), response=response,
                  parsed=parsed, invalid=0 if parsed else 1, calls=[call] if call else [])

        try:
            fresh = await self._get_state(mcp)
        except RuntimeError:
            self._say("waiting for game: connection interrupted; discarding this action")
            return
        if fresh.get("game_id") != self.game_id or fresh.get("win_state") != "Undefined":
            self._say("game changed or finished; discarding this action")
            return

        errors = 0
        for name, args in ([call] if call else []):
            text, is_error = await self._call_tool(mcp, name, args)
            errors += is_error
            self._say(f"{name}({json.dumps(args)}) -> {'ERROR ' if is_error else ''}{text[:160]}")
            if is_error and any(reason in text for reason in _STALE):
                return
        if not parsed:
            self._say("unparsable response; penalized")
        self.buffer.add(Transition(self.system, prompt, response, list(tokens)))
        self.last_state, self.feedback = state, (parsed, 0, errors)

    def _score(self, state: dict[str, Any]) -> float:
        parsed, invalid, errors = self.feedback
        reward, parts = turn_reward(self.last_state or state, state, parsed=parsed, invalid=invalid,
                                    errors=errors, cfg=self.rewards)
        self.episode_return += reward
        self._log("reward", reward=round(reward, 4), parts=parts)
        return reward

    def _close_episode(self, result: str, final_reward: float | None) -> None:
        turns = len(self.buffer.episode)
        self.buffer.end_episode(final_reward)
        self._metric("episode", game_id=self.game_id, result=result, turns=turns,
                     episode_return=round(self.episode_return, 3))
        self._say(f"episode finished ({result}): return {self.episode_return:.2f} over {turns} turns")
        self.episode_return, self.last_state = 0.0, None

    async def _train_ready(self) -> None:
        while (batch := self.buffer.take(self.batch_size)) is not None:
            self._say(f"training on {len(batch)} turns...")
            stats = await asyncio.to_thread(self.policy.update, self.cfg.name, batch)
            mean_return = sum(t.ret for t in batch) / len(batch)
            self._metric("update", mean_return=round(mean_return, 4), **stats)
            self._say(f"update {stats.get('update')}: mean return {mean_return:.3f}, kl {stats.get('kl')}")

    def _metric(self, event: str, **data: Any) -> None:
        record = {"time": time.time(), "agent": self.cfg.name, "event": event, **data}
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._log(event, **data)
