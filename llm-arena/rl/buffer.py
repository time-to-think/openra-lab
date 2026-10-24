"""On-policy experience buffer with truncated discounted returns.

Matches are long and only end when a player restarts them, so the learner trains during play. A turn becomes
trainable once `horizon` later rewards are known or its episode ended; its return is the discounted sum of
those rewards. Advantages are normalized across the batch (a GRPO-style baseline without a value network).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field


@dataclass
class Transition:
    system: str
    prompt: str
    response: str
    response_ids: list[int] = field(default_factory=list)
    reward: float | None = None
    ret: float | None = None
    advantage: float = 0.0


@dataclass
class ExperienceBuffer:
    gamma: float = 0.9
    horizon: int = 8
    episode: list[Transition] = field(default_factory=list)
    ready: list[Transition] = field(default_factory=list)
    _next_return: int = 0

    def add(self, transition: Transition) -> None:
        self.episode.append(transition)

    @property
    def pending(self) -> Transition | None:
        """The last action whose reward is still waiting for the next observation."""
        return self.episode[-1] if self.episode and self.episode[-1].reward is None else None

    def reward_pending(self, reward: float) -> None:
        if self.pending is not None:
            self.pending.reward = reward
            self._collect(final=False)

    def add_to_last(self, bonus: float) -> None:
        """Credit a late reward (e.g. the match outcome) to the last scored action of the episode."""
        if self.episode and self.episode[-1].reward is not None:
            self.episode[-1].reward += bonus

    def end_episode(self, final_reward: float | None = None) -> None:
        """Close the match; an unrewarded last action receives `final_reward` or is dropped."""
        if self.pending is not None:
            if final_reward is None:
                self.episode.pop()
            else:
                self.pending.reward = final_reward
        self._collect(final=True)
        self.episode, self._next_return = [], 0

    def take(self, batch_size: int) -> list[Transition] | None:
        if len(self.ready) < batch_size:
            return None
        batch, self.ready = self.ready[:batch_size], self.ready[batch_size:]
        returns = [t.ret for t in batch]
        mean = statistics.fmean(returns)
        std = statistics.pstdev(returns) or 1.0
        for t in batch:
            t.advantage = (t.ret - mean) / std
        return batch

    def _collect(self, final: bool) -> None:
        rewarded = sum(1 for t in self.episode if t.reward is not None)
        last = rewarded if final else rewarded - self.horizon + 1
        while self._next_return < last:
            window = self.episode[self._next_return:self._next_return + self.horizon]
            transition = self.episode[self._next_return]
            transition.ret = sum(self.gamma ** i * t.reward for i, t in enumerate(window) if t.reward is not None)
            self.ready.append(transition)
            self._next_return += 1
