"""Shaped per-turn rewards computed from consecutive game snapshots.

All signals come from the player's own `stats` counters (the observer statistics panel) and own state, so the
reward never leaks information hidden by fog of war. Values are scaled so that one turn is roughly in [-1, 1]
and the match outcome dominates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RewardConfig:
    win: float = 10.0
    loss: float = -10.0
    kills_per_1000: float = 1.0
    deaths_per_1000: float = -1.0
    earned_per_1000: float = 0.5
    growth_per_1000: float = 0.3
    low_power: float = -0.1
    idle_cash_threshold: int = 3000
    idle_cash: float = -0.05
    invalid_command: float = -0.1
    tool_error: float = -0.05
    unparsable: float = -0.5


def turn_reward(prev: dict[str, Any], cur: dict[str, Any], *, parsed: bool, invalid: int, errors: int,
                cfg: RewardConfig) -> tuple[float, dict[str, float]]:
    """Reward for the action taken in `prev`, observed once the game reached `cur`."""
    a, b = prev.get("stats") or {}, cur.get("stats") or {}
    parts = {
        "kills": _delta(a, b, "kills_cost") / 1000 * cfg.kills_per_1000,
        "deaths": _delta(a, b, "deaths_cost") / 1000 * cfg.deaths_per_1000,
        "earned": _delta(a, b, "earned") / 1000 * cfg.earned_per_1000,
        "growth": (_delta(a, b, "army_value") + _delta(a, b, "assets_value")) / 1000 * cfg.growth_per_1000,
        "power": cfg.low_power if _low_power(cur) else 0.0,
        "hoarding": cfg.idle_cash if (cur.get("cash") or 0) > cfg.idle_cash_threshold else 0.0,
        "format": (0.0 if parsed else cfg.unparsable) + invalid * cfg.invalid_command + errors * cfg.tool_error,
        "outcome": outcome_reward(cur, cfg),
    }
    return sum(parts.values()), {k: round(v, 4) for k, v in parts.items() if v}


def outcome_reward(state: dict[str, Any], cfg: RewardConfig) -> float:
    win_state = state.get("win_state")
    return cfg.win if win_state == "Won" else cfg.loss if win_state == "Lost" else 0.0


def _delta(a: dict[str, Any], b: dict[str, Any], key: str) -> float:
    # Counters reset only with a new match, which starts a new episode, so a negative delta is a real change
    # for values like army_value and never happens for cumulative counters.
    return float((b.get(key) or 0) - (a.get(key) or 0))


def _low_power(state: dict[str, Any]) -> bool:
    power = state.get("power") or {}
    return bool(power) and (power.get("drained") or 0) > (power.get("provided") or 0)
