"""Compact text observations for the trainable policy.

The learner sees the same fog-respecting information as the Cloud commanders, but trimmed so that a full
training sequence (observation + response) stays short enough for LoRA training on a single consumer GPU.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

SYSTEM = """\
You command one player in a 1v1 OpenRA Red Alert skirmish. Win by destroying every enemy building and unit.
Coordinates are map cells (x, y); x grows east, y grows south. Deploy the 'mcv' first, keep power positive,
build refineries for income, keep production queues busy, and attack with grouped units.
You are rewarded for income, growing your army and base, destroying enemy value and winning; you are
penalized for losses, power shortages, hoarding cash and invalid commands.

Each turn has a numbered list of currently available commands. Pick one number only.

{briefing}"""

TURN = """\
Tick {tick}. Cash {cash} (storage {storage}). Power {power}.
Score: {score}
Buildings: {buildings}
Units (id type x,y hp% idle): {units}
Visible enemies (id type x,y hp%): {enemies}
Production:
{production}
Your note from last turn: {note}"""

MAX_UNITS = 40
MAX_ENEMIES = 20


def briefing(info: dict[str, Any] | None) -> str:
    """Static per-match facts: spawns and nearby ore. The buildable list in each turn replaces the tech tree."""
    m = (info or {}).get("map") or {}
    if not m:
        return "Map briefing unavailable."
    lines = [f"Map \"{m.get('title')}\". Your spawn {_cell(m.get('your_spawn'))}."]
    spawns = m.get("other_spawns") or []
    if spawns:
        lines.append("Possible enemy spawns: " + "; ".join(_cell(s.get("cell")) for s in spawns[:4]) + ".")
    fields = m.get("ore_fields") or []
    if fields:
        lines.append("Nearest ore: " + "; ".join(
            f"{_cell(f.get('center'))} ({f.get('cells')} cells)" for f in fields[:4]) + ".")
    return "\n".join(lines)


def system_prompt(info: dict[str, Any] | None) -> str:
    return SYSTEM.format(briefing=briefing(info))


def turn_prompt(state: dict[str, Any], note: str) -> str:
    power = state.get("power") or {}
    stats = state.get("stats") or {}
    units = state.get("units") or []
    enemies = state.get("visible_enemies") or []
    return TURN.format(
        tick=state.get("tick"),
        cash=state.get("cash"),
        storage=state.get("resource_storage"),
        power=f"{power.get('provided', 0)} provided / {power.get('drained', 0)} drained" if power else "n/a",
        score=", ".join(f"{k}={v}" for k, v in stats.items() if k != "income") or "n/a",
        buildings=_counts(state.get("buildings") or []),
        units=_list(units, MAX_UNITS, lambda u: f"{u['id']} {u['type']} {u['x']},{u['y']} {u.get('hp')}"
                    + (" idle" if u.get("idle") else "")),
        enemies=_list(enemies, MAX_ENEMIES, lambda e: f"{e['id']} {e['type']} {e['x']},{e['y']} {e.get('hp')}"),
        production="\n".join(_queue(q) for q in state.get("production") or []) or "  (no production queues)",
        note=note or "(none)",
    )


def _queue(queue: dict[str, Any]) -> str:
    queued = ", ".join(f"{i['item']} {i.get('progress', 0)}%" for i in queue.get("queued") or []) or "idle"
    buildable = " ".join(f"{b['name']}${b.get('cost')}" for b in queue.get("buildable") or []) or "none"
    return f"  {queue.get('queue')}: [{queued}] can build: {buildable}"


def _counts(actors: list[dict[str, Any]]) -> str:
    counts = Counter(a.get("type") for a in actors)
    return ", ".join(f"{name} x{n}" for name, n in sorted(counts.items())) or "none"


def _list(items: list[dict[str, Any]], limit: int, fmt) -> str:
    text = "; ".join(fmt(i) for i in items[:limit])
    if len(items) > limit:
        text += f"; ... {len(items) - limit} more"
    return text or "none"


def _cell(cell: Any) -> str:
    return f"({cell[0]}, {cell[1]})" if cell else "(unknown)"
