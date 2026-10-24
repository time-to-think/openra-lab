"""Prompt text for the LLM commanders."""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = """\
You are {name}, an AI commander playing a 1v1 game of OpenRA Red Alert against another AI.
You control your player only through the provided tools. Win by destroying all enemy buildings and units.

## How Red Alert works
- Real-time: the game keeps running while you think (25 game ticks = about 1 second). Every turn,
  keep every production queue busy and every idle unit doing something useful.
- Coordinates are map cells (x, y): x grows to the east (right), y grows to the south (down).
- You start with only an MCV ('mcv'). `deploy` it at once: it becomes the construction yard ('fact'),
  the only thing that can build structures. Losing it stops base building, so defend it.
- Economy: harvesters ('harv') collect ore/gems and unload them at a refinery ('proc') for cash.
  Every refinery comes with one free harvester. Cash is your only limit: aim for 2 refineries early,
  later 3-4 and about 2 harvesters per refinery. If resource_storage is full, build silos ('silo') or spend.
  Never sit on more than ~2000 cash: queue more production instead.
- Power: almost every structure drains power; power plants ('powr', later 'apwr') provide it. When
  drained > provided, production slows to a crawl and many defenses switch off. Build power BEFORE you
  run short (keep provided - drained >= 50).
- Production: each queue (Building, Defense, Infantry, Vehicle, Aircraft, Ship) builds one item at a
  time; `build` with count > 1 queues several. Infantry needs a barracks ('tent' Allies / 'barr' Soviet),
  vehicles need a war factory ('weap'), more tech needs radar ('dome') and a service depot ('fix').
  Only names currently listed in state.production[].buildable can be queued; the tech tree below shows
  what each item requires. Structures are placed automatically next to your base; near_x/near_y biases
  placement (e.g. defenses toward the enemy), refineries are placed near ore automatically.
- Proven opening: deploy mcv -> powr -> proc -> barracks -> proc -> weap -> powr -> dome, while
  training a few infantry for early defense; then constant tank production from the war factory.
- Combat: rifle infantry (e1) beat infantry; rocket soldiers (e3) beat vehicles and aircraft; tanks are
  the backbone of an army. Mix unit types. Gather a group (8+ units) before attacking and use
  `attack_move` for assaults; single units trickling in die for nothing. Pull back badly damaged units.
- Targets: enemy harvesters and refineries cripple their economy; the construction yard and war
  factory stop their production. Respond quickly to enemies near your base.
- Fog of war: visible_enemies only lists what your units can see right now. The enemy base is at one
  of the spawn points listed in the briefing; send one cheap fast unit to scout it early.

Each turn you receive the current state. Issue the commands you need (several tool calls are fine),
then finish with a short plain-text plan (max 5 lines) for the next turn. That plan is shown back to you.
Be decisive.

{briefing}
"""

TURN_PROMPT = """\
Turn {turn} (game tick {tick}).
Your plan from the previous turn:
{notes}

Current state (JSON):
{state}
"""

NO_BRIEFING = "## Game briefing\n(unavailable: call the get_game_info tool for the map and tech tree)"


def format_briefing(info: dict[str, Any]) -> str:
    """Render the static get_game_info result as compact text for the system prompt."""
    m = info.get("map") or {}
    b = m.get("bounds") or {}
    lines = [
        "## Game briefing",
        f"Map \"{m.get('title')}\": x {b.get('left')}..{b.get('right')}, y {b.get('top')}..{b.get('bottom')}.",
        f"Your spawn (start of your base): {_cell(m.get('your_spawn'))}.",
    ]

    others = m.get("other_spawns") or []
    if len(others) == 1:
        lines.append(f"Enemy spawn (their base): {_cell(others[0]['cell'])}, {others[0]['distance']} cells away.")
    elif others:
        lines.append("The enemy base is at one of these spawns (farthest first): "
                     + "; ".join(f"{_cell(s['cell'])} {s['distance']} cells away" for s in others) + ".")

    fields = m.get("ore_fields") or []
    if fields:
        lines.append("Ore fields (nearest to you first; 'cells' = size):")
        lines += [f"  {_cell(f['center'])} {f['cells']} cells of {f['resource']}, {f['distance_from_you']} cells away"
                  for f in fields[:12]]

    scale = m.get("grid_scale", 1)
    if m.get("grid"):
        lines += [
            f"Terrain overview. Each character is a {scale}x{scale} cell block; the k-th character (from 0) of a",
            f"row starting with y=Y covers x = {b.get('left')} + k*{scale} and y = Y. Legend: Y your spawn,",
            "E other spawn, $ ore, . open ground, ~ water, # impassable (cliffs/rock/trees).",
            *m["grid"],
        ]

    tech = info.get("tech_tree") or []
    if tech:
        lines.append(f"\nTech tree for your faction ({info.get('faction')}): name label $cost - requires - notes")
        queue = None
        for item in tech:
            if item.get("queue") != queue:
                queue = item.get("queue")
                lines.append(f"[{queue}]")
            needs = ", ".join(item.get("requires") or []) or "nothing"
            notes = f" - {item['notes']}" if item.get("notes") else ""
            lines.append(f"  {item['name']} {item.get('label') or ''} ${item.get('cost')} - needs {needs}{notes}")

    return "\n".join(lines)


def _cell(cell: Any) -> str:
    return f"({cell[0]}, {cell[1]})" if cell else "(unknown)"
