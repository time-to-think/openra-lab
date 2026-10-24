"""Turn-specific OpenRA commands presented as a small discrete policy action space."""

from __future__ import annotations

import re
from typing import Any

Call = tuple[str, dict[str, Any]]
Candidate = tuple[str, Call | None]

PRIORITY = ("proc", "powr", "apwr", "tent", "barr", "weap", "afld", "dome", "stek",
            "harv", "e1", "e3", "2tnk", "1tnk", "3tnk")
NONCOMBAT = {"mcv", "harv", "ore", "c17", "tran"}


def candidates(state: dict[str, Any], enemy_spawns: list[tuple[int, int]] | None = None) -> list[Candidate]:
    """Offer commands with observed actor IDs, current build options and in-bounds targets."""
    result: list[Candidate] = []
    units = state.get("units") or []
    mcvs = [u["id"] for u in units if u.get("type") == "mcv" and isinstance(u.get("id"), int)]
    if mcvs:
        result.append((f"Deploy MCV {mcvs[0]}", ("deploy", {"unit_ids": [mcvs[0]]})))

    cash = state.get("cash") or 0
    queues = state.get("production") or []
    buildable = []
    for queue in queues:
        if queue.get("queued"):
            continue
        for item in queue.get("buildable") or []:
            name, cost = item.get("name"), item.get("cost")
            if isinstance(name, str) and isinstance(cost, int) and cost <= cash:
                buildable.append(name)
    counts: dict[str, int] = {}
    for building in state.get("buildings") or []:
        kind = building.get("type")
        counts[kind] = counts.get(kind, 0) + 1
    power = state.get("power") or {}

    def rank(name: str) -> tuple[int, str]:
        if name == "proc" and not counts.get("proc"):
            return -50, name
        if name in ("powr", "apwr") and power.get("provided", 0) - power.get("drained", 0) < 100:
            return -40, name
        if name in ("tent", "barr", "weap") and not counts.get(name):
            return -30, name
        if name in ("powr", "apwr") and power.get("provided", 0) - power.get("drained", 0) > 200:
            return 40, name
        return PRIORITY.index(name) if name in PRIORITY else 30, name

    seen: set[str] = set()
    for name in sorted(buildable, key=rank):
        if name in seen:
            continue
        seen.add(name)
        result.append((f"Build {name}", ("build", {"item": name})))
        if len(result) >= 10:
            break

    combat = [u for u in units if u.get("type") not in NONCOMBAT and isinstance(u.get("id"), int)]
    ids = [u["id"] for u in combat[:12]]
    if ids:
        for enemy in (state.get("visible_enemies") or [])[:2]:
            if isinstance(enemy.get("id"), int):
                result.append((f"Attack visible {enemy.get('type')} {enemy['id']} with units {ids}",
                               ("attack", {"unit_ids": ids, "target_id": enemy["id"]})))
        bounds = state.get("map") or {}
        for x, y in (enemy_spawns or [])[:2]:
            if all(k in bounds for k in ("left", "right", "top", "bottom")):
                if not (bounds["left"] <= x <= bounds["right"] and bounds["top"] <= y <= bounds["bottom"]):
                    continue
            result.append((f"Attack-move units {ids} toward {x},{y}",
                           ("attack_move", {"unit_ids": ids, "x": x, "y": y})))

    wait = ("Wait for the next observation", None)
    defense = [(label, call) for label, call in result if call and call[0] == "build"
               and call[1]["item"] in ("sbag", "brik")]
    result = [option for option in result if option not in defense]
    result.append(wait)
    result.extend(defense)
    return result


def choice(text: str, options: list[Candidate]) -> tuple[Call | None, bool]:
    """Accept a choice number on its own line; reject ambiguous or out-of-range output."""
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    match = re.fullmatch(r"(?:choice\s*[:#]?\s*)?(\d{1,2})\s*\.?", cleaned, flags=re.I)
    if match is None:
        match = re.fullmatch(r"(\d{1,2})\.\s+(.+)", cleaned)
        if match and 1 <= int(match[1]) <= len(options) and match[2].strip() != options[int(match[1]) - 1][0]:
            return None, False
    if not match or not 1 <= int(match[1]) <= len(options):
        return None, False
    return options[int(match[1]) - 1][1], True


def choice_prompt(options: list[Candidate]) -> str:
    return "Choose one action. Reply with its number only.\n" + "\n".join(
        f"{i}. {label}" for i, (label, _) in enumerate(options, 1))
