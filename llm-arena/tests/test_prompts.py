"""Offline checks for the static game briefing text."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prompts import SYSTEM_PROMPT, format_briefing

GAME_INFO = {
    "faction": "soviet",
    "map": {
        "title": "Test Map",
        "bounds": {"left": 2, "top": 2, "right": 97, "bottom": 97},
        "your_spawn": [10, 12],
        "other_spawns": [{"cell": [88, 85], "distance": 108}],
        "ore_fields": [{"center": [15, 20], "cells": 40, "resource": "Ore", "distance_from_you": 9}],
        "grid_scale": 2,
        "grid": ["y=  2 Y.$~#", "y=  4 ....E"],
    },
    "tech_tree": [
        {"name": "powr", "label": "Power Plant", "queue": "Building", "cost": 300, "requires": [], "notes": None},
        {"name": "barr", "label": "Barracks", "queue": "Building", "cost": 500, "requires": ["powr"],
         "notes": "Trains infantry."},
        {"name": "e1", "label": "Rifle Infantry", "queue": "Infantry", "cost": 100, "requires": ["barr|tent"],
         "notes": "Strong vs Infantry"},
    ],
}


class BriefingTests(unittest.TestCase):
    def test_single_enemy_spawn_is_named(self):
        text = format_briefing(GAME_INFO)
        self.assertIn("Your spawn (start of your base): (10, 12).", text)
        self.assertIn("Enemy spawn (their base): (88, 85), 108 cells away.", text)
        self.assertIn("(15, 20) 40 cells of Ore, 9 cells away", text)
        self.assertIn("x = 2 + k*2", text)
        self.assertIn("y=  4 ....E", text)

    def test_tech_tree_grouped_by_queue(self):
        text = format_briefing(GAME_INFO)
        self.assertIn("[Building]\n  powr Power Plant $300 - needs nothing\n  barr Barracks $500 - needs powr - Trains", text)
        self.assertIn("[Infantry]\n  e1 Rifle Infantry $100 - needs barr|tent - Strong vs Infantry", text)

    def test_multiple_spawns_and_missing_fields(self):
        info = {"map": {"other_spawns": [{"cell": [1, 2], "distance": 5}, {"cell": [3, 4], "distance": 3}]}}
        self.assertIn("one of these spawns (farthest first): (1, 2) 5 cells away; (3, 4) 3 cells away.",
                      format_briefing(info))
        self.assertTrue(format_briefing({}).startswith("## Game briefing"))

    def test_system_prompt_accepts_briefing(self):
        prompt = SYSTEM_PROMPT.format(name="GLM", briefing=format_briefing(GAME_INFO))
        self.assertIn("You are GLM", prompt)
        self.assertTrue(prompt.rstrip().endswith("Strong vs Infantry"))


if __name__ == "__main__":
    unittest.main()
