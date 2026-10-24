"""Summarize learning curves written by train.py.

    python progress.py [--checkpoint-dir checkpoints] [--window 10]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def summarize(metrics_file: Path, window: int) -> list[str]:
    records = [json.loads(line) for line in metrics_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    episodes = [r for r in records if r["event"] == "episode"]
    updates = [r for r in records if r["event"] == "update"]
    lines = [f"{metrics_file.parent.name}: {len(updates)} updates, {len(episodes)} episodes"]
    for start in range(0, len(episodes), window):
        chunk = episodes[start:start + window]
        wins = sum(r["result"] == "Won" for r in chunk)
        decided = sum(r["result"] in ("Won", "Lost") for r in chunk)
        mean = sum(r["episode_return"] for r in chunk) / len(chunk)
        rate = f"{wins}/{decided} won" if decided else "no decided matches"
        lines.append(f"  episodes {start + 1}-{start + len(chunk)}: mean return {mean:8.2f}, {rate}")
    for start in range(0, len(updates), window * 5):
        chunk = updates[start:start + window * 5]
        mean = sum(r["mean_return"] for r in chunk) / len(chunk)
        kl = sum(r.get("kl", 0) for r in chunk) / len(chunk)
        lines.append(f"  updates {start + 1}-{start + len(chunk)}: mean turn return {mean:7.3f}, kl {kl:.4f}")
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", type=Path, default=HERE / "checkpoints")
    p.add_argument("--window", type=int, default=10, help="Episodes per row")
    args = p.parse_args()
    files = sorted(args.checkpoint_dir.glob("*/metrics.jsonl"))
    if not files:
        print(f"No learning curves in {args.checkpoint_dir} yet.")
    for metrics_file in files:
        print("\n".join(summarize(metrics_file, max(1, args.window))))


if __name__ == "__main__":
    main()
