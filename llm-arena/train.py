"""Train local LLM commanders with reinforcement learning by playing OpenRA skirmishes.

Every player slot gets its own LoRA adapter on a shared local base model. The adapters are updated from
game rewards while matches run and are saved after every update, so training resumes across sessions.
For Qwen against a built-in OpenRA AI, add the "Qwen (LLM)" bot and run:
    python train.py --agents alpha:7703

The default two-agent command instead trains adapters on the GLM and DeepSeek bot slots.
Restart or start new matches whenever one ends; the learners keep training across matches.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from agent import AgentConfig, Commander
from arena import run_commanders
from cloud import CLOUD_HOST, CloudError, check_inference, check_models, make_client
from rl.buffer import ExperienceBuffer
from rl.candidates import candidates, choice, choice_prompt
from rl.learner import LearnerCommander
from rl.observation import system_prompt, turn_prompt
from rl.policy import PolicyConfig

HERE = Path(__file__).resolve().parent
SAMPLE_STATE = {
    "tick": 1, "cash": 5000, "resource_storage": "0/0", "power": {"provided": 0, "drained": 0},
    "stats": {"earned": 0, "kills_cost": 0, "deaths_cost": 0, "army_value": 0, "assets_value": 0},
    "buildings": [], "units": [{"id": 101, "type": "mcv", "x": 30, "y": 40, "hp": 100, "idle": True}],
    "visible_enemies": [], "production": [],
}


def parse_agents(text: str) -> list[tuple[str, int]]:
    agents = []
    for entry in text.split(","):
        name, _, port = entry.strip().partition(":")
        if not name.isidentifier() or not port.isdigit() or not 1 <= int(port) <= 65535:
            raise argparse.ArgumentTypeError(f"expected NAME:PORT entries, got {entry!r}")
        agents.append((name, int(port)))
    if len({name for name, _ in agents}) != len(agents) or len({port for _, port in agents}) != len(agents):
        raise argparse.ArgumentTypeError("agent names and ports must be unique")
    return agents


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agents", type=parse_agents, default=parse_agents("alpha:7701,beta:7702"),
                   help="Learners as NAME:PORT pairs; each name keeps its own adapter and metrics")
    p.add_argument("--cloud-opponent", metavar="MODEL",
                   help="Let a fixed Ollama Cloud model play the last port instead of a learner (benchmark)")
    p.add_argument("--host", default=os.environ.get("OLLAMA_HOST_URL", CLOUD_HOST))
    p.add_argument("--base-model", default=os.environ.get("ARENA_BASE_MODEL", PolicyConfig.base_model))
    p.add_argument("--checkpoint-dir", type=Path, default=HERE / "checkpoints")
    p.add_argument("--lora-rank", type=int, default=PolicyConfig.lora_rank)
    p.add_argument("--lr", type=float, default=PolicyConfig.learning_rate)
    p.add_argument("--kl-beta", type=float, default=PolicyConfig.kl_beta)
    p.add_argument("--temperature", type=float, default=PolicyConfig.temperature)
    p.add_argument("--max-new-tokens", type=int, default=PolicyConfig.max_new_tokens)
    p.add_argument("--batch-size", type=int, default=16, help="Scored turns per policy update")
    p.add_argument("--gamma", type=float, default=0.9, help="Discount for future turn rewards")
    p.add_argument("--horizon", type=int, default=8, help="Turns of future reward in each return")
    p.add_argument("--interval", type=float, default=3.0, help="Seconds between turns")
    p.add_argument("--check-policy", action="store_true", help="Load the model, sample one action and exit")
    args = p.parse_args()
    if args.batch_size < 2 or args.horizon < 2 or not 0 < args.gamma <= 1 or args.interval <= 0:
        p.error("--batch-size and --horizon must be at least 2, --gamma in (0, 1], --interval positive")
    if args.cloud_opponent and len(args.agents) < 2:
        p.error("--cloud-opponent needs at least two --agents entries")
    return args


def make_policy(args: argparse.Namespace, names: list[str]):
    try:
        from rl.policy import LoraPolicy
        return LoraPolicy(PolicyConfig(
            base_model=args.base_model, checkpoint_dir=args.checkpoint_dir, lora_rank=args.lora_rank,
            learning_rate=args.lr, kl_beta=args.kl_beta, temperature=args.temperature,
            max_new_tokens=args.max_new_tokens), names)
    except ImportError as error:
        raise CloudError(f"Training dependencies are missing ({error.name}). Run setup.ps1 -WithTraining.") from error


async def main() -> None:
    args = parse_args()
    learners = args.agents[:-1] if args.cloud_opponent else args.agents
    print(f"Loading {args.base_model} with adapters {', '.join(n for n, _ in learners)}...", flush=True)
    policy = make_policy(args, [name for name, _ in learners])
    if args.check_policy:
        name = learners[0][0]
        options = candidates(SAMPLE_STATE)
        response, tokens = policy.generate(name, system_prompt(None),
                                           turn_prompt(SAMPLE_STATE, "") + "\n\n" + choice_prompt(options))
        print(response)
        call, parsed = choice(response, options)
        print(json.dumps({"tokens": len(tokens), "parsed": parsed, "call": call}))
        return

    log_dir = HERE / "logs" / time.strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    commanders: list[Commander] = [
        LearnerCommander(AgentConfig(name, args.base_model, port, args.interval), policy, log_dir,
                         args.checkpoint_dir, buffer=ExperienceBuffer(args.gamma, args.horizon),
                         batch_size=args.batch_size)
        for name, port in learners
    ]
    if args.cloud_opponent:
        llm = make_client(args.host)
        await check_models(llm, [args.cloud_opponent])
        await check_inference(llm, [args.cloud_opponent])
        name, port = args.agents[-1]
        commanders.append(Commander(AgentConfig(name, args.cloud_opponent, port, args.interval), llm, log_dir))

    print(f"Logs: {log_dir}\nCheckpoints and learning curves: {args.checkpoint_dir}", flush=True)
    results = await run_commanders(commanders, keep_alive=True)
    if any(isinstance(result, BaseException) for result in results):
        raise CloudError(f"A commander failed: {results!r}. See the match logs.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except (CloudError, ValueError) as error:
        print(f"Setup error: {error}", file=sys.stderr)
        sys.exit(1)
