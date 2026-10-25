# openra-lab

Local Qwen reinforcement learning and optional Ollama Cloud commanders for OpenRA Red Alert skirmishes.
The training code and Windows launcher live in `llm-arena/`. The `openra/` Git submodule pins the matching
OpenRA engine, LLM bridge, and bot definitions to one exact commit. While the core pull request is open,
that commit is served from the contributor fork.

## Start on Windows

Install Git, the .NET 10 SDK, and Python 3.11 or later. Clone with submodules, then run from this directory:

```powershell
git submodule update --init --recursive
dotnet build .\openra\OpenRA.slnx -c Debug
.\llm-arena\setup.cmd -WithTraining -SkipApiKey
powershell -NoProfile -ExecutionPolicy Bypass -File .\llm-arena\launcher\build.ps1
.\OpenRA-LLM-Arena.exe
```

Install the Red Alert content pack if the game prompts for it. In Singleplayer > Skirmish, select
**Qwen (LLM)** and a built-in AI on opposing teams, select Spectator, and start the match. Start another
Skirmish after it ends to continue training. See [llm-arena/README.md](llm-arena/README.md) for the
training flow, Cloud modes, and test commands.

The first training run downloads `Qwen/Qwen3-1.7B`. Adapter checkpoints, optimizer state, learning logs,
game content, credentials, local runtimes, Python packages, and the built launcher are local artifacts
excluded from Git. A fresh clone begins with the base model and creates its own adapter.

## Repository boundary

Changes to the OpenRA bridge or bot definitions go through a pull request into `time-to-think/openra:develop`.
The submodule tracks the `engine/llm-bridge` branch of `time-to-think/openra`, which carries the pinned
LLM bridge commit until it is merged into `develop`. After the merge, repoint the submodule branch to `develop`.
The submodule commit must include the `Qwen (LLM)` bot on port 7703; the default launcher mode depends on it.

OpenRA and this lab code are distributed under the GNU GPL v3 or later; see [COPYING](COPYING).
