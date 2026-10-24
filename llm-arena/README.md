# OpenRA LLM Arena

Train a local Qwen commander against a built-in Red Alert AI in Singleplayer Skirmish. The Windows launcher
also supports fixed Ollama Cloud commanders (GLM and DeepSeek). Each commander controls its player through
MCP tools backed by an OpenRA bot bridge.

The `openra-lab` layout keeps this directory beside an `openra/` Git submodule. While the core PR is open,
it points to the contributor fork commit containing the LLM bridge and Qwen bot registration. After merge,
it can point to `time-to-think/openra`. The launcher also supports a combined development checkout.

```
 OpenRA (host)                                   llm-arena (Python)
 ┌──────────────────────────────┐   TCP JSON    ┌──────────────┐ stdio MCP ┌──────────────┐  HTTPS  ┌─────────────┐
 │ GLM bot ── LlmBridgeBotModule│◄────:7701────►│ mcp_server.py│◄─────────►│ Commander GLM│◄───────►│ Ollama Cloud│
 │ DeepSeek ─ LlmBridgeBotModule│◄────:7702────►│ mcp_server.py│◄─────────►│ Commander DS │◄───────►│  (models)   │
 └──────────────────────────────┘               └──────────────┘           └──────────────┘         └─────────────┘
```

- `LlmBridgeBotModule` (C#, `OpenRA.Mods.Common/Traits/BotModules/Llm/`) is a regular bot module. It listens on
  `127.0.0.1:<Port>`, answers state queries and turns commands into ordinary player orders on the game thread.
  Finished structures are placed automatically, so models only decide *what* to build and roughly *where*.
- `mcp_server.py` exposes one player as MCP tools: `get_state`, `get_game_info`, `build`, `cancel_build`, `move`,
  `attack_move`, `attack`, `stop`, `deploy`, `sell`.
- `arena.py` starts one MCP server per player, connects to it as an MCP client, and runs an Ollama tool-calling
  loop per model (`agent.py`). Every turn the model gets a fresh state snapshot plus its own plan from the
  previous turn.
- The system prompt (`prompts.py`) explains Red Alert basics (economy, power, production, combat) and embeds a
  briefing fetched once per game with `get_game_info`: your spawn, possible enemy spawns, ore fields, a coarse
  terrain grid, and the faction's full tech tree with costs, requirements and unit roles. The briefing only
  contains what a human sees in the lobby map preview and the build palette, never hidden enemy information.

## Local reinforcement learning

`train.py` shares a local base model between one LoRA adapter per player. It uses the player's own score
counters as rewards and saves adapter weights and optimizer state after each update. Finished matches
record the outcome before the commanders wait for a new game. Cloud commanders remain fixed models.
Training starts from the base model when no local checkpoint exists; learned weights are not published.

```powershell
.\llm-arena\setup.cmd -WithTraining -SkipApiKey
.\llm-arena\.venv\Scripts\python.exe llm-arena/train.py --agents alpha:7703
.\llm-arena\.venv\Scripts\python.exe llm-arena/progress.py
```

Start a Red Alert skirmish with **Qwen (LLM)** and a built-in AI. The command above trains the `alpha`
adapter on port 7703. Without `--agents`, the terminal trainer instead uses two learner adapters named
`alpha` and `beta` on the GLM and DeepSeek bot slots (ports 7701 and 7702). Start new matches to continue learning.
The default base model is `Qwen/Qwen3-1.7B`; its first load downloads the model. A CUDA GPU is recommended;
CPU training is slow. Install the appropriate PyTorch build for your GPU before running training.
`--check-policy` loads the model and samples one action without connecting to a game.
Checkpoints and learning metrics are stored in the ignored `llm-arena/checkpoints/` directory by default.
Use `--checkpoint-dir` to choose another location. `--cloud-opponent MODEL` replaces the last learner with
a fixed Ollama Cloud opponent and requires the normal Cloud credentials.

## Setup

For a new Windows checkout of `openra-lab`, install the .NET 10 SDK, Python 3.11 or later, and Git. Then run:

```powershell
git submodule update --init --recursive
dotnet build .\openra\OpenRA.slnx -c Debug
.\llm-arena\setup.cmd -WithTraining -SkipApiKey
powershell -NoProfile -ExecutionPolicy Bypass -File .\llm-arena\launcher\build.ps1
```

Open `OpenRA-LLM-Arena.exe` from the `openra-lab` root. On first launch, Red Alert may show its content
installer; install the free game asset pack there before starting a Skirmish. The launcher uses the system
.NET 10 runtime or an existing private runtime in `llm-arena/.local/dotnet`. The base Qwen model downloads
on first use. Game assets, Python packages, model downloads and learned adapters stay on the local machine.

For this combined development checkout, build with `dotnet build .\OpenRA.slnx -c Debug` instead of the
`openra/` command above. The remaining commands are the same.

### Optional Cloud setup

Run from the repository root:

```powershell
.\llm-arena\setup.cmd
```

The script creates or reuses the Python environment, installs dependencies, and checks both MCP servers.
Create a key at https://ollama.com/settings/keys and paste it into the hidden prompt in your terminal.
The key is encrypted with Windows DPAPI in the ignored `.local/ollama-key.xml` file. Only the same Windows
account on the same computer can decrypt it. `run.ps1` loads it for the Python process and restores the
calling shell's previous environment afterward. Do not paste your key into chat or source files.

The `.cmd` entry points use a child PowerShell process with an execution policy override, so they also
work when this machine blocks `.ps1` files. They do not change the machine or user execution policy.

No local Ollama installation, GPU, or model download is required for direct Cloud access. These agents use
Ollama Cloud independently of the model powering your Codex conversation. See the
[Ollama Cloud documentation](https://docs.ollama.com/cloud).

```powershell
# Prepare dependencies without entering a key yet:
.\llm-arena\setup.cmd -SkipApiKey
# Replace a saved key:
.\llm-arena\setup.cmd -ReplaceApiKey
# Check authentication and model tool calling without starting a game:
.\llm-arena\run.cmd -CheckCloud
# Check MCP without a key or network requests:
.\llm-arena\run.cmd -CheckMcp
# Show available model names:
.\llm-arena\run.cmd -ListModels
```

Cloud checks send two short inference requests per distinct selected model, consuming Ollama usage.
They verify authentication, a tool call, and the follow-up tool result. The same checks run before each
match, so bad keys and unavailable models fail before waiting for the game. A model list alone does not
verify authenticated inference. When using an existing `OLLAMA_API_KEY` environment variable, the launcher
prefers the process value, then the encrypted file, then the Windows user environment variable.

For other operating systems, create a virtual environment with Python 3.11 or later, install
`requirements.txt`, set `OLLAMA_API_KEY` in your shell, and run `python arena.py` from this directory.

## Running a match

### Desktop launcher on Windows

Double-click **`OpenRA-LLM-Arena.exe`** in the repository root. It opens the Red Alert game and starts
the selected agent. The default mode is **Qwen RL vs OpenRA AI** and needs no Ollama API key.
For the Cloud modes, configure `OLLAMA_API_KEY` first. No terminal commands are needed.
Keep the EXE beside `llm-arena/` and either `openra/` or `bin/`; use a shortcut if you want to launch it elsewhere.
This is a launcher for the existing checkout, not a standalone copy of the entire game and Python runtime.

The launcher shows live output, connection errors, and whether the agents are waiting or playing. For local
training, create a Skirmish with **Qwen (LLM)** and one built-in AI (**Rush**, **Normal**, **Turtle** or **Naval**) on
opposing teams, choose Spectator, and start the game. The local Qwen learner connects on port 7703.
Rewards update its LoRA adapter every few turns. The launcher resumes a local legacy
checkpoint when present; otherwise it starts at `llm-arena/checkpoints`. `ARENA_CHECKPOINT_DIR` overrides
the checkpoint location. After a match, start another Skirmish to continue training with the same adapter.
The built-in AI does not learn. Cloud modes use fixed Ollama models and require an API key.

For GLM vs DeepSeek, create a Skirmish with **GLM (LLM)** and **DeepSeek (LLM)** on opposing teams.
The agents automatically wait for subsequent matches, including an in-game restart. A new match resets
the plans and reloads the map and tech tree briefing. Temporary bridge interruptions or a paused game
do not terminate the commanders. Responses from the previous match are discarded, and commands are
bound to the last observed match so they cannot be applied to a replacement game.
**Restart agents** stops the current agent session before starting a new one; **Start / Retry** is shown
when agents are stopped. **Open logs** opens the current run's logs.
**Stop agents** and closing the launcher stop its agents and MCP subprocesses; the game stays open.
Closing the game also stops the agents. Opening the launcher again reuses an existing game window from
this checkout. It does not stop agents or games launched separately.

Launcher logs and control files are kept under the ignored `.local/launcher/` directory. Keys are read at
runtime and never embedded in the EXE. Local game data and model checkpoints are excluded from Git.
Building the launcher itself uses the Windows .NET Framework compiler:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\llm-arena\launcher\build.ps1 -Verify
```

Close the launcher before rebuilding. `-Verify` also runs native Windows argument-parsing regression
tests, including checkout folders with spaces and a trailing backslash, and a process restart test that
checks shutdown ordering, game preservation and separate session logs.

### Terminal workflow

1. Build and start the game (`make.ps1 all`, then `launch-game.cmd`) with the Red Alert mod in the combined checkout.
2. Create a skirmish. Add the bots **GLM (LLM)** and **DeepSeek (LLM)** to two slots on opposing teams and move
   yourself to a spectator slot.
3. Start the game. The in-game chat shows `LLM bridge for ... listening on 127.0.0.1:7701/7702`.
4. Run the arena (it waits for the game if started earlier):

   ```powershell
   # From the repository root:
   .\llm-arena\run.cmd
   .\llm-arena\run.cmd -GlmModel glm-5.3 -DeepSeekModel deepseek-v4.1-flash -Interval 3
   ```

To play one Cloud LLM against a built-in OpenRA bot, put **GLM (LLM)** and a standard bot (**Rush AI**, **Normal AI**, **Turtle AI** or **Naval AI**) on
opposing teams and run only that commander:

```powershell
.\llm-arena\run.cmd -Players glm
```

Transcripts are written to `logs/<timestamp>/<agent>.jsonl`.

| Option | Default | Notes |
| --- | --- | --- |
| `--glm-model` / `ARENA_GLM_MODEL` | `glm-5.3` | Any model name from `--list-models` |
| `--deepseek-model` / `ARENA_DEEPSEEK_MODEL` | `deepseek-v4-pro:0813` | |
| `--host` / `OLLAMA_HOST_URL` | `https://ollama.com` | `http://localhost:11434` uses a signed-in local Ollama (`<model>:cloud` names) |
| `--interval` | `5` | Seconds between turns |
| `--max-tool-rounds` | `8` | LLM/tool round trips per turn |
| `--check-cloud` | off | Check MCP and authenticated tool inference, then exit |
| `--check-mcp` | off | Check both MCP servers without a key or game |
| `--watch-games` | off | Wait for subsequent matches; enabled by the desktop launcher |

API model names must match `--list-models`. Direct Cloud names such as `glm-5.3` differ from the
`:cloud` names used by the signed-in local Ollama app.

## Verification

Run the offline tests from the repository root. They cover authentication failures, quota and network
errors, missing models, tool inference checks, and a commander using the real MCP subprocess with a
simulated game bridge and simulated Ollama responses. Launcher tests also verify parent-exit detection
and cancellation of a real MCP session. Match restart tests cover briefing and plan resets, stale Cloud
responses, state timeouts and socket reconnection without duplicate orders. They do not require a key
or consume Cloud usage.

```powershell
.\llm-arena\.venv\Scripts\python.exe -m unittest discover -s llm-arena/tests -v
```

## Using the MCP server from other clients

Any MCP client can drive a player while a match runs, e.g. Claude Code (`.mcp.json`):

```json
{
  "mcpServers": {
    "openra-glm": {
      "command": "llm-arena/.venv/Scripts/python",
      "args": ["llm-arena/mcp_server.py", "--port", "7701"]
    }
  }
}
```

## Notes and limitations

- Bridge ports and bot types are defined in `mods/ra/rules/llm-ai.yaml`; currently only the Red Alert mod
  registers the LLM bots. Add more `ModularBot` / `LlmBridgeBotModule` pairs there for other models.
- The bridge only listens on loopback and only exists while a game with an LLM bot is running on the host.
- Requests are processed on bot ticks, so they time out while the game is paused.
- Expired bridge requests are discarded before execution, rather than applied later when the game resumes.
- `get_state` and `get_game_info` include an optional additive `game_id`. The MCP bridge sends it back with
  commands to reject orders from a previous match; older bridge clients may omit it.
- Like every OpenRA bot, LLM bots run only on the host and are disabled in replays. Their orders are recorded,
  so replays of LLM matches play back normally.
