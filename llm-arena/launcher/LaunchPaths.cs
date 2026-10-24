using System;
using System.Diagnostics;
using System.IO;
using System.Text;

namespace OpenRA.ArenaLauncher
{
	sealed class LaunchPaths
	{
		public readonly string Root;
		public readonly string GameRoot;
		public readonly string GameExecutable;
		public readonly string Arena;
		public readonly string Runtime;
		public readonly string Support;
		public readonly string Logs;

		public LaunchPaths(string root)
		{
			Root = Path.GetFullPath(root);
			var submodule = Path.Combine(Root, "openra");
			GameRoot = File.Exists(Path.Combine(submodule, "mods", "ra", "mod.yaml")) ? submodule : Root;
			GameExecutable = Path.Combine(GameRoot, "bin", "OpenRA.exe");
			Arena = Path.Combine(Root, "llm-arena");
			Runtime = Path.Combine(Arena, ".local", "dotnet");
			Support = Path.Combine(Arena, ".local", "openra-support");
			Logs = Path.Combine(Arena, ".local", "launcher");
		}

		public string ApiKey
		{
			get
			{
				var key = Environment.GetEnvironmentVariable("OLLAMA_API_KEY", EnvironmentVariableTarget.User);
				if (string.IsNullOrWhiteSpace(key))
					key = Environment.GetEnvironmentVariable("OLLAMA_API_KEY");
				return (key ?? "").Trim();
			}
		}

		public void Validate()
		{
			Require(GameExecutable, "The OpenRA game build is missing. Build the openra submodule first.");
			Require(Path.Combine(Arena, ".venv", "Scripts", "python.exe"), "Python is missing. Open llm-arena/setup.cmd first.");
			Require(Path.Combine(Arena, "launcher_runner.py"), "Keep this EXE beside the llm-arena folder. Use a shortcut to move it.");
			if (Players != "local-rl" && ApiKey.Length == 0)
				throw new InvalidOperationException("OLLAMA_API_KEY is not configured. Open llm-arena/setup.cmd to enter your key.");
			Directory.CreateDirectory(Support);
			Directory.CreateDirectory(Logs);
		}

		public ProcessStartInfo GameInfo()
		{
			var info = BaseInfo(GameExecutable);
			info.Arguments = Quote("Engine.EngineDir=" + GameRoot) + " " + Quote("Engine.SupportDir=" + Support)
				+ " Game.Mod=ra Graphics.Mode=Windowed Graphics.WindowedSize=1280,800";
			if (File.Exists(Path.Combine(Runtime, "dotnet.exe")))
			{
				info.EnvironmentVariables["DOTNET_ROOT"] = Runtime;
				info.EnvironmentVariables["DOTNET_ROOT_X64"] = Runtime;
			}
			// CreateNoWindow suppresses the console; do not hide the SDL game window.
			return info;
		}

		// "local-rl" trains the Qwen adapter; cloud modes select one or both arena players.
		public string Players { get; set; }

		public string CheckpointDirectory
		{
			get
			{
				var configured = Environment.GetEnvironmentVariable("ARENA_CHECKPOINT_DIR");
				if (!string.IsNullOrWhiteSpace(configured))
					return Path.GetFullPath(configured);
				var previous = Path.Combine(Arena, ".local", "live-preview-checkpoints-v2");
				if (File.Exists(Path.Combine(previous, "alpha", "adapter.pt")))
					return previous;
				return Path.Combine(Arena, "checkpoints");
			}
		}

		public ProcessStartInfo AgentInfo(string stopFile, int parentId)
		{
			var info = BaseInfo(Path.Combine(Arena, ".venv", "Scripts", "python.exe"));
			info.Arguments = "-u " + Quote(Path.Combine(Arena, "launcher_runner.py"))
				+ " --stop-file " + Quote(stopFile) + " --parent-pid " + parentId;
			if (Players == "local-rl")
				info.Arguments += " --training --agents alpha:7703 --checkpoint-dir " + Quote(CheckpointDirectory)
					+ " --batch-size 4 --horizon 2 --max-new-tokens 32 --interval 3";
			else
			{
				info.Arguments += " --host https://ollama.com --watch-games";
				if (!string.IsNullOrEmpty(Players))
					info.Arguments += " --players " + Quote(Players);
				info.EnvironmentVariables["OLLAMA_API_KEY"] = ApiKey;
			}
			info.WorkingDirectory = Arena;
			info.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
			return info;
		}

		ProcessStartInfo BaseInfo(string executable)
		{
			return new ProcessStartInfo(executable)
			{
				WorkingDirectory = GameRoot,
				UseShellExecute = false,
				CreateNoWindow = true,
				RedirectStandardOutput = true,
				RedirectStandardError = true,
				StandardOutputEncoding = Encoding.UTF8,
				StandardErrorEncoding = Encoding.UTF8
			};
		}

		static void Require(string path, string message)
		{
			if (!File.Exists(path))
				throw new FileNotFoundException(message, path);
		}

		public static string Quote(string value)
		{
			// Windows argv parsing requires backslashes before quotes to be doubled.
			var result = new StringBuilder("\"");
			var slashes = 0;
			foreach (var character in value)
			{
				if (character == '\\') { slashes++; continue; }
				result.Append('\\', character == '"' ? slashes * 2 + 1 : slashes);
				result.Append(character);
				slashes = 0;
			}
			result.Append('\\', slashes * 2);
			return result.Append('"').ToString();
		}
	}
}
