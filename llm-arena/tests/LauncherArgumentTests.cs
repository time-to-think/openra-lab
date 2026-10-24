using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using OpenRA.ArenaLauncher;

static class LauncherArgumentTests
{
	static int Main(string[] args)
	{
		if (args.Length >= 2 && args[0] == "--echo")
		{
			File.WriteAllLines(args[1], args.Skip(2));
			return 0;
		}
		var directory = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
		var paths = new LaunchPaths(Path.Combine(directory, "root with spaces") + "\\");
		var output = Path.Combine(directory, "argument-test-result.txt");
		Check(paths.GameInfo(), new[] { "Engine.EngineDir=" + paths.Root, "Engine.SupportDir=" + paths.Support,
			"Game.Mod=ra", "Graphics.Mode=Windowed", "Graphics.WindowedSize=1280,800" }, output, directory);
		var labRoot = Path.Combine(directory, "lab with spaces");
		var gameRoot = Path.Combine(labRoot, "openra");
		var manifest = Path.Combine(gameRoot, "mods", "ra", "mod.yaml");
		Directory.CreateDirectory(Path.GetDirectoryName(manifest));
		File.WriteAllText(manifest, "Metadata:");
		var labPaths = new LaunchPaths(labRoot);
		Check(labPaths.GameInfo(), new[] { "Engine.EngineDir=" + gameRoot, "Engine.SupportDir=" + labPaths.Support,
			"Game.Mod=ra", "Graphics.Mode=Windowed", "Graphics.WindowedSize=1280,800" }, output, directory);
		foreach (var file in new[] { labPaths.GameExecutable,
			Path.Combine(labPaths.Arena, ".venv", "Scripts", "python.exe"),
			Path.Combine(labPaths.Arena, "launcher_runner.py") })
		{
			Directory.CreateDirectory(Path.GetDirectoryName(file));
			File.WriteAllText(file, "test");
		}
		labPaths.Players = "local-rl";
		labPaths.Validate();
		if (!Directory.Exists(labPaths.Support))
			throw new Exception("The launcher did not create the content installer support directory.");
		Directory.Delete(labRoot, true);
		var stopFile = Path.Combine(directory, "stop file with spaces");
		Check(paths.AgentInfo(stopFile, 123), new[] { "-u", Path.Combine(paths.Arena, "launcher_runner.py"),
			"--stop-file", stopFile, "--parent-pid", "123", "--host", "https://ollama.com", "--watch-games" }, output, directory);
		paths.Players = "glm";
		Check(paths.AgentInfo(stopFile, 123), new[] { "-u", Path.Combine(paths.Arena, "launcher_runner.py"),
			"--stop-file", stopFile, "--parent-pid", "123", "--host", "https://ollama.com", "--watch-games",
			"--players", "glm" }, output, directory);
		paths.Players = "local-rl";
		Check(paths.AgentInfo(stopFile, 123), new[] { "-u", Path.Combine(paths.Arena, "launcher_runner.py"),
			"--stop-file", stopFile, "--parent-pid", "123", "--training", "--agents", "alpha:7703",
			"--checkpoint-dir", paths.CheckpointDirectory, "--batch-size", "4", "--horizon", "2",
			"--max-new-tokens", "32", "--interval", "3" }, output, directory);
		Console.WriteLine("OK: game and agent arguments survive Windows parsing, spaces and trailing backslashes.");
		return 0;
	}

	static void Check(ProcessStartInfo info, string[] expected, string output, string directory)
	{
		info.FileName = Assembly.GetExecutingAssembly().Location;
		info.WorkingDirectory = directory;
		info.Arguments = "--echo " + LaunchPaths.Quote(output) + " " + info.Arguments;
		using (var process = Process.Start(info))
		{
			if (!process.WaitForExit(5000)) throw new Exception("Argument test child did not exit.");
			if (process.ExitCode != 0) throw new Exception("Argument test child failed.");
		}
		if (!File.ReadAllLines(output).SequenceEqual(expected))
			throw new Exception("Windows argument boundaries were not preserved.");
	}
}
