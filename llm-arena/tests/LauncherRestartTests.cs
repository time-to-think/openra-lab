using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Threading;
using OpenRA.ArenaLauncher;

static class LauncherRestartTests
{
	static int Main(string[] args)
	{
		if (args.Length > 0 && args[0].StartsWith("Engine.EngineDir="))
		{
			while (!File.Exists("stop-game")) Thread.Sleep(20);
			return 0;
		}
		if (Array.IndexOf(args, "--stop-file") >= 0)
		{
			var stop = args[Array.IndexOf(args, "--stop-file") + 1];
			// Only one agent session may own this lock, including during restarts.
			using (new FileStream("agent-session.lock", FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None))
			{
				Console.WriteLine("READY " + Process.GetCurrentProcess().Id);
				while (!File.Exists(stop)) Thread.Sleep(20);
				Thread.Sleep(100);
				Console.WriteLine("STOPPED");
			}
			return 0;
		}

		var root = Path.Combine(Path.GetTempPath(), "arena-restart-" + Guid.NewGuid().ToString("N"));
		var paths = new LaunchPaths(root);
		var self = Assembly.GetExecutingAssembly().Location;
		LauncherSession session = null;
		try
		{
			Prepare(paths, self);
			session = new LauncherSession(paths, delegate { });
			session.Start();
			Wait(session, delegate { return Ready(session); });
			var first = session.LogDirectory;
			session.RestartAgents();
			if (!session.RestartPending) throw new Exception("A running session must stop before restart.");
			Wait(session, delegate { return session.LogDirectory != first && Ready(session); });
			if (!session.GameRunning) throw new Exception("Restart closed the game.");
			if (!File.ReadAllText(Path.Combine(first, "agents.log")).Contains("STOPPED"))
				throw new Exception("Previous agents did not finish before starting new agents.");
			if (File.ReadAllText(Path.Combine(session.LogDirectory, "agents.log")).Contains("STOPPED"))
				throw new Exception("Previous agent output leaked into the new log.");
			session.StopAgents();
			Wait(session, delegate { return !session.AgentsRunning; });
			if (!session.GameRunning) throw new Exception("Stopping agents closed the game.");
			Console.WriteLine("OK: restart waits for previous agents, preserves the game and separates logs.");
			return 0;
		}
		finally
		{
			if (session != null) session.StopAgents();
			File.WriteAllText(Path.Combine(root, "stop-game"), "stop");
			if (session != null) Wait(session, delegate { return !session.GameRunning && !session.AgentsRunning; });
			Directory.Delete(root, true);
		}
	}

	static bool Ready(LauncherSession session)
	{
		var log = Path.Combine(session.LogDirectory, "agents.log");
		return session.AgentsRunning && File.Exists(log) && File.ReadAllText(log).Contains("READY");
	}

	static void Wait(LauncherSession session, Func<bool> completed)
	{
		var deadline = DateTime.UtcNow.AddSeconds(10);
		while (!completed())
		{
			if (DateTime.UtcNow > deadline) throw new Exception("Launcher restart test timed out.");
			session.Poll();
			Thread.Sleep(20);
		}
	}

	static void Prepare(LaunchPaths paths, string executable)
	{
		var files = new[] { Path.Combine(paths.Root, "bin", "OpenRA.exe"),
			Path.Combine(paths.Arena, ".venv", "Scripts", "python.exe"),
			Path.Combine(paths.Arena, "launcher_runner.py") };
		foreach (var file in files)
		{
			Directory.CreateDirectory(Path.GetDirectoryName(file));
			File.Copy(executable, file);
		}
		Environment.SetEnvironmentVariable("OLLAMA_API_KEY", "offline-test-key");
	}
}
