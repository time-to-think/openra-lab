using System;
using System.Diagnostics;
using System.IO;
using System.Text;

namespace OpenRA.ArenaLauncher
{
	sealed class LauncherSession
	{
		readonly LaunchPaths paths;
		readonly Action<string, string> report;
		readonly object logLock = new object();
		readonly string secret;
		string stopFile;
		Process agents;
		Process game;
		bool stopping;
		bool restartRequested;
		public bool RestartPending { get { return restartRequested; } }
		public string LogDirectory { get; private set; }
		public bool AgentsRunning { get { return IsRunning(agents); } }
		public bool GameRunning { get { return IsRunning(game); } }
		public int? AgentExitCode { get { return agents != null && !IsRunning(agents) ? (int?)agents.ExitCode : null; } }
		public int? GameExitCode { get { return game != null && !IsRunning(game) ? (int?)game.ExitCode : null; } }

		public LauncherSession(LaunchPaths paths, Action<string, string> report)
		{
			this.paths = paths;
			this.report = report;
			secret = paths.ApiKey;
		}

		public void Start()
		{
			if (AgentsRunning)
				throw new InvalidOperationException("Wait for the previous agents to stop before starting again.");
			if (agents != null)
			{
				agents.WaitForExit();
				agents.Dispose();
				agents = null;
			}
			paths.Validate();
			stopping = false;
			var name = DateTime.Now.ToString("yyyyMMdd-HHmmss") + "-" + Guid.NewGuid().ToString("N").Substring(0, 6);
			LogDirectory = Path.Combine(paths.Logs, name);
			Directory.CreateDirectory(LogDirectory);
			stopFile = Path.Combine(LogDirectory, "stop-agents");
			if (!GameRunning)
			{
				game = FindExistingGame();
				if (game == null)
					game = StartChild("Game", paths.GameInfo());
				else
					Write("Launcher", "Reusing the open Red Alert game window.");
			}
			agents = StartChild("Agents", paths.AgentInfo(stopFile, Process.GetCurrentProcess().Id));
			Write("Launcher", "Game and agents started. Add the LLM bots for the selected mode in Skirmish and select Spectator.");
		}

		public void StopAgents()
		{
			restartRequested = false;
			RequestStop();
		}

		public void RestartAgents()
		{
			if (!AgentsRunning)
				Start();
			else
			{
				restartRequested = true;
				RequestStop();
			}
		}

		void RequestStop()
		{
			if (!AgentsRunning || stopping)
				return;
			stopping = true;
			File.WriteAllText(stopFile, "stop", Encoding.UTF8);
			Write("Launcher", "Stopping agents and their MCP servers. The game window stays open.");
		}

		public void Poll()
		{
			if (game != null && !GameRunning && AgentsRunning)
				StopAgents();
			if (restartRequested && !AgentsRunning)
			{
				restartRequested = false;
				Start();
			}
		}

		Process FindExistingGame()
		{
			foreach (var candidate in Process.GetProcessesByName("OpenRA"))
			{
				try
				{
					if (candidate.MainWindowHandle != IntPtr.Zero && string.Equals(candidate.MainModule.FileName,
						paths.GameExecutable, StringComparison.OrdinalIgnoreCase))
						return candidate;
				}
				catch (System.ComponentModel.Win32Exception) { }
				catch (InvalidOperationException) { }
				candidate.Dispose();
			}
			return null;
		}

		Process StartChild(string name, ProcessStartInfo info)
		{
			var child = new Process { StartInfo = info };
			var directory = LogDirectory;
			child.OutputDataReceived += delegate(object sender, DataReceivedEventArgs e) { if (e.Data != null) Write(name, e.Data, directory); };
			child.ErrorDataReceived += delegate(object sender, DataReceivedEventArgs e) { if (e.Data != null) Write(name, e.Data, directory); };
			child.Start();
			child.BeginOutputReadLine();
			child.BeginErrorReadLine();
			return child;
		}

		void Write(string name, string line, string directory = null)
		{
			directory = directory ?? LogDirectory;
			if (secret.Length > 0)
				line = line.Replace(secret, "[redacted]");
			lock (logLock)
				File.AppendAllText(Path.Combine(directory, name.ToLowerInvariant() + ".log"),
					DateTime.Now.ToString("HH:mm:ss") + " " + line + Environment.NewLine, Encoding.UTF8);
			if (directory == LogDirectory)
				report(name, line);
		}

		static bool IsRunning(Process process)
		{
			if (process == null)
				return false;
			try { return !process.HasExited; }
			catch (InvalidOperationException) { return false; }
		}
	}
}
