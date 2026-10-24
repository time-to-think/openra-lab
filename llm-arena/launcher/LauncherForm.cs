using System;
using System.Diagnostics;
using System.Drawing;
using System.Windows.Forms;

namespace OpenRA.ArenaLauncher
{
	sealed class LauncherForm : Form
	{
		readonly LaunchPaths paths;
		readonly Label gameStatus = new Label();
		readonly Label agentStatus = new Label();
		readonly RichTextBox output = new RichTextBox();
		readonly Button start = new Button();
		readonly Button stop = new Button();
		readonly ComboBox players = new ComboBox();
		readonly Label instructions = new Label();
		readonly Label heading = new Label();
		static readonly string[][] PlayerChoices =
		{
			new[] { "Qwen RL vs OpenRA AI", "local-rl" },
			new[] { "GLM vs DeepSeek", null },
			new[] { "GLM vs OpenRA AI", "glm" },
			new[] { "DeepSeek vs OpenRA AI", "deepseek" },
		};
		readonly Timer timer = new Timer();
		LauncherSession session;
		bool closing;

		public LauncherForm(LaunchPaths paths)
		{
			this.paths = paths;
			Text = "OpenRA LLM Arena";
			StartPosition = FormStartPosition.CenterScreen;
			Size = new Size(860, 640);
			MinimumSize = new Size(700, 520);
			Font = new Font("Segoe UI", 10);
			BackColor = Color.FromArgb(242, 245, 249);

			var layout = new TableLayoutPanel { Dock = DockStyle.Fill, Padding = new Padding(20), RowCount = 6, ColumnCount = 1 };
			foreach (var height in new[] { 38, 100, 28, 28, 44 })
				layout.RowStyles.Add(new RowStyle(SizeType.Absolute, height));
			layout.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
			heading.Dock = instructions.Dock = DockStyle.Fill;
			heading.Font = new Font("Segoe UI", 18, FontStyle.Bold);
			layout.Controls.Add(heading, 0, 0);
			layout.Controls.Add(instructions, 0, 1);
			gameStatus.Dock = agentStatus.Dock = DockStyle.Fill;
			gameStatus.Text = "Game: starting";
			agentStatus.Text = "Agents: checking connection";
			layout.Controls.Add(gameStatus, 0, 2);
			layout.Controls.Add(agentStatus, 0, 3);

			var buttons = new FlowLayoutPanel { Dock = DockStyle.Fill };
			start.Text = "Start / Retry";
			start.AutoSize = true;
			start.Click += delegate { StartSession(); };
			stop.Text = "Stop agents";
			stop.AutoSize = true;
			stop.Click += delegate { if (session != null) session.StopAgents(); };
			var logs = new Button { Text = "Open logs", AutoSize = true };
			logs.Click += delegate { OpenLogs(); };
			players.DropDownStyle = ComboBoxStyle.DropDownList;
			players.Width = 300;
			foreach (var choice in PlayerChoices)
				players.Items.Add(choice[0]);
			players.SelectedIndex = 0;
			paths.Players = PlayerChoices[0][1];
			UpdateInstructions();
			players.SelectedIndexChanged += delegate
			{
				paths.Players = PlayerChoices[players.SelectedIndex][1];
				UpdateInstructions();
				if (session != null && session.AgentsRunning)
					Report("Launcher", "Mode changed to " + players.SelectedItem + ". Select Restart agents to apply it.");
			};
			buttons.Controls.AddRange(new Control[] { players, start, stop, logs });
			layout.Controls.Add(buttons, 0, 4);

			output.Dock = DockStyle.Fill;
			output.ReadOnly = true;
			output.BackColor = Color.FromArgb(22, 28, 38);
			output.ForeColor = Color.FromArgb(223, 233, 243);
			output.Font = new Font("Consolas", 9);
			output.BorderStyle = BorderStyle.None;
			layout.Controls.Add(output, 0, 5);
			Controls.Add(layout);
			Shown += delegate { StartSession(); };
			timer.Interval = 1000;
			timer.Tick += delegate { RefreshStatus(); };
			timer.Start();
			FormClosing += delegate
			{
				closing = true;
				timer.Stop();
				if (session != null) session.StopAgents();
			};
		}

		void UpdateInstructions()
		{
			var mode = paths.Players;
			heading.Text = mode == "local-rl" ? "OpenRA  |  Local Qwen Training" : "OpenRA  |  Ollama Cloud";
			var bots = mode == "local-rl" ? "Qwen (LLM)" : mode == "glm" ? "GLM (LLM)" : "DeepSeek (LLM)";
			instructions.Text = mode == null
				? "1. Choose Skirmish and add GLM (LLM) and DeepSeek (LLM) on opposing teams.\n"
					+ "2. Choose Spectator and start the match."
				: "1. Choose Skirmish and add " + bots + " and Rush, Normal, Turtle or Naval AI on opposing teams.\n"
					+ (mode == "local-rl" ? "2. Choose Spectator and start. Start another match to continue training."
						: "2. Choose Spectator and start the match.");
		}

		void StartSession()
		{
			try
			{
				agentStatus.Text = paths.Players == "local-rl" ? "Learner: loading local Qwen adapter"
					: "Agents: checking Cloud authentication and MCP";
				// Keep the same session's game reference when restarting agents.
				if (session == null) session = new LauncherSession(paths, Report);
				session.RestartAgents();
				RefreshStatus();
			}
			catch (Exception error)
			{
				agentStatus.Text = "Agents: startup failed";
				Report("Launcher", error.Message);
				MessageBox.Show(this, error.Message, Text, MessageBoxButtons.OK, MessageBoxIcon.Error);
			}
		}

		void RefreshStatus()
		{
			if (session == null)
				return;
			try { session.Poll(); }
			catch (Exception error) { Report("Launcher", error.Message); }
			gameStatus.Text = session.GameRunning ? "Game: running - select the Red Alert window" : "Game: closed";
			if (!session.AgentsRunning && session.AgentExitCode.HasValue)
				agentStatus.Text = session.AgentExitCode.Value == 0 ? "Agents: stopped / match finished"
					: "Agents: failed - see the log below";
			if (session.GameExitCode.HasValue && session.GameExitCode.Value != 0)
				gameStatus.Text = "Game: failed - see game.log";
			start.Enabled = !session.RestartPending;
			start.Text = session.AgentsRunning ? "Restart agents" : "Start / Retry";
			if (session.RestartPending) agentStatus.Text = "Agents: stopping previous session before restart";
			stop.Enabled = session.AgentsRunning;
		}

		void Report(string source, string line)
		{
			if (closing || IsDisposed || !IsHandleCreated)
				return;
			// The commander's next line describes this expected startup condition.
			if (source == "Agents" && line.StartsWith("Tool 'get_state' failed:") && line.Contains("not reachable"))
				return;
			if (source == "Agents" && line.Contains("waiting for game"))
				line = "waiting for game: start or resume the selected skirmish.";
			try
			{
				BeginInvoke((Action)delegate
				{
					if (closing) return;
					if (output.TextLength > 120000) output.Clear();
					output.AppendText("[" + source + "] " + line + Environment.NewLine);
					output.SelectionStart = output.TextLength;
					output.ScrollToCaret();
					if (source == "Agents")
					{
						if (line.Contains("waiting for game")) agentStatus.Text = "Agents: waiting for the selected skirmish";
						else if (line.Contains("new game detected")) agentStatus.Text = "Agents: connected to game, loading new briefing";
						else if (line.Contains("update ")) agentStatus.Text = "Learner: updating local Qwen adapter";
						else if (line.Contains("turn ") || line.Contains(") ->")) agentStatus.Text = "Agents: playing - live commands below";
						else if (line.Contains("LLM request failed")) agentStatus.Text = "Agents: Cloud request failed - retrying";
					}
				});
			}
			catch (InvalidOperationException) { }
		}

		void OpenLogs()
		{
			try
			{
				var directory = session != null && session.LogDirectory != null ? session.LogDirectory : paths.Logs;
				System.IO.Directory.CreateDirectory(directory);
				Process.Start(new ProcessStartInfo("explorer.exe", LaunchPaths.Quote(directory)) { UseShellExecute = true });
			}
			catch (Exception error) { Report("Launcher", error.Message); }
		}
	}
}
