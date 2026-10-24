using System;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Windows.Forms;

namespace OpenRA.ArenaLauncher
{
	static class Program
	{
		[STAThread]
		static int Main(string[] args)
		{
			var root = AppDomain.CurrentDomain.BaseDirectory;
			var paths = new LaunchPaths(root);
			if (Array.IndexOf(args, "--check") >= 0)
			{
				Directory.CreateDirectory(paths.Logs);
				try
				{
					paths.Validate();
						File.WriteAllText(Path.Combine(paths.Logs, "check.txt"), "OK: game build, Python and selected mode are configured. Game content is checked at launch.");
					return 0;
				}
				catch (Exception error)
				{
					File.WriteAllText(Path.Combine(paths.Logs, "check.txt"), "ERROR: " + error.Message);
					return 1;
				}
			}
			Application.EnableVisualStyles();
			Application.SetCompatibleTextRenderingDefault(false);
			bool created;
			using (var mutex = new Mutex(true, MutexName(root), out created))
			{
				if (!created)
				{
					MessageBox.Show("The launcher is already open. Select OpenRA LLM Arena on the taskbar.", "OpenRA LLM Arena");
					return 0;
				}
				try { Application.Run(new LauncherForm(paths)); }
				finally { mutex.ReleaseMutex(); }
			}
			return 0;
		}

		static string MutexName(string root)
		{
			using (var hash = SHA256.Create())
				return "Local\\OpenRAArena-" + BitConverter.ToString(hash.ComputeHash(Encoding.UTF8.GetBytes(root.ToLowerInvariant()))).Replace("-", "");
		}
	}
}
