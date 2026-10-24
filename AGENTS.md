# Repository guidance

- Write source, comments, documentation, and commit messages in English.
- Keep OpenRA engine and bot bridge changes in a pull request to `time-to-think/openra`. While that PR is
  open, pin a commit published in the contributor fork; after merge, switch to the organization core URL.
- Keep checkpoints, optimizer state, learning logs, game content, local runtimes, virtual environments,
  credentials, and generated executables out of Git.
- Never commit local `AGENTS.local.md` or other `*.local.md` agent notes.
- Verify Python tests and the launcher build before publishing changes to `llm-arena/`.
