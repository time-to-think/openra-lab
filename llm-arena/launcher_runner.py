"""Run cloud commanders or the local learner until the desktop launcher stops."""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import sys
from ctypes import wintypes
from pathlib import Path

import arena
from cloud import CloudError


async def wait_for_stop(stop_file: Path, parent_pid: int) -> None:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x00100000, False, parent_pid)  # SYNCHRONIZE only; no process modification rights.
    if not handle:
        return
    try:
        while not stop_file.exists() and kernel.WaitForSingleObject(handle, 0) == 258:  # WAIT_TIMEOUT
            await asyncio.sleep(0.5)
    finally:
        kernel.CloseHandle(handle)


async def supervise(stop_file: Path, parent_pid: int, arena_args: list[str], training: bool = False) -> None:
    if stop_file.exists():
        return
    if training:
        import train
        main = train.main
        script = train.HERE / "train.py"
    else:
        main = arena.main
        script = arena.HERE / "arena.py"
    sys.argv = [str(script), *arena_args]
    watcher = asyncio.create_task(wait_for_stop(stop_file, parent_pid))
    agents = asyncio.create_task(main())
    try:
        done, _ = await asyncio.wait([watcher, agents], return_when=asyncio.FIRST_COMPLETED)
        if agents in done:
            await agents
        else:
            print("Launcher closed or Stop agents selected. Shutting down MCP sessions.", flush=True)
    finally:
        for task in (watcher, agents):
            if not task.done():
                task.cancel()
        await asyncio.gather(watcher, agents, return_exceptions=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--training", action="store_true")
    options, arena_args = parser.parse_known_args()
    try:
        asyncio.run(supervise(options.stop_file, options.parent_pid, arena_args, options.training))
    except (CloudError, ValueError) as error:
        print(f"Setup error: {error}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
