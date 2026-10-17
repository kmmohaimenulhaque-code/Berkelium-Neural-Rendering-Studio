"""
Entry point for the backend when it ships inside the desktop app.

During development you still run:  uvicorn main:app --reload
The installed app instead starts this file (frozen by PyInstaller), and the
Electron main process owns its life:

  BERKELIUM_PORT          the port Electron picked (a free one, so a second
                          copy of the app or another local server never clashes)
  BERKELIUM_CONFIG        the per-user config.json in the app data folder
  BERKELIUM_LOG_DIR       where backend.log goes
  BERKELIUM_WATCH_STDIN   "1" means: exit when stdin closes

Why the stdin watch: if Electron crashes or is force-quit, it never gets the
chance to stop us, and an orphan backend would hold the port and keep the
user's key in memory. Electron keeps a pipe open to our stdin; the OS closes
it when Electron dies for any reason, and we exit on that EOF. This works the
same on macOS, Windows and Linux, unlike polling a parent PID.
"""

import os
import sys
import threading

DEFAULT_PORT = 8000


def port_from_env(env) -> int:
    raw = env.get("BERKELIUM_PORT", "")
    if not raw:
        return DEFAULT_PORT
    if not raw.isdigit() or not (1 <= int(raw) <= 65535):
        raise SystemExit(f"BERKELIUM_PORT must be a number from 1 to 65535, got {raw!r}")
    return int(raw)


def exit_when_closed(stream, on_close) -> None:
    """Blocks until the stream hits EOF, then calls on_close."""
    try:
        while stream.read(1024):
            pass
    except (OSError, ValueError):
        pass  # a broken pipe means the same thing as EOF here
    on_close()


def main() -> None:
    port = port_from_env(os.environ)
    if os.environ.get("BERKELIUM_WATCH_STDIN") == "1":
        # os._exit, not sys.exit: we must stop even if a request is mid-flight,
        # and sys.exit from a thread would only end that thread.
        threading.Thread(target=exit_when_closed, args=(sys.stdin.buffer, lambda: os._exit(0)),
                         daemon=True, name="parent-watch").start()

    import uvicorn
    import main as app_module  # imported after the env is read: it loads config at import

    # 127.0.0.1 only. The backend holds API keys and must never be reachable
    # from the network, even on a cafe or campus Wi-Fi.
    uvicorn.run(app_module.app, host="127.0.0.1", port=port, log_config=None)


if __name__ == "__main__":
    main()
