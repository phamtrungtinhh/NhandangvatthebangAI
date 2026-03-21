from __future__ import annotations

"""Launcher for Streamlit app with hot-restart for local fixes.

Behavior:
- Always prefer the project `.venv` Python for the Streamlit child process.
- Replace the previous Streamlit child from this workspace before starting.
- Open a browser when the server becomes available (unless `--no-browser`).
- Restart Streamlit automatically when project code/config files change.
- Optionally restart on non-zero exit unless `--no-restart` is passed.
"""

import argparse
import os
import pathlib
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from typing import Optional


APP_PATH = "app.py"
DEFAULT_PORT = 8501
HOST = "http://localhost"
URL_TEMPLATE = "{host}:{port}"
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
PID_FILE = PROJECT_ROOT / ".streamlit_app.pid"
LAUNCHER_PID_FILE = PROJECT_ROOT / ".streamlit_launcher.pid"
WATCH_EXTENSIONS = {".py", ".toml", ".json", ".yaml", ".yml"}
WATCH_IGNORED_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    "runs",
}


def find_chrome() -> Optional[str]:
    candidates = [
        os.path.join(os.environ.get("PROGRAMFILES", "C:\\Program Files"), "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(os.environ.get("PROGRAMFILES(X86)", "C:\\Program Files (x86)"), "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Google", "Chrome", "Application", "chrome.exe"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def wait_for_server(url: str, timeout: float = 60.0, interval: float = 0.5) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(interval)
    return False


def get_streamlit_python() -> str:
    venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def terminate_pid(pid: int) -> None:
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            os.kill(pid, 15)
    except Exception:
        pass


def read_pid_file() -> Optional[int]:
    try:
        if not PID_FILE.exists():
            return None
        raw = PID_FILE.read_text(encoding="utf-8").strip()
        return int(raw) if raw else None
    except Exception:
        return None


def write_pid_file(pid: int) -> None:
    try:
        PID_FILE.write_text(str(int(pid)), encoding="utf-8")
    except Exception:
        pass


def write_launcher_pid_file(pid: int) -> None:
    try:
        LAUNCHER_PID_FILE.write_text(str(int(pid)), encoding="utf-8")
    except Exception:
        pass


def read_launcher_pid_file() -> Optional[int]:
    try:
        if not LAUNCHER_PID_FILE.exists():
            return None
        raw = LAUNCHER_PID_FILE.read_text(encoding="utf-8").strip()
        return int(raw) if raw else None
    except Exception:
        return None


def clear_launcher_pid_file(expected_pid: Optional[int] = None) -> None:
    try:
        if not LAUNCHER_PID_FILE.exists():
            return
        if expected_pid is None:
            LAUNCHER_PID_FILE.unlink(missing_ok=True)
            return
        current_pid = read_launcher_pid_file()
        if current_pid == int(expected_pid):
            LAUNCHER_PID_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def clear_pid_file(expected_pid: Optional[int] = None) -> None:
    try:
        if not PID_FILE.exists():
            return
        if expected_pid is None:
            PID_FILE.unlink(missing_ok=True)
            return
        current_pid = read_pid_file()
        if current_pid == int(expected_pid):
            PID_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def stop_previous_launcher() -> None:
    stale_pid = read_launcher_pid_file()
    if stale_pid is None or stale_pid == os.getpid():
        return
    print(f"Stopping previous launcher pid={stale_pid}...")
    terminate_pid(stale_pid)
    time.sleep(1.0)
    clear_launcher_pid_file()


def stop_previous_streamlit_child() -> None:
    stale_pid = read_pid_file()
    if stale_pid is None:
        return
    print(f"Stopping previous Streamlit child pid={stale_pid}...")
    terminate_pid(stale_pid)
    time.sleep(1.0)
    clear_pid_file()


def build_watch_snapshot(root: pathlib.Path) -> dict[str, float]:
    snapshot: dict[str, float] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in WATCH_IGNORED_DIRS]
        for filename in filenames:
            path = pathlib.Path(dirpath) / filename
            if path.suffix.lower() not in WATCH_EXTENSIONS:
                continue
            try:
                snapshot[str(path.resolve())] = path.stat().st_mtime
            except Exception:
                continue
    return snapshot


def snapshots_differ(old: dict[str, float], new: dict[str, float]) -> bool:
    if old.keys() != new.keys():
        return True
    for key, old_mtime in old.items():
        if abs(float(new.get(key, -1.0)) - float(old_mtime)) > 1e-9:
            return True
    return False


def stream_process_output(pipe) -> None:
    try:
        for line in iter(pipe.readline, ""):
            print(line, end="")
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Streamlit app and hot-restart when local fixes are saved")
    parser.add_argument("--port", "-p", type=int, default=DEFAULT_PORT)
    parser.add_argument("--timeout", "-t", type=float, default=60.0, help="Timeout waiting for server (seconds)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser")
    parser.add_argument("--no-restart", action="store_true", help="Do not restart on non-zero exit")
    parser.add_argument("--no-watch", action="store_true", help="Do not restart when project files change")
    args = parser.parse_args()

    stop_previous_launcher()
    write_launcher_pid_file(os.getpid())
    stop_previous_streamlit_child()

    selected_port = int(args.port)
    url = URL_TEMPLATE.format(host=HOST, port=selected_port)
    python = get_streamlit_python()
    watch_enabled = not bool(args.no_watch)
    watch_snapshot = build_watch_snapshot(PROJECT_ROOT) if watch_enabled else {}

    first_start = True
    try:
        while True:
            print(f"Starting Streamlit server on port {selected_port} with {python}...")
            proc = subprocess.Popen(
                [python, "-m", "streamlit", "run", APP_PATH, "--server.port", str(selected_port), "--server.headless", "true"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            write_pid_file(proc.pid)

            stream_thread = threading.Thread(target=stream_process_output, args=(proc.stdout,), daemon=True)
            stream_thread.start()

            print(f"Streamlit pid={proc.pid}")
            print("Waiting for Streamlit to become available...")

            ready = wait_for_server(url, timeout=args.timeout)
            if ready:
                print(f"Server reachable at {url}")
                if not args.no_browser and first_start:
                    chrome = find_chrome()
                    if chrome:
                        try:
                            subprocess.Popen([chrome, url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        except Exception:
                            webbrowser.open(url)
                    else:
                        webbrowser.open(url)
                    print(f"Opened browser at {url}")
            else:
                print("Server did not respond within timeout. Check Streamlit logs above.")

            first_start = False
            restart_for_changes = False

            try:
                while True:
                    try:
                        return_code = proc.wait(timeout=1.0)
                        break
                    except subprocess.TimeoutExpired:
                        if not watch_enabled:
                            continue
                        current_snapshot = build_watch_snapshot(PROJECT_ROOT)
                        if snapshots_differ(watch_snapshot, current_snapshot):
                            print("Detected project changes. Restarting Streamlit to apply the latest fixes...")
                            watch_snapshot = current_snapshot
                            restart_for_changes = True
                            try:
                                proc.terminate()
                                proc.wait(timeout=10)
                            except Exception:
                                terminate_pid(proc.pid)
                            return_code = None
                            break
            except KeyboardInterrupt:
                print("Stopping Streamlit...")
                try:
                    proc.terminate()
                    proc.wait(timeout=10)
                except Exception:
                    terminate_pid(proc.pid)
                clear_pid_file(proc.pid)
                break

            clear_pid_file(proc.pid)

            if restart_for_changes:
                continue

            if return_code == 0 or args.no_restart:
                print("Streamlit exited cleanly; launcher will exit.")
                break

            print(f"Streamlit exited with code {return_code}; restarting in 2s...")
            time.sleep(2)
            if watch_enabled:
                watch_snapshot = build_watch_snapshot(PROJECT_ROOT)

    except KeyboardInterrupt:
        clear_pid_file()
        clear_launcher_pid_file(os.getpid())
        print("Keyboard interrupt received - exiting launcher.")
    finally:
        clear_pid_file()
        clear_launcher_pid_file(os.getpid())


if __name__ == "__main__":
    main()
