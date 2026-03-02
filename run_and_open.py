from __future__ import annotations

"""Launcher for Streamlit app with optional auto-restart on crash.

Behavior:
- Run Streamlit as a subprocess and stream its output to this terminal.
- Open a browser when the server becomes available (unless `--no-browser`).
- If Streamlit exits with a non-zero code, restart after a short delay.
- If you press Ctrl+C in the terminal, the launcher will stop Streamlit and exit
  (no automatic restart).
"""

import argparse
import os
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from typing import Optional
import socket


APP_PATH = "app.py"
DEFAULT_PORT = 8501
HOST = "http://localhost"
URL_TEMPLATE = "{host}:{port}"


def find_chrome() -> Optional[str]:
    candidates = [
        os.path.join(os.environ.get("PROGRAMFILES", "C:\\Program Files"), "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(os.environ.get("PROGRAMFILES(X86)", "C:\\Program Files (x86)"), "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Google", "Chrome", "Application", "chrome.exe"),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def wait_for_server(url: str, timeout: float = 60.0, interval: float = 0.5) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(interval)
    return False


def find_free_port(start_port: int = DEFAULT_PORT, max_tries: int = 50) -> int:
    port = start_port
    tries = 0
    while tries < max_tries:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except OSError:
                port += 1
                tries += 1
    raise RuntimeError(f"Could not find a free port starting at {start_port}")


def stream_process_output(pipe):
    try:
        for line in iter(pipe.readline, ""):
            print(line, end="")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Run Streamlit app and optionally autorestart on crash")
    parser.add_argument("--port", "-p", type=int, default=DEFAULT_PORT)
    parser.add_argument("--timeout", "-t", type=float, default=60.0, help="Timeout waiting for server (seconds)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser")
    parser.add_argument("--no-restart", action="store_true", help="Do not restart on non-zero exit")
    args = parser.parse_args()

    # pick a free port starting from requested port
    try:
        selected_port = find_free_port(args.port)
    except Exception as e:
        print(f"Failed to find free port: {e}")
        selected_port = args.port

    url = URL_TEMPLATE.format(host=HOST, port=selected_port)
    python = sys.executable

    first_start = True
    try:
        while True:
            print(f"Starting Streamlit server on port {selected_port}...")
            proc = subprocess.Popen([python, "-m", "streamlit", "run", APP_PATH, "--server.port", str(selected_port), "--server.headless", "true"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

            t = threading.Thread(target=stream_process_output, args=(proc.stdout,), daemon=True)
            t.start()

            print(f"Streamlit pid={proc.pid}")

            print("Waiting for Streamlit to become available...")
            ready = wait_for_server(url, timeout=args.timeout)
            if ready:
                print(f"Server reachable at {url}")
                if not args.no_browser:
                    chrome = find_chrome()
                    if chrome:
                        try:
                            subprocess.Popen([chrome, url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        except Exception:
                            webbrowser.open(url)
                    else:
                        webbrowser.open(url)
                if first_start:
                    print(f"Opened browser at {url}")
            else:
                print("Server did not respond within timeout. Check Streamlit logs above.")

            first_start = False

            try:
                return_code = proc.wait()
            except KeyboardInterrupt:
                print("Stopping Streamlit...")
                try:
                    proc.terminate()
                except Exception:
                    pass
                proc.wait()
                break

            if return_code == 0 or args.no_restart:
                print("Streamlit exited cleanly; launcher will exit.")
                break
            else:
                print(f"Streamlit exited with code {return_code}; restarting in 2s...")
                time.sleep(2)

    except KeyboardInterrupt:
        print("Keyboard interrupt received — exiting launcher.")


if __name__ == "__main__":
    main()
