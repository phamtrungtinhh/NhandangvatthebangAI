#!/usr/bin/env python3
"""
launcher_local.py

Windows-friendly launcher that:
- copies the project to a local folder (default D:\\workspace) using robocopy
- creates a local .venv if missing
- installs requirements.txt into the venv
- kills any process listening on the target port (8501)
- starts Streamlit using the venv Python and logs output
- polls the server until ready and opens the browser

Usage:
    py launcher_local.py [--target D:\\workspace] [--port 8501]

"""
import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
import webbrowser

DEFAULT_TARGET = r"D:\workspace"
APP = "app.py"
PORT = 8501
HOST = "127.0.0.1"
URL = f"http://{HOST}:{PORT}"


def run(cmd, check=False, capture=False, shell=False):
    print("$", " ".join(cmd) if isinstance(cmd, (list, tuple)) else cmd)
    if capture:
        return subprocess.run(cmd, check=check, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=shell)
    return subprocess.run(cmd, check=check, shell=shell)


def robocopy_copy(src, dst):
    # Use robocopy on Windows for robustness when src is UNC
    if platform.system().lower() != 'windows':
        # fallback to shutil.copytree
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        return 0

    # Build robocopy args. Exclude .venv, datasets, models, runs, node_modules
    excludes = [
        os.path.join(src, '.venv'),
        os.path.join(src, 'dataset'),
        os.path.join(src, 'datasets'),
        os.path.join(src, 'models'),
        os.path.join(src, 'runs'),
        os.path.join(src, 'node_modules'),
    ]
    # robocopy wants paths relative to src for /XD; we'll pass names
    exclude_names = [os.path.basename(p) for p in excludes if p]
    args = [
        'robocopy', src, dst, '/MIR',
    ]
    # Append exclusions
    args += ['/XD'] + exclude_names
    # Exclude large file masks
    args += ['/XF', '*.pt', '*.pth', '*.zip', '*.tar.gz', '*.tgz', '*.whl']

    res = subprocess.run(args)
    return res.returncode


def ensure_venv(target):
    venv_py = os.path.join(target, '.venv', 'Scripts', 'python.exe') if platform.system().lower() == 'windows' else os.path.join(target, '.venv', 'bin', 'python')
    if os.path.exists(venv_py):
        return venv_py
    # create venv using system py launcher if available
    print('Creating venv in', os.path.join(target, '.venv'))
    try:
        subprocess.run([sys.executable, '-m', 'venv', os.path.join(target, '.venv')], check=True)
    except Exception as e:
        print('Failed to create venv:', e)
        raise
    return venv_py


def pip_install(venv_py, target):
    req = os.path.join(target, 'requirements.txt')
    # upgrade pip
    subprocess.run([venv_py, '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel'])
    if os.path.exists(req):
        print('Installing requirements from', req)
        subprocess.run([venv_py, '-m', 'pip', 'install', '-r', req])
    else:
        print('No requirements.txt found, skipping pip install')


def kill_port(port):
    if platform.system().lower() != 'windows':
        return
    # Find PID listening on port via netstat
    try:
        out = subprocess.check_output(['netstat', '-ano'], universal_newlines=True)
    except Exception:
        return
    pids = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0].lower().startswith('tcp'):
            local = parts[1]
            pid = parts[-1]
            if local.endswith(f':{port}') or local.endswith(f'.{port}'):
                pids.add(pid)
    for pid in pids:
        try:
            subprocess.run(['taskkill', '/PID', pid, '/F'])
            print('Killed PID', pid)
        except Exception:
            pass


def start_streamlit(venv_py, target, port=PORT):
    cwd = target
    cmd = [venv_py, '-m', 'streamlit', 'run', APP, '--server.port', str(port), '--server.address', HOST]
    logfile = os.path.join(target, 'launcher_streamlit.log')
    print('Starting streamlit with:', cmd)
    f = open(logfile, 'ab')
    # Ensure the venv Scripts/bin directory is first on PATH so any child subprocesses
    # that call `python` resolve to the venv interpreter instead of a global shim.
    env = os.environ.copy()
    venv_bin = os.path.join(target, '.venv', 'Scripts') if platform.system().lower() == 'windows' else os.path.join(target, '.venv', 'bin')
    env['PATH'] = venv_bin + os.pathsep + env.get('PATH', '')
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=env)
    return proc, logfile


def poll_ready(url, timeout=60):
    deadline = time.time() + timeout
    import urllib.request
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def tail_file(path, lines=30):
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = 1024
            data = b''
            while size > 0 and lines > 0:
                read = min(block, size)
                f.seek(size - read)
                chunk = f.read(read)
                data = chunk + data
                size -= read
                lines -= chunk.count(b'\n')
            return data.decode(errors='replace').splitlines()[-30:]
    except Exception:
        return []


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--target', default=DEFAULT_TARGET)
    p.add_argument('--port', type=int, default=PORT)
    args = p.parse_args()

    src = os.path.abspath(os.getcwd())
    dst = os.path.abspath(args.target)
    port = args.port

    print('Source:', src)
    print('Target:', dst)

    # If the requested target exists but is not a directory (e.g. Windows shortcut file),
    # redirect to a safe sibling folder.
    if os.path.exists(dst) and not os.path.isdir(dst):
        redirected = dst + "_real"
        print('Target path is not a directory (possibly a shortcut file):', dst)
        print('Redirecting local target to:', redirected)
        dst = redirected

    if not os.path.exists(dst):
        try:
            os.makedirs(dst, exist_ok=True)
        except Exception as e:
            print('Failed to create target directory:', dst)
            print('Error:', repr(e))
            # Try a safe fallback to DEFAULT_TARGET (if different)
            if os.path.abspath(dst) != os.path.abspath(DEFAULT_TARGET):
                fallback = DEFAULT_TARGET
                print('Falling back to default target:', fallback)
                dst = fallback
                try:
                    os.makedirs(dst, exist_ok=True)
                except Exception as e2:
                    print('Also failed to create fallback target:', fallback)
                    print('Error:', repr(e2))
                    print('Please check that the path is valid and accessible.')
                    sys.exit(4)
            else:
                print('Please check that the path is valid and accessible.')
                sys.exit(4)

    print('Copying project to local target (robocopy) ...')
    rc = robocopy_copy(src, dst)
    if rc >= 8:
        print('Robocopy failed with code', rc)
        print('Aborting')
        sys.exit(2)

    print('Ensuring venv...')
    venv_py = os.path.join(dst, '.venv', 'Scripts', 'python.exe') if platform.system().lower() == 'windows' else os.path.join(dst, '.venv', 'bin', 'python')
    if not os.path.exists(venv_py):
        try:
            subprocess.run([sys.executable, '-m', 'venv', os.path.join(dst, '.venv')], check=True)
        except Exception as e:
            print('Failed to create venv:', e)
            sys.exit(3)

    print('Installing requirements (may take a while)')
    pip_install(venv_py, dst)

    print('Killing any process on port', port)
    kill_port(port)

    print('Starting streamlit in background')
    proc, logfile = start_streamlit(venv_py, dst, port=port)

    launch_url = f"http://{HOST}:{port}"
    print('Waiting for server to become ready at', launch_url)
    ok = poll_ready(launch_url, timeout=120)
    if ok:
        print('Server ready — opening browser', launch_url)
        try:
            webbrowser.open(launch_url)
        except Exception:
            pass
    else:
        print('Server did not become ready within timeout. See log:', logfile)

    print('\n--- Last 30 lines of log ---')
    for l in tail_file(logfile, lines=30):
        print(l)

    print('Launcher finished. Streamlit PID:', getattr(proc, 'pid', None))


if __name__ == '__main__':
    main()
