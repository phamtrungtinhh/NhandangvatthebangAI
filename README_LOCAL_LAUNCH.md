Local launcher
=================

Usage
-----

1. From any shell on the machine that can access the network share, run:

```powershell
pushd \\\\192.168.1.10\\workspace
local_launch.bat
popd
```

2. Or specify a different local folder as an argument:

```powershell
local_launch.bat D:\\workspace_real
```

What it does
------------

- Copies the project files from the network share into the local target folder (default `D:\workspace_real`) using `robocopy` while excluding `.venv`, `dataset*`, `models`, `runs`, and `node_modules`.
- Creates a local `.venv` (using `py -3 -m venv` or `python -m venv` fallback).
- Upgrades `pip` and installs `requirements.txt` if present.
- Runs `streamlit` inside the local `.venv`.

Why use this
------------

- Running from a local disk avoids Windows UNC / Python C-extension / file-lock issues.
- Ensures the app uses a clean local `venv` and consistent Python binary.

Notes
-----

- The script uses `robocopy` which is available on Windows. If `robocopy` is not present, copy files manually to the target.
- The first run may take time to install dependencies.
- If you prefer an ephemeral copy, I can add a temp-folder mode that deletes the copy after stopping the server.

Production-grade launcher
-------------------------

I added a more robust launcher `launcher_local.py` and a batch wrapper `run_prod_launcher.bat`.

Usage (recommended):

```powershell
pushd \\\\192.168.1.10\\workspace
run_prod_launcher.bat D:\\workspace
popd
```

What it does:
- Mirror-copy project to the local target using `robocopy` (Windows)
- Create local `.venv` if missing and install `requirements.txt`
- Kill any process listening on port 8501
- Start Streamlit with the venv Python and tail the logs
- Open the browser when the server is ready

If you want extra features (incremental sync, better process management, auto-retry, logging rotation), tell me "build launcher xịn" and I'll implement them.

Converting a Windows shortcut to a real folder
----------------------------------------------

If you see `workspace` on `D:` is a Windows shortcut (a `.lnk`) pointing to the network share, you can convert it to a real local folder using the included helper:

PowerShell / CMD on the local machine:

```powershell
pushd D:\
convert_shortcut_to_folder.bat D:\workspace
popd
```

The script will:
- resolve the `.lnk` target
- copy the target contents to a safe temporary folder using `robocopy`
- prompt for confirmation
- replace the `.lnk` with a real folder containing the copied files

Use this only if you want `D:\workspace` to be a real local folder rather than a shortcut. The script prompts before destructive changes.
