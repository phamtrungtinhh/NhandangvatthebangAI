@echo off
REM Double-click helper: activate existing .venv and install requirements, then npm start
cd /d %~dp0
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0bootstrap_venv.ps1"
if exist ".venv\Scripts\Activate.ps1" (
  echo Activating venv and starting app...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "& '%~dp0.venv\Scripts\Activate.ps1'; npm start"
) else (
  echo .venv not present or activation script missing. Run bootstrap_venv.ps1 first.
)
pause
