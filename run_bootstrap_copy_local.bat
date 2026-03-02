@echo off
REM Double-click helper: copy project to D:\workspace, create venv and install deps, then start from local
cd /d %~dp0
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0bootstrap_copy_local.ps1"
echo After copy completes, run from local with:
echo   cd /d D:\workspace
echo   & .\.venv\Scripts\Activate.ps1
echo   npm start
pause
