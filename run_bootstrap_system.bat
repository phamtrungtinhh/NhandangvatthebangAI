@echo off
REM Double-click helper: install requirements into system/user Python then start app
cd /d %~dp0
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0bootstrap_system.ps1"
echo Starting app using system Python...
powershell -NoProfile -ExecutionPolicy Bypass -Command "cd '%~dp0'; npm start"
pause
