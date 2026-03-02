@echo off
REM Wrapper to run the production-grade launcher using the py launcher
if "%~1"=="" (
  set "TARGET=D:\workspace"
) else (
  set "TARGET=%~1"
)

set "RUNNER="

if exist "%~dp0.venv\Scripts\python.exe" (
  set "RUNNER=%~dp0.venv\Scripts\python.exe"
)

if "%RUNNER%"=="" (
  where py >nul 2>nul
  if %ERRORLEVEL%==0 set "RUNNER=py"
)

if "%RUNNER%"=="" (
  where python >nul 2>nul
  if %ERRORLEVEL%==0 set "RUNNER=python"
)

if "%RUNNER%"=="" (
  echo No Python runner found. Install Python or create .venv first.
  exit /b 1
)

echo Using launcher runner: %RUNNER%
%RUNNER% launcher_local.py --target "%TARGET%"

exit /b %ERRORLEVEL%
