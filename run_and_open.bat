@echo off
setlocal EnableExtensions
REM Always run from this script directory
cd /d %~dp0

set "VENV_PY=.venv\Scripts\python.exe"
set "VENV_PYW=.venv\Scripts\pythonw.exe"

if exist "%VENV_PY%" goto RUN_VENV_PY
if exist "%VENV_PYW%" goto RUN_VENV_PYW

where py >nul 2>nul
if %ERRORLEVEL%==0 goto RUN_PY

where python >nul 2>nul
if %ERRORLEVEL%==0 goto RUN_PYTHON

echo No Python runner found. Please install Python or create .venv.
exit /b 1

:RUN_VENV_PYW
echo Starting streamlit with venv pythonw...
"%VENV_PYW%" run_and_open.py --no-restart %*
exit /b %ERRORLEVEL%

:RUN_VENV_PY
echo Starting streamlit with venv python...
"%VENV_PY%" run_and_open.py --no-restart %*
exit /b %ERRORLEVEL%

:RUN_PY
echo Starting streamlit with py launcher...
py -3 run_and_open.py --no-restart %*
exit /b %ERRORLEVEL%

:RUN_PYTHON
echo Starting streamlit with system python...
python run_and_open.py --no-restart %*
exit /b %ERRORLEVEL%
