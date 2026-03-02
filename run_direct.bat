@echo off
REM Direct helper: map current UNC via pushd and call run_and_open.bat
SET UNC=\\\\192.168.1.10\\workspace
pushd %UNC% || (
  echo Failed to access %UNC%
  exit /b 1
)
if exist "run_and_open.bat" (
  echo Calling run_and_open.bat in %CD%
  call run_and_open.bat %*
  set rc=%ERRORLEVEL%
  popd
  exit /b %rc%
) else (
  echo run_and_open.bat not found in %CD%
  popd
  exit /b 2
)
