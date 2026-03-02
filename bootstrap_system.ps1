<#
Bootstrap system Python: install requirements into user site-packages
Usage:
  Open PowerShell and run (may require internet):
    .\bootstrap_system.ps1
#>
$ErrorActionPreference = 'Stop'

Write-Host "Checking for python..."
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
  Write-Host "Python not found in PATH. Please install Python 3 and retry." -ForegroundColor Red
  exit 1
}

if (-not (Test-Path -LiteralPath 'requirements.txt')) {
  Write-Host "requirements.txt not found. Installing streamlit into --user..." -ForegroundColor Yellow
  python -m pip install --user streamlit
} else {
  Write-Host "Installing requirements into --user site packages..."
  python -m pip install --user -r requirements.txt
}

Write-Host "Bootstrap system Python complete. Run: npm start" -ForegroundColor Green
