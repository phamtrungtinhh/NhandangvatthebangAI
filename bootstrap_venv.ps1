<#
Bootstrap venv: activate existing .venv and install requirements
Usage:
  Open PowerShell in project root and run:
    .\bootstrap_venv.ps1
#>
$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath '.venv')) {
  Write-Host ".venv not found in project root." -ForegroundColor Yellow
  Write-Host "To create a new venv, run: python -m venv .venv" -ForegroundColor Cyan
  exit 1
}

$venvActivate = Join-Path $PWD '.venv\Scripts\Activate.ps1'
if (-not (Test-Path -LiteralPath $venvActivate)) {
  Write-Host "Activate script not found: $venvActivate" -ForegroundColor Red
  exit 1
}

Write-Host "Activating venv..."
& $venvActivate
if ($LASTEXITCODE -ne 0) { Write-Host "Failed to activate venv." -ForegroundColor Red; exit 2 }

if (-not (Test-Path -LiteralPath 'requirements.txt')) {
  Write-Host "requirements.txt not found. Installing streamlit only..." -ForegroundColor Yellow
  pip install streamlit
} else {
  Write-Host "Installing requirements from requirements.txt..."
  pip install -r requirements.txt
}

Write-Host "Bootstrap venv complete. Run: npm start" -ForegroundColor Green
