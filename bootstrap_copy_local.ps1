<#
Copy project local and create fresh venv on D:\workspace (will overwrite target).
Usage (run as user with write access to D:):
  .\bootstrap_copy_local.ps1
#>
param(
  [string]$Target = 'D:\workspace'
)
$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath '\\192.168.1.10\workspace')) {
  Write-Host "Source UNC not found: \\192.168.1.10\workspace" -ForegroundColor Red
  exit 1
}

Write-Host "Copying project to $Target (may take time)..."
# Use robocopy if available
$robocopy = Get-Command robocopy -ErrorAction SilentlyContinue
if ($robocopy) {
  robocopy "\\192.168.1.10\workspace" "$Target" /MIR /Z /MT:8 /ETA
  if ($LASTEXITCODE -ge 8) { Write-Host "Robocopy failed with code $LASTEXITCODE" -ForegroundColor Red; exit 2 }
} else {
  Write-Host "robocopy not available. Use PowerShell copy (may be slower)..."
  Remove-Item -LiteralPath $Target -Recurse -Force -ErrorAction SilentlyContinue
  New-Item -ItemType Directory -Path $Target -Force | Out-Null
  Copy-Item -Path "\\192.168.1.10\workspace\*" -Destination $Target -Recurse -Force
}

Push-Location $Target
try {
  Write-Host "Creating venv in $Target\.venv"
  python -m venv .venv
  & .\.venv\Scripts\Activate.ps1
  if (Test-Path -LiteralPath 'requirements.txt') {
    pip install -r requirements.txt
  } else {
    pip install streamlit
  }
  Write-Host "Copy-and-bootstrap complete. Run: cd /d $Target; & .\.venv\Scripts\Activate.ps1; npm start" -ForegroundColor Green
} finally {
  Pop-Location
}
