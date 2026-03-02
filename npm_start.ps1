$ErrorActionPreference = 'Stop'

$projectRoot = $env:npm_config_local_prefix
if ([string]::IsNullOrWhiteSpace($projectRoot)) {
  $projectRoot = Split-Path -Parent $PSCommandPath
}

if (-not (Test-Path -LiteralPath $projectRoot)) {
  Write-Host "Không tìm thấy thư mục dự án: $projectRoot" -ForegroundColor Red
  exit 1
}

Push-Location -LiteralPath $projectRoot
try {
  $venvPy = Join-Path $projectRoot '.venv\Scripts\python.exe'
  if (Test-Path -LiteralPath $venvPy) {
    & $venvPy .\run_and_open.py --no-restart
    exit $LASTEXITCODE
  }

  if (Get-Command python -ErrorAction SilentlyContinue) {
    & python .\run_and_open.py --no-restart
    exit $LASTEXITCODE
  }

  Write-Host "Không tìm thấy Python để chạy run_and_open.py (thiếu .venv\\Scripts\\python.exe và lệnh python)." -ForegroundColor Red
  exit 2
}
finally {
  Pop-Location
}
