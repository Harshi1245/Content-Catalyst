# start_backend.ps1
# Persistent backend starter for local dev. Keeps restarting the backend if it exits unexpectedly
# Usage: run this PowerShell script from project root (where .venv and backend/ are located)

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
# try to activate venv if present
$activate = Join-Path $projectRoot '.venv\Scripts\Activate.ps1'
if (Test-Path $activate) {
    Write-Output "Sourcing virtualenv activate: $activate"
    & $activate
} else {
    # avoid special punctuation that can break some PowerShell parsers; build the message safely
    $warn = "Warning: .venv not found at $activate - ensure your venv is activated before running this script"
    Write-Output $warn
}

$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    $python = 'python' # fallback to PATH
}

$log = Join-Path $projectRoot 'outputs\backend_run.log'
Write-Output "Backend logs: $log"

while ($true) {
    $ts = Get-Date -Format o
    # write timestamped header to the log (use Add-Content to avoid quoting/parsing issues)
    Add-Content -Path $log -Value "$ts Starting backend..."
    # run the backend and append each output line to the log while also echoing to console
    & $python backend/run_server.py 2>&1 | ForEach-Object { Add-Content -Path $log -Value $_; Write-Host $_ }
    $ts2 = Get-Date -Format o
    Add-Content -Path $log -Value "$ts2 Backend exited; will restart in 2 seconds..."
    Start-Sleep -Seconds 2
}
