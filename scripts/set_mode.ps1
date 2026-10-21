# Switch the web app between local inference (model inside the container) and remote inference (Azure API).
# Usage: powershell -ExecutionPolicy Bypass -File scripts\set_mode.ps1 local|remote
param([ValidateSet("local", "remote")][string]$Mode = "local")
$env:PATH = "$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin;$env:PATH"
Set-Location (Split-Path -Parent $PSScriptRoot)
(Get-Content .env) -replace '^INFERENCE_MODE=.*', "INFERENCE_MODE=$Mode" | Set-Content .env -Encoding ascii
docker compose up -d
Write-Host "Waiting for the web app to become healthy..." -ForegroundColor Cyan
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep 3
    try { $h = Invoke-RestMethod http://localhost:8000/health -TimeoutSec 60; break } catch { }
}
docker compose ps
Write-Host "inference_mode = $($h.inference_mode)  status = $($h.status)" -ForegroundColor Green
