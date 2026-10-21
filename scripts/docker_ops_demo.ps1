# Task 6.5 - operate the Compose stack and troubleshoot a deliberate failure (wrong DB password).
# Usage: powershell -ExecutionPolicy Bypass -File scripts\docker_ops_demo.ps1   (add -NoPause to run straight through)
param([switch]$NoPause)
$ErrorActionPreference = "Continue"
$env:PATH = "$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin;$env:PATH"
Set-Location (Split-Path -Parent $PSScriptRoot)

function Step($t) { Write-Host "`n==================== $t ====================" -ForegroundColor Cyan }
function Run($c) { Write-Host "PS> $c" -ForegroundColor Yellow; Invoke-Expression $c }
function Shot($w) { Write-Host "`n>>> SCREENSHOT NOW: $w" -ForegroundColor Magenta; if (-not $NoPause) { Read-Host "Press Enter" | Out-Null } }

Write-Host "User: $env:USERNAME   Time: $(Get-Date)   Project: 23L0839_A2_Ops (Compose operations)" -ForegroundColor Green

Step "1. Start / status / stop / start / restart"
Run "docker compose up -d"
Run "docker compose ps"
Run "docker compose stop"
Run "docker compose ps -a"
Run "docker compose start"
Run "docker compose restart web"
Start-Sleep 25
Run "docker compose ps"
Shot "up -d, stop, start, restart and ps showing both services (healthy)"

Step "2. Logs and a shell inside each service"
Run "docker compose logs --tail 8 db"
Run "docker compose logs --tail 8 web"
Run "docker compose exec web sh -c 'whoami; python --version; ls /app /app/models'"
Run "docker compose exec db sh -c 'psql -U `$POSTGRES_USER -d `$POSTGRES_DB -c \`"select id, created_at, label, round(confidence::numeric,3) as conf, model_version from predictions order by id desc limit 5;\`"'"
Shot "logs of both services, shell in web (non-root user) and psql inside db showing the prediction table"

Step "3. Break it on purpose: web gets a WRONG database password"
$env:POSTGRES_PASSWORD = "wrong-password"      # shell env overrides .env for compose interpolation
Run "docker compose up -d web"
Start-Sleep 20
Run "docker compose ps"
Run "docker compose logs --tail 25 web"
Shot "FAILURE: web is restarting; the log ends with 'password authentication failed for user'"

Step "4. Diagnose from logs only"
Run "docker compose logs web 2>&1 | Select-String -Pattern 'OperationalError|authentication failed' | Select-Object -First 3"
Run "docker compose logs db 2>&1 | Select-String -Pattern 'password authentication failed|FATAL' | Select-Object -Last 3"
Shot "diagnosis: web log shows psycopg2 OperationalError, db log shows FATAL password authentication failed"

Step "5. Fix: remove the wrong value so the password from .env is used again, recreate web"
Remove-Item Env:\POSTGRES_PASSWORD
Run "docker compose up -d web"
Start-Sleep 30
Run "docker compose ps"
Run "(Invoke-WebRequest http://localhost:8000/health -UseBasicParsing).Content"
Shot "FIXED: both services healthy, /health returns database ok"
