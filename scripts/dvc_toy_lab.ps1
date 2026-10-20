# Task 1.4 toy part - DVC basics in a NEW empty folder, Git + DVC remote on DagsHub.
# Usage (from the project folder):  powershell -ExecutionPolicy Bypass -File scripts\dvc_toy_lab.ps1
# Add -NoPause to run without screenshot pauses.
param([switch]$NoPause)
$ErrorActionPreference = "Continue"   # native tools write progress to stderr; we check exit codes instead
$Project = Split-Path -Parent $PSScriptRoot
$Toy = "D:\dvc_toy_23L0839"
$User = "ALiAdnan3"
$Repo = "23L0839_A2_Ops_dvc_toy"

# read the DagsHub token from the project's .env (never printed)
$Token = (Get-Content "$Project\.env" | Where-Object { $_ -match '^MLFLOW_TRACKING_PASSWORD=' }) -replace '^MLFLOW_TRACKING_PASSWORD=', ''
$Basic = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("${User}:$Token"))
$env:PATH = "$Project\.venv\Scripts;$env:PATH"

function Step($title) { Write-Host "`n==================== $title ====================" -ForegroundColor Cyan }
function Run($cmd) {
    Write-Host "PS $(Get-Location)> $cmd" -ForegroundColor Yellow
    Invoke-Expression $cmd
    if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) { throw "failed: $cmd" }
}
function Shot($what) {
    Write-Host "`n>>> SCREENSHOT NOW: $what" -ForegroundColor Magenta
    if (-not $NoPause) { Read-Host "Press Enter to continue" | Out-Null }
}

Write-Host "User: $env:USERNAME   Time: $(Get-Date)   Project: 23L0839_A2_Ops (DVC toy)" -ForegroundColor Green
if (Test-Path $Toy) { Remove-Item -Recurse -Force $Toy }
New-Item -ItemType Directory $Toy | Out-Null
Set-Location $Toy

Step "1. New empty folder: initialise Git and DVC"
Run "git init -b main -q"
Run "dvc init -q"
Run "Set-Content students.csv 'id,name,score','1,Ali,90','2,Sara,85' -Encoding ascii"
Run "Get-Content students.csv"

Step "2. Track the CSV with DVC and commit version 1"
Run "dvc add students.csv"
Run "Get-Content students.csv.dvc"
Run "Get-Content .gitignore"
Run "git add students.csv.dvc .gitignore"
Run "git commit -q -m 'v1: students.csv tracked by DVC (2 rows)'"
Shot "steps 1-2: git/dvc init, the pointer file students.csv.dvc, first commit"

Step "3. Change the CSV and commit version 2"
Run "Add-Content students.csv '3,Omar,77' -Encoding ascii"
Run "dvc status"
Run "dvc add students.csv"
Run "Get-Content students.csv.dvc"
Run "git add students.csv.dvc"
Run "git commit -q -m 'v2: added Omar (3 rows)'"
Run "git log --oneline"

Step "4. Return to the older version"
Run "git checkout HEAD~1 -- students.csv.dvc"
Run "dvc checkout"
Run "Get-Content students.csv"
Shot "steps 3-4: v2 commit, git log with both versions, v1 restored (2 rows)"
Run "git checkout main -- students.csv.dvc"
Run "dvc checkout"
Run "Get-Content students.csv"

Step "5. Configure DagsHub as remote storage, push Git commits + data"
Run "git remote add origin https://dagshub.com/$User/$Repo.git"
Run "dvc remote add -d origin s3://dvc"
Run "dvc remote modify origin endpointurl https://dagshub.com/$User/$Repo.s3"
Write-Host "PS $(Get-Location)> dvc remote modify --local origin access_key_id / secret_access_key <token hidden>" -ForegroundColor Yellow
dvc remote modify --local origin access_key_id $Token
dvc remote modify --local origin secret_access_key $Token
Run "Get-Content .dvc\config"
Run "git add .dvc\config"
Run "git commit -q -m 'Configure DagsHub DVC remote'"
Write-Host "PS $(Get-Location)> git push --force -u origin main   (auth header hidden)" -ForegroundColor Yellow
git -c "http.extraHeader=Authorization: Basic $Basic" push --force -u origin main
Run "dvc push"
Shot "step 5: dvc remote config + git push + dvc push to DagsHub"

Step "6. Delete local data (and cache) then pull it back"
Run "Remove-Item students.csv"
Run "Remove-Item -Recurse -Force .dvc\cache"
Run "Test-Path students.csv"
Run "dvc pull"
Run "Get-Content students.csv"
Shot "step 6: data deleted, dvc pull brings it back. Then open https://dagshub.com/$User/$Repo and screenshot the repo page (commits + students.csv)"
Set-Location $Project
