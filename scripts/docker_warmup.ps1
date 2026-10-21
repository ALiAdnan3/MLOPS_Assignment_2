# Task 6.1 - Docker warm-up with a ready-made image (nginx). Pauses at each screenshot point.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\docker_warmup.ps1   (add -NoPause to run straight through)
param([switch]$NoPause)
$ErrorActionPreference = "Continue"
$env:PATH = "$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin;$env:PATH"
$Site = "$env:TEMP\warmup_site_23L0839"

function Step($t) { Write-Host "`n==================== $t ====================" -ForegroundColor Cyan }
function Run($c) { Write-Host "PS> $c" -ForegroundColor Yellow; Invoke-Expression $c }
function Shot($w) { Write-Host "`n>>> SCREENSHOT NOW: $w" -ForegroundColor Magenta; if (-not $NoPause) { Read-Host "Press Enter" | Out-Null } }

Write-Host "User: $env:USERNAME   Time: $(Get-Date)   Project: 23L0839_A2_Ops (Docker warm-up)" -ForegroundColor Green
New-Item -ItemType Directory -Force $Site | Out-Null
Set-Content "$Site\index.html" "<h1>23L0839_A2_Ops - served from a mounted volume</h1>" -Encoding ascii
docker rm -f warmup-nginx 2>$null | Out-Null

Step "1. Run a container from a ready-made image, map a port, mount a volume"
Run "docker pull nginx:alpine"
Run "docker run -d --name warmup-nginx -p 8081:80 -v `"${Site}:/usr/share/nginx/html:ro`" nginx:alpine"
Run "docker ps"
Start-Sleep 2
Run "(Invoke-WebRequest http://localhost:8081 -UseBasicParsing).Content"
Shot "docker run with -p 8081:80 and -v, docker ps, page served from the mounted folder"

Step "2. Read its logs"
Run "docker logs warmup-nginx --tail 15"
Step "3. Open a shell inside it"
Run "docker exec warmup-nginx sh -c 'whoami; hostname; cat /etc/os-release | head -2; ls /usr/share/nginx/html'"
Shot "docker logs (shows the GET request) and docker exec shell commands"

Step "4. List and remove containers and images"
Run "docker ps -a"
Run "docker stop warmup-nginx"
Run "docker rm warmup-nginx"
Run "docker ps -a"
Run "docker images"
Run "docker rmi nginx:alpine"
Run "docker images"
Shot "container stopped + removed, image removed"
