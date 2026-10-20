# Task 1.4 – DVC toy lab (run these yourself, take screenshots marked 📸)

Run in **PowerShell** (VS Code terminal). If activation is blocked, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## 0. Prerequisite – DagsHub
1. Sign up at https://dagshub.com with **Sign up with GitHub**.
2. Top-right **+ Create → New Repository**, name it `23L0839_A2_Ops_dvc_toy`, keep it empty (no README). Create.
3. Avatar → **Your Settings → Tokens** → copy your default access token (treat it like a password).

## 1. New empty folder with a small CSV
```powershell
mkdir D:\dvc_toy_23L0839 ; cd D:\dvc_toy_23L0839
& "D:\LOPS assignment2\.venv\Scripts\Activate.ps1"      # reuse our venv (has dvc)
git init -b main                                        # Git will track code + pointer files
dvc init                                                # creates .dvc/ (DVC config + cache dir)
Set-Content students.csv "id,name,score`n1,Ali,90`n2,Sara,85" -Encoding utf8
```

## 2. Track the CSV with DVC – version 1
```powershell
dvc add students.csv          # moves data to .dvc/cache, writes students.csv.dvc, git-ignores students.csv
type students.csv.dvc         # the pointer: md5 + size + path   📸
git add students.csv.dvc .gitignore
git commit -m "v1: students.csv tracked by DVC"
```

## 3. Change it – version 2
```powershell
Add-Content students.csv "3,Omar,77"
dvc add students.csv          # new md5 -> pointer file changes
git add students.csv.dvc
git commit -m "v2: added Omar"
git log --oneline             # 📸 both commits
```

## 4. Return to the older version
```powershell
git checkout HEAD~1 -- students.csv.dvc    # old pointer back
dvc checkout                               # DVC restores the data matching that pointer
type students.csv                          # 📸 only 2 rows again
git checkout main -- students.csv.dvc ; dvc checkout   # back to v2 (3 rows)
```

## 5. Remote storage (DagsHub), push, delete local data, pull back
Replace `<user>` with your DagsHub username and `<token>` with your token.
```powershell
git remote add origin https://dagshub.com/<user>/23L0839_A2_Ops_dvc_toy.git
dvc remote add -d origin s3://dvc
dvc remote modify origin endpointurl https://dagshub.com/<user>/23L0839_A2_Ops_dvc_toy.s3
dvc remote modify --local origin access_key_id <token>       # --local => stored in .dvc/config.local (git-ignored)
dvc remote modify --local origin secret_access_key <token>
git add .dvc/config ; git commit -m "Configure DagsHub DVC remote"
git push -u origin main        # Git username = DagsHub user, password = token
dvc push                       # 📸 uploads the data blobs
Remove-Item students.csv ; Remove-Item -Recurse -Force .dvc\cache   # simulate a fresh machine
dvc pull                       # 📸 data comes back
type students.csv
```
📸 Also screenshot the DagsHub repo page showing the commits and the DVC-tracked file.

## What to write in the report
- **Pointer file (`*.dvc`)**: a few lines of YAML holding the MD5 hash of the data, its size, (for folders) the
  number of files, and the path. Git versions this tiny file; the hash tells DVC exactly which bytes to fetch from cache/remote.
- **Why large data does not belong in Git**: Git stores every version of every file forever in every clone, so a
  500 MB dataset changed 5 times makes every clone ~2.5 GB; Git diffs/compression are designed for text, not binary
  images; GitHub rejects files > 100 MB. DVC keeps data in cheap object storage and only the pointer in Git.
- **`--local` for credentials**: the token is written to `.dvc/config.local`, which DVC git-ignores, so it never reaches the repo.
