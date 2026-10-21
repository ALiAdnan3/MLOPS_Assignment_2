# VisionGuard – screenshot guide (run in this order)

Every 📸 number matches a yellow box in `reports\23L0839_VisionGuard.docx`.
Rules for every screenshot: show the **terminal prompt / browser address bar** and the **Windows clock**.

**Start:** open VS Code in `D:\LOPS assignment2` → Terminal → New Terminal (PowerShell) and run once:
```powershell
cd "D:\LOPS assignment2"
.venv\Scripts\activate
$env:PATH = "$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin;$env:PATH"
```
Make sure Docker Desktop is running (whale icon in the taskbar).

---

## Step 0 – one click on DagsHub
https://dagshub.com/ALiAdnan3/23L0839_A2_Ops/settings → bottom of the page → **Make public**.

## Task 1
| 📸 | Do this |
|---|---|
| 1 | `python scripts\check_env.py` |
| 2–5 | `powershell -ExecutionPolicy Bypass -File scripts\dvc_toy_lab.ps1` → it **pauses 4 times**: screenshot, press Enter, repeat |
| 6 | Browser: https://dagshub.com/ALiAdnan3/23L0839_A2_Ops_dvc_toy (commits + students.csv) |
| 7 | `type data\raw.dvc; dvc status -c` |
| 14 | Browser: https://dagshub.com/ALiAdnan3/23L0839_A2_Ops → click folder `data` → `raw` (DVC-tracked images) |

## Task 2
| 📸 | Do this |
|---|---|
| 61 | `python -c "from visionguard.models import build_baseline_cnn; build_baseline_cnn().summary()"` |

(All other Task 2 figures are already inside the report.)

## Task 3
| 📸 | Do this |
|---|---|
| 8–10 | `mlflow ui --backend-store-uri sqlite:///mlflow_local.db` → open http://127.0.0.1:5000 → top-left switch to **Model training** → experiment **23L0839_A2_Ops-toy-fashion-mnist**. 📸8 run list · 📸9a click `toy_lr_0.001` Overview (params, metrics) · 📸9b Model metrics tab · 📸9c Artifacts tab (`loss_curve.png`, `model`) · 📸10 back, tick both runs → **Compare**. Then Ctrl+C in the terminal |
| 11 | https://dagshub.com/ALiAdnan3/23L0839_A2_Ops (project page, "mirrored from GitHub") |
| 12 | https://dagshub.com/ALiAdnan3/23L0839_A2_Ops.mlflow → Model training → experiment **23L0839_A2_Ops-toy-fashion-mnist** (the two toy runs) |
| 13 | `git check-ignore -v .env; git status --short; type .gitignore` |
| 15 | Same MLflow site → experiment **23L0839_A2_Ops-VisionGuard** → click **R1_mnv2_reference** → Overview (parameters, metrics, tags git_commit / dvc_data_version / seed) |
| 16 | R1 → **Model metrics** tab (per-epoch curves) |
| 17 | Click **R2_mnv2_autolog** → Overview (many `opt_*` params) |
| 18 | Click **R4_mnv2_no_augment** → **Artifacts** (expand `evaluation`, `splits`, `environment`, `keras_file`, `model`) |
| 19 | Run list → tick R1, R3, R4, R5, R6, R7, R8, R9 → **Compare** (params table + val_auc / val_loss charts) |
| 20 | Left menu **Models** → **23L0839_A2_Ops-visionguard** (versions 1–3, aliases @champion, @challenger) |
| 21 | `python scripts\predict_champion.py screenshot_files\real_1.jpg` |

## Task 4 + 5 (web app in LOCAL mode)
First run: `powershell -ExecutionPolicy Bypass -File scripts\set_mode.ps1 local` (wait for `inference_mode = local`).

| 📸 | Do this |
|---|---|
| 22 | `python scripts\inference_demo.py screenshot_files\ai_1.png screenshot_files\real_1.jpg` |
| 27 | Browser http://localhost:8000 (upload page) |
| 28 | Upload `D:\LOPS assignment2\screenshot_files\ai_1.png` → result page |
| 29 | Upload `screenshot_files\real_1.jpg` → result page |
| 30 | Upload `screenshot_files\notes.txt` → "Unsupported file type" (tip: in the file dialog choose **All files** to see it) |
| 31 | Upload `screenshot_files\too_big_6MB.png` → "File too large" |
| 32 | Upload `screenshot_files\fake_image.jpg` → "Not a readable image" |
| 33 | Click **History** in the menu |
| 34 | Click **Health** (http://localhost:8000/health) |

## Task 6
| 📸 | Do this |
|---|---|
| 35–37 | `powershell -ExecutionPolicy Bypass -File scripts\docker_warmup.ps1` → pauses **3 times** |
| 39–43 | `powershell -ExecutionPolicy Bypass -File scripts\docker_ops_demo.ps1` → pauses **5 times** (takes ~3 min) |
| 44a | `docker run --rm --entrypoint id 23l0839_a2_ops-web` |
| 44b | `docker compose ps; Test-NetConnection 127.0.0.1 -Port 5432` (shows `TcpTestSucceeded : False`) |
| 62 | `docker images 23l0839_a2_ops-web; docker history 23l0839_a2_ops-web:latest` |
| 24 | Browser https://github.com/ALiAdnan3/23L0839_A2_Ops_ci_toy/actions (✅ ❌ ✅) |
| 25 | Click the ❌ run → click a red job → expand the red **Run pytest -v** step |
| 26 | Click the last ✅ "Fix the bug" run (both matrix jobs green) |
| 23 | https://github.com/ALiAdnan3/MLOPS_Assignment_2/actions → click the newest ✅ run (both jobs green) |
| 63 | https://github.com/ALiAdnan3/MLOPS_Assignment_2 (README with green CI badge) |

## Task 7 (Azure)
| 📸 | Do this |
|---|---|
| 45 | https://portal.azure.com → search **23L0839_A2_Ops** → open the resource group (list of resources) |
| 46 | Click **acr23l0839a2ops** → Services → **Repositories** → `23l0839-a2-ops-api` (tag v2) |
| 47 | Back → click **vg-23l0839-a2-ops-api** → Overview (Application Url, Running) |
| 48 | Same app → **Containers** (or Application → Containers → Environment variables): API_KEY = secret reference, THRESHOLD, MODEL_VERSION |
| 50–53 | Postman → **Import** → `D:\LOPS assignment2\postman\23L0839_A2_Ops_VisionGuard.postman_collection.json` → click the collection → **Variables** → set `api_key` = the `MODEL_API_KEY=` value from `.env` → Save. Send each request and screenshot the response + **Test Results**: 📸50 Health · 📸51 Predict AI image · 📸52 Predict real image · 📸53 Predict WITHOUT key (401). If Postman says it cannot read the file, click the request → Body → re-select the file (ai_1.png / real_1.jpg in `screenshot_files`) |
| 54 | On your **phone with Wi-Fi OFF**, open https://vg-23l0839-a2-ops-api.gentleglacier-7f267617.eastus.azurecontainerapps.io/health (first load can take ~20 s) |

## Task 8 (web app in REMOTE mode → calls Azure)
First run: `powershell -ExecutionPolicy Bypass -File scripts\set_mode.ps1 remote` (wait for `inference_mode = remote`).
In the Azure portal open **vg-23l0839-a2-ops-api → Monitoring → Log stream** and keep that tab open.

| 📸 | Do this |
|---|---|
| 55 | http://localhost:8000 → upload `screenshot_files\ai_1.png` (result shows *Inference: remote*) |
| 56 | Upload `screenshot_files\ai_2.jpg` |
| 57 | Upload `screenshot_files\real_1.jpg` |
| 58 | Upload `screenshot_files\real_2.jpg` |
| 49 | Switch to the Azure **Log stream** tab: lines `predict file=... label=...` |
| 59 | **History** page (rows with mode `local` and `remote`) |
| 60 | **Health** page (`inference_mode: remote` + `model_api` block) |

---

## Finish the report
1. Open `D:\LOPS assignment2\reports\23L0839_VisionGuard.docx`.
2. For each yellow box: click inside it → Ctrl+V the matching screenshot → delete the yellow text.
3. Right-click the table of contents → **Update Field → Update entire table**.
4. **File → Save As → PDF** → name `23L0839_VisionGuard.pdf`.
5. Submit: the PDF, DagsHub link https://dagshub.com/ALiAdnan3/23L0839_A2_Ops, GitHub link, API URL.
