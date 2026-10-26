## Introduction
**VisionGuard** is a web service that tells whether an uploaded image is AI-generated or real. I trained TensorFlow/Keras models, tracked experiments with MLflow on DagsHub, versioned the data with DVC, served the model with Flask + PostgreSQL in Docker Compose, added GitHub Actions CI, and deployed the model API on Microsoft Azure.

- **Dataset:** *AI Generated Images vs Real Images* (Kaggle) – folder `AiArtData` = AI, `RealArt` = real.
- **Project name everywhere:** 23L0839_A2_Ops.
- **Final model:** `23L0839_A2_Ops-visionguard` v2 `@champion` (MobileNetV2, threshold 0.80, test ROC-AUC 0.86).

## Task 1 – Data Understanding and Versioning

### 1.1 Environment
- Created a virtual environment (`python -m venv .venv`) so the project's packages cannot clash with other projects; versions pinned in `requirements.txt`.
- `scripts/check_env.py` prints the versions: Python 3.13.15, TensorFlow 2.21.0, Keras 3.15.1, MLflow 3.17.1, DVC 3.67.1, Flask 3.1.3.
- **No GPU visible:** TensorFlow > 2.10 has no GPU support on native Windows and my laptop has no NVIDIA GPU, so I trained on the CPU (fine for ~1 000 images).

![label=env; Exact library versions (scripts/check_env.py)](screenshots/Task1_img_1.png)

### 1.2 Dataset audit
- `src/audit_dataset.py` records class, file size, width, height, colour mode, format and SHA-256 for every file. Each file is opened with Pillow (`verify()` + `load()`); a failure = corrupt. Same SHA-256 = exact duplicate.

| Class | Files | Corrupt | Duplicates | Width min/median/max | Height min/median/max | Size KB min/median/max |
|---|---|---|---|---|---|---|
| AI | 539 | 0 | 1 | 225 / 1024 / 6000 | 135 / 800 / 4001 | 11.5 / 195 / 5494 |
| Real | 436 | 0 | 0 | 183 / 920 / 8495 | 158 / 720 / 7091 | 11.1 / 141 / 9956 |

- **Findings:** 20 file extensions are wrong (e.g. `.png` files that are really JPEG) → I use the format detected from the content. Modes include RGBA, palette and greyscale → everything is converted to RGB. Possible shortcuts: 20 % of AI images are PNG (real: 8 %), 14 % of AI images are exactly 512/1024/2048 px wide (real: 3 %).
- **Splits:** duplicate removed, stratified 70/15/15 with seed 42 → train 681, validation 146, test 147 (`data/splits/*.csv`, committed to Git).

### 1.3 Visual EDA

![label=grid; Eight random images per class](reports/figures/sample_grid.png)

{fig:grid}: AI images are polished, saturated portraits and fantasy scenes. The real class also contains human paintings and digital art, so the task is really "AI vs human-made", which is harder than "AI vs photo".

![label=hw; Width per class](reports/figures/hist_width.png)

{fig:hw}: AI widths pile up at 512, 1024 and ~2048 px (typical generator sizes); real widths are spread out with a long tail of large camera photos.

![label=hh; Height per class](reports/figures/hist_height.png)

{fig:hh}: the same pattern for height – AI heights sit on a few fixed values, real heights vary much more.

![label=hs; File size per class (log scale)](reports/figures/hist_file_size.png)

{fig:hs}: both classes overlap between 100 KB and 1 MB, but the real class has a cluster of tiny 10–20 KB thumbnails – a weak signal I do not want the model to rely on.

### 1.4 DVC
**Toy part** (`scripts/dvc_toy_lab.ps1`, new folder `D:\dvc_toy_23L0839`):
1. `git init`, `dvc init`, small `students.csv`, `dvc add` → DVC moves the file to its cache, writes the pointer `students.csv.dvc` and git-ignores the CSV.
2. Changed the CSV, `dvc add` again (new MD5), second commit.
3. Back to the old version: `git checkout HEAD~1 -- students.csv.dvc` + `dvc checkout`.
4. DagsHub remote (`dvc remote add -d origin s3://dvc`, token stored with `--local` so it is never committed), `git push` + `dvc push`.
5. Deleted the CSV and the whole cache, `dvc pull` → the file came back.

![label=dvc1; Toy: git init, dvc init, small CSV](screenshots/Task1_img_2.png)

![label=dvc2; Toy: pointer file and first commit](screenshots/Task1_img_3.png)

![label=dvc3; Toy: second version, git log, back to version 1](screenshots/Task1_img_4.png)

![label=dvc4; Toy: DagsHub remote configured and git push](screenshots/Task1_img_5.png)

![label=dvc5; Toy: data and cache deleted, dvc pull restores it](screenshots/Task1_img_6.png)

![label=dvc6; Toy repository on DagsHub](screenshots/Task1_img_7.png)

**Apply part:** `dvc add data/raw` → `data/raw.dvc` committed to Git; `dvc push` uploaded all 975 images (480 MB) to DagsHub; `dvc status -c` shows cache and remote in sync. Every training run logs this hash as the MLflow tag `dvc_data_version`.

![label=dvc7; Real dataset: pointer file and dvc status -c](screenshots/Task1_img_8.png)

- **Pointer file:** MD5 of the data (`.dir` = hash of the list of all file hashes), total size, number of files and path – tiny text that Git can version; DVC uses the hash to fetch the exact bytes.
- **Why not Git:** Git keeps every version in every clone forever (a 480 MB folder changed a few times = GBs per clone), diffs are for text, and GitHub rejects files > 100 MB.

## Task 2 – Model Development with TensorFlow/Keras

### 2.1 Input pipeline (`visionguard/data.py`)
1. **Read split list** → (path, label) dataset – every run sees the same images.
2. **Decode + resize** with Pillow (`load_image` in `visionguard/preprocess.py`): first frame, convert to RGB, resize to 224×224. Pillow is used because TensorFlow cannot read WEBP; **the Flask app imports the same function**, so serving preprocessing is identical.
3. **Cache** decoded images in memory → decode only once.
4. **Shuffle** (train only, seed 42) → 5. **Batch** 32 → 6. **Augment** (train only): flip, rotation ±18°, zoom 10 % – mild, so the fine AI texture details survive → 7. **Prefetch**.
- **Image size 224:** the input size MobileNetV2/EfficientNetB0 were pre-trained on, and cheap enough for CPU training.
- **Aspect ratio:** squashed to a square (not preserved). Cropping would cut content and padding would leak the original shape; both classes have the same median ratio (1.5), so squashing affects both equally.

![label=aug; One training batch before (top) and after (bottom) augmentation](reports/figures/augmented_batch.png)

### 2.2 Baseline CNN
Rescaling(1/255) → 4 × [Conv2D 3×3 → BatchNorm → ReLU → MaxPool] (32/64/128/256 filters) → GlobalAveragePooling → Dropout 0.4 → Dense 64 → Dense 1 sigmoid (406 369 parameters).
- **Conv2D** learns edges → textures → parts; **BatchNorm** stabilises training; **MaxPool** halves the size and adds shift tolerance; **GlobalAveragePooling** keeps the head tiny (256 vs 50 176 inputs) → less overfitting; **Dropout** stops reliance on a few features; **sigmoid** outputs P(AI).
- **Loss:** binary cross-entropy (the right loss for one sigmoid output). **Optimizer:** Adam, lr 1e-3 (adaptive, little tuning). **Metrics:** accuracy, precision, recall (the two errors matter differently) and AUC (threshold-free).

![label=cnnsum; model.summary() of the baseline CNN](screenshots/Task2_img_1.png)

CNN_TABLE

- **Problem & fix:** the first run predicted "real" for every validation image (val accuracy stuck at 65/146). BatchNorm uses moving averages at validation time and with momentum 0.99 and only 22 batches/epoch they had not caught up. Momentum 0.9 + more patience fixed it → best epoch 16, test accuracy 0.640, ROC-AUC 0.759.

![label=cnncurves; Baseline CNN curves (best epoch 16 in red)](reports/figures/baseline_cnn_training_curves.png)

### 2.3 Transfer learning (MobileNetV2)
- `MobileNetV2(include_top=False, weights="imagenet")` + my head (GlobalAveragePooling → Dropout 0.3 → Dense 1 sigmoid).
- **Stage 1:** backbone frozen, lr 1e-3, only the head is trained (1 281 params) – protects the pretrained filters from the random head's large gradients.
- **Stage 2:** unfroze only the last block (`block_16_*`, `Conv_1`; BatchNorm stays frozen), lr 10× smaller (1e-4) → 880 321 trainable params.
- **Preprocessing:** MobileNetV2 expects pixels in [-1, 1] (x/127.5 − 1); it is a `Rescaling` layer inside the model so serving can never forget it.
- **Result:** val AUC 0.82 → 0.87 after fine-tuning; test accuracy 0.789, precision 0.891, ROC-AUC 0.872.

![label=tlcurves; MobileNetV2: stage 1 left of the dotted line, fine-tuning right (best epoch 24)](reports/figures/mobilenetv2_tl_training_curves.png)

### 2.4 Training discipline
Seeds fixed (`set_random_seed(42)`); EarlyStopping and ModelCheckpoint on validation loss; ReduceLROnPlateau. Best epochs: CNN 16, MobileNetV2 24, final model R4 **26**.

![label=r4curves; Final model R4: loss and accuracy with the best epoch (26) marked](reports/figures/R4_mnv2_no_augment_training_curves.png)

### 2.5 Evaluation
- **Worse error:** calling a **real photo fake** – it falsely accuses a real photographer or artist. So I want high precision for the "AI" class.
- **Threshold:** chosen on the **validation** set by maximising F0.5 (precision weighted ×2); the test set was used **once**.

| Test set | Baseline CNN | MobileNetV2 | Final R4 |
|---|---|---|---|
| Threshold (from validation) | 0.67 | 0.72 | 0.80 |
| Accuracy | 0.640 | 0.789 | 0.714 |
| Precision | 0.726 | 0.891 | 0.868 |
| Recall | 0.556 | 0.704 | 0.568 |
| F1 | 0.629 | 0.786 | 0.687 |
| ROC-AUC | 0.759 | 0.872 | 0.860 |

![label=cm; Test confusion matrices (CNN, MobileNetV2, R4)](reports/figures/eval_confusion_matrix_3models.png)

![label=roc; Test ROC curves (CNN, MobileNetV2, R4)](reports/figures/eval_roc_curve_3models.png)

### 2.6 Error analysis
All 10 most confident errors are AI images called "real" (the error I chose to accept).

![label=errors; The 10 most confident test errors of R4](reports/figures/error_analysis_champion.png)

Hypotheses: (1) screenshot of AI art in Van Gogh style – looks painted; (2) flat pop-art banner – no photo texture; (3) HDR-like landscape; (4) half real photo with an AI effect – questionable label; (5) photorealistic Midjourney landscape; (6) realistic face dominates a cartoon; (7) Eldagsen's prize-winning AI "photo" – fooled human judges too; (8) brush-stroke painting style; (9) digital painting like stock art; (10) 49 KB wallpaper – compression hides generator traces.

### Bonus – Grad-CAM

![label=gradcam; Grad-CAM for 4 correct and 4 wrong predictions](reports/figures/gradcam_champion.png)

The model looks at the **main subject** (castle, faces, canvas), not tiny pixel artefacts. The real painting *Girl with a Pearl Earring* is called AI because of its smooth, perfect face – the "AI look". This explains why it fails on paintings and photorealistic fakes.

### 2.7 Keras Tuner
**Toy:** Fashion-MNIST, Dense(units), RandomSearch with 10 trials over units and learning rate.

TOY_TABLE

Best: 256 units, lr 0.0071 (val accuracy 0.860). Learning rates below 2e-4 were always worst.

**Apply:** two-stage MobileNetV2 inside each trial; 10 trials over fine-tune lr, dropout and number of unfrozen layers; objective val AUC.

APPLY_TABLE

- **Mattered most:** the **fine-tune learning rate together with the number of unfrozen layers** – with 40 unfrozen layers lr 2.3e-4 was best (0.893) and lr 6.7e-4 worst (0.703); 0 unfrozen layers stayed at 0.78. Dropout had no clear effect.
- **Overfitting the validation set:** all trials are scored on the same 146 images and I pick the maximum, so part of the winner's score is luck (identical runs differed by 0.035 AUC). Only an untouched test set gives an honest estimate.

## Task 3 – Experiment Tracking with MLflow and DagsHub

### 3.1 MLflow locally
`src/mlflow_toy.py --target local`: tiny Dense net on Fashion-MNIST, two runs (lr 0.001 and 0.01), logging parameters, per-epoch metrics, a loss plot and the model; UI: `mlflow ui --backend-store-uri sqlite:///mlflow_local.db`.

![label=mlruns; Local MLflow UI: the two toy runs](screenshots/extra_mlflow_local_runs.png)

![label=mlrun; Toy run lr 0.001: metrics, parameters, tags and the logged model](screenshots/extra_mlflow_local_run.png)

![label=mlcompare; Comparing the two learning rates in the local MLflow UI](screenshots/extra_mlflow_local_compare.png)

- **Experiment** – a group of runs for one problem. **Run** – one execution of the training code. **Parameter** – a setting chosen before the run (lr). **Metric** – a number measured during/after the run (loss, AUC). **Artifact** – a file produced by the run (plot, model). **Tag** – extra labels (git commit, data version, seed).
- **Problem & fix:** Keras 3 `compile()` takes `loss_weights` as 3rd positional argument → crash; fixed with keyword arguments.

### 3.2 DagsHub setup
- Created **ALiAdnan3/23L0839_A2_Ops** on DagsHub, connected to (mirroring) my GitHub repo. Tracking address: `https://dagshub.com/ALiAdnan3/23L0839_A2_Ops.mlflow`.
- Address + token are in `.env` (`MLFLOW_TRACKING_URI/USERNAME/PASSWORD`), which MLflow reads itself – never in code; `.env` is git-ignored. Repeated the toy run on DagsHub.

![label=dagshub; DagsHub project connected to GitHub](screenshots/Task1_img_9.png)

![label=toyruns; Toy runs on the DagsHub MLflow server](screenshots/Task2_img_3.png)

![label=gitignore; .gitignore: .env and the DVC credentials are never committed](screenshots/Task2_img_4.png)

### 3.3 Instrumenting VisionGuard training
- **Manual** (`--mlflow manual`): a Keras callback logs train/val metrics + learning rate every epoch; I also log all hyper-parameters, test metrics, threshold, best epoch and tags `git_commit`, `dvc_data_version`, `seed`.
- **Autolog** (`--mlflow autolog`): records `fit()` arguments, optimizer and callback settings, per-epoch metrics, model summary and the model automatically.
- **Captures / misses:** autolog misses my design choices (augmentation, image size, backbone), everything outside `fit()` (test metrics, threshold) and the second training stage (MLflow rejected changing already-logged parameters). Manual logging needs more code but records what matters. Best: combine both.
- **Problem & fix:** autolog crashed with `TBNotInstalledError` → installed `tensorboard`.

![label=runs; VisionGuard runs on DagsHub (R2 = autolog, others = manual)](screenshots/Task2_img_2.png)

![label=autolog; Autolog run R2 with automatic and manually added metrics](screenshots/Task2_img_5.png)

### 3.4 Artifacts
Each run stores the confusion matrix, ROC curve, training curves, split lists, `pip freeze` package list, the `.keras` file and the MLflow model (see the logged model in {fig:mlrun} and the "model" column in {fig:runs}).

### 3.5 Experiment comparison
Each run changes **one** thing compared to R1 (`scripts/run_experiments.sh`).

![label=compare; Compare view of the run parameters on DagsHub](screenshots/Task2_img_6.png)

| Run | Change vs R1 | Val loss | Val AUC | Val F0.5 | Test AUC |
|---|---|---|---|---|---|
| R1 | reference (MobileNetV2, lr 1e-3, light aug, 224 px) | 0.517 | 0.819 | 0.797 | 0.834 |
| R2 | autologging | 0.494 | 0.854 | 0.840 | 0.863 |
| R3 | lr 3e-4 | 0.532 | 0.815 | 0.825 | 0.848 |
| **R4** | **no augmentation** | **0.445** | **0.875** | **0.853** | 0.860 |
| R5 | strong augmentation | 0.452 | 0.859 | 0.842 | 0.846 |
| R6 | frozen backbone only | 0.589 | 0.767 | 0.761 | 0.808 |
| R7 | image size 160 | 0.500 | 0.840 | 0.821 | 0.837 |
| R8 | EfficientNetB0 | 0.461 | 0.865 | 0.831 | 0.884 |
| R9 | baseline CNN | 0.552 | 0.774 | 0.753 | 0.742 |

- **Choice: R4** – best on all three **validation** metrics (AUC, F0.5, loss). I did not pick by test AUC (R8), because that would turn the test set into a second validation set.
- **Caveat:** R1 and R2 are identical yet differ by 0.035 AUC, so R4/R5/R8 are practically tied. Clear findings: fine-tuning > frozen, transfer learning > CNN from scratch.

### 3.6 Model registry
Registered `23L0839_A2_Ops-visionguard` v1 (R1), **v2 (R4) = `@champion`**, v3 (R8) = `@challenger`. `scripts/predict_champion.py` loads `models:/23L0839_A2_Ops-visionguard@champion` and classifies one image.

![label=registry; Model registry: three versions with aliases](screenshots/Task2_img_7.png)

![label=champion; Loading the champion by alias and classifying one image](screenshots/Task2_img_8.png)

- **Problem & fix:** MLflow 3 "LoggedModels" could not be registered via `runs:/…/model` on DagsHub → registered each run's own `models:/m-…` URI.
- **Model to server:** `python -m src.registry fetch` downloads the champion **once** to `models/champion.keras`; the app never calls DagsHub per request.

## Task 4 – Flask Web Application

### 4.1 Inference module (`visionguard/inference.py`)
- Loads the model **once** at start-up (+ one warm-up call), then each request only predicts. Uses the same `load_image()` as training. Returns label, confidence of the chosen class, P(AI), threshold and model version.
- **Per-request loading is unacceptable:** loading takes 1–8 s, a prediction 0.1–0.5 s.
- **Preprocessing must match:** a different resize, colour order or scaling gives confident but wrong answers.

![label=demo; Model loaded once, two known images classified](screenshots/Task4_img_1.png)

### 4.2 Routes and validation
Routes: `/` upload, `/predict` result, `/history`, `/health`. Rejects wrong type (**415**), > 5 MB (**413**) and undecodable files (**422**); names cleaned with `secure_filename`; the upload stays in memory and is **never written to disk**; friendly error pages.

![label=e415; Wrong file type (415)](screenshots/extra_error_415.png)

![label=e413; File over the size limit (413)](screenshots/Task4_img_4.png)

![label=e422; Image extension but not a readable image (422)](screenshots/extra_error_422.png)

### 4.3 HTML and CSS
One base template with menu **Upload · History · Health**; the result page shows verdict, confidence bar, P(AI), threshold, model, inference mode, latency, DB record and a caveat.

![label=upload; Upload page](screenshots/extra_upload_page.png)

![label=resai; Result: AI-generated image](screenshots/Task4_img_2.png)

![label=resreal; Result: real image](screenshots/Task4_img_3.png)

### 4.4 Configuration and production server
- All settings come from environment variables (model path, threshold, database URL, inference mode, API URL/key) – same code everywhere, no secrets in code.
- Served by **waitress** with **debug off**: debug mode exposes source code and an interactive console that can run code on the server, and auto-reloads the model.

![label=health; /health: database ok, debug false](screenshots/Task4_img_6.png)

## Task 5 – Database Trail
- Every prediction inserts one row into `predictions` (SQLAlchemy, PostgreSQL 16 in Docker, SQLite for tests): **when** (`created_at`), **what** (`label`, `prob_ai`, `threshold`), **how confidently** (`confidence`), **which model** (`model_name`, `model_version`, `inference_mode`), plus file name, SHA-256 and latency.
- The **History** menu shows the newest rows first; data lives in a named volume, so it survives restarts.

![label=hist1; History: what, when, which model, how confident, local or remote](screenshots/Task4_img_5.png)

![label=hist2; History with record numbers (each opens one record)](screenshots/Task8_img_1.png)

## Task 6 – Containerization with Docker and Docker Compose

### 6.1 Docker warm-up (`scripts/docker_warmup.ps1`, nginx:alpine)

![label=w1; Run with port mapping (8081:80) and a mounted volume](screenshots/Task6_img_1.png)

![label=w2; Container logs](screenshots/Task6_img_2.png)

![label=w3; Shell inside the container](screenshots/Task6_img_3.png)

![label=w4; Stopping and removing the container](screenshots/Task6_img_5.png)

- **Image** – read-only template a container is created from. **Container** – a running instance with its own writable layer. **Layer** – the result of one Dockerfile instruction, cached and shared. **Volume** – storage outside the container that survives its deletion. **Port mapping** – forwards a host port to a container port.

### 6.2 Dockerfile for the web app
Dockerfile: https://github.com/ALiAdnan3/MLOPS_Assignment_2/blob/main/Dockerfile

1. `python:3.13-slim` base → 2. `ENV` (no pip cache, unbuffered logs) → 3. copy `requirements-serve.txt` + `pip install` **before** the code: dependencies rarely change, so a code change reuses the cached 1.6 GB layer (rebuild ~7 s) → 4. non-root `appuser` → 5. copy code → 6. `HEALTHCHECK` on `/health` → 7. `CMD`.
- `.dockerignore` keeps `.env`, `.git`, data and outputs out → no secrets in the image.
- **`tensorflow-cpu`** (274 MB wheel vs 573 MB for `tensorflow` with unusable GPU code); CPU inference is fast enough.
- The container runs as `appuser` ({fig:compexec}).

### 6.3 Compose stack
Compose file: https://github.com/ALiAdnan3/MLOPS_Assignment_2/blob/main/docker-compose.yml

- Services **web** and **db** (postgres:16-alpine); web reaches the DB by service name `db:5432`.
- Named volume `pgdata`; `pg_isready` health check + `depends_on: condition: service_healthy`.
- **DB publishes no port** (only `5432/tcp`, no host mapping); passwords from untracked `.env`, documented in `.env.example`.
- **Model mounted** read-only (`./models:/app/models:ro`): a new champion only needs fetch + restart, no 2 GB rebuild.

![label=compose; Compose start/stop/restart; both healthy, db port not published](screenshots/Task6_img_6.png)

### 6.4 Image analysis
Largest layer: `pip install` (1.66 GB, TensorFlow alone 1.3 GB). Compared with a deliberately bad image (full base, full `tensorflow`, pip cache kept):

| Image | Disk | Download |
|---|---|---|
| Before (`Dockerfile.unoptimized`) | 5.88 GB | 1.74 GB |
| After (`Dockerfile`) | **2.22 GB** | **421 MB** |

![label=sizes; docker images: optimised vs unoptimised](screenshots/Task6_img_4.png)

**Conclusion:** −62 % on disk, −76 % to download; most of what remains is TensorFlow itself.

### 6.5 Operations and troubleshooting

![label=compexec; Shell in web (appuser, mounted model) and psql in db](screenshots/Task6_img_7.png)

**Break on purpose:** started web with a wrong DB password (`$env:POSTGRES_PASSWORD="wrong-password"`). Web kept restarting. Logs only:
```
web-1 | psycopg2.OperationalError: connection to server at "db" (172.18.0.2), port 5432 failed:
        FATAL:  password authentication failed for user "visionguard"
db-1  | FATAL:  password authentication failed for user "visionguard"
```
- **Diagnosis:** the DB answered (network OK) but refused the password; Postgres only applies `POSTGRES_PASSWORD` when it creates a new volume, so the old password stayed.
- **Fix:** removed the wrong variable, recreated web → healthy, history still there.

![label=ops3; Fixed: both healthy, /health database ok](screenshots/Task6_img_8.png)

### 6.6 GitHub Actions (CI)
- **Toy** (`23L0839_A2_Ops_ci_toy`): runs on every push, matrix Python 3.11 + 3.12, pytest. Green → red (bug on purpose: `assert -1 == 5`) → green (fixed).
- **VisionGuard** (`.github/workflows/ci.yml`): flake8 + 7 pytest tests, then Docker build; badge in the README.
- **Secrets** live in GitHub *Settings → Secrets and variables → Actions* (encrypted, masked) and are used as `${{ secrets.NAME }}` – never in the workflow file, because it is public and stays in Git history.

![label=citoy; Toy CI: green, red, green](screenshots/Task6_img_9.png)

![label=ciproj; VisionGuard CI: all runs green](screenshots/Task6_img_10.png)

## Task 7 – API Deployment (Microsoft Azure, pathway C)
**API URL:** https://vg-23l0839-a2-ops-api.gentleglacier-7f267617.eastus.azurecontainerapps.io (`GET /health`, `POST /predict` with header `X-API-Key`).

**Research:**
- Free account: USD 200 credit; Container Apps has a monthly free grant, and with **0 minimum replicas** it costs ~0 when idle (price: ~15 s cold start).
- Free trial blocks `az acr build` (`TasksOperationsNotAllowed`) → built locally and pushed.
- Compose → Azure: image → **Container Registry**; service → **Container App**; ports → **ingress** (HTTPS → 8080); `.env` → **secrets/env vars**; logs → **Log Analytics**; DB stays local (Task 8).
- Model **built into the image** (no shared disk; immutable = tested model). API key compared in constant time; HTTPS only; 5 MB limit.

**Steps:**
1. `az login`; `az provider register` (App, OperationalInsights, ContainerRegistry).
2. `az group create -n 23L0839_A2_Ops -l eastus`.
3. `az acr create -n acr23l0839a2ops --sku Basic`.
4. `python -m src.registry fetch` → `docker build -f Dockerfile.api -t acr23l0839a2ops.azurecr.io/23l0839-a2-ops-api:v2 .` → `az acr login` + `docker push`.
5. `az containerapp env create -n env-23l0839-a2-ops`.
6. `az containerapp create -n vg-23l0839-a2-ops-api --image …:v2 --target-port 8080 --ingress external --cpu 1.0 --memory 2.0Gi --min-replicas 0 --max-replicas 1 --secrets api-key=… --env-vars API_KEY=secretref:api-key THRESHOLD=0.8 MODEL_VERSION=v2@champion`.

![label=azure; Azure resource group: registry, environment, Container App, log workspace](screenshots/Task7_img_1.png)

Container App configuration (`az containerapp show`):
```
state: Succeeded   running: Running   location: East US   external ingress -> port 8080
url:   vg-23l0839-a2-ops-api.gentleglacier-7f267617.eastus.azurecontainerapps.io
image: acr23l0839a2ops.azurecr.io/23l0839-a2-ops-api:v2   cpu 1.0  memory 2Gi  replicas 0-1
env:   API_KEY -> secretRef api-key | THRESHOLD=0.8 | MODEL_VERSION=v2@champion | MAX_UPLOAD_MB=5
```

**Postman test:** I ran my Postman collection (`postman/23L0839_A2_Ops_VisionGuard.postman_collection.json`) with Newman, Postman's command-line runner – 4 requests, 7 assertions, 0 failed:
```
1 Health                    GET  /health   [200 OK, 1057 ms]   √ 200 OK   √ model loaded
2 Predict AI image          POST /predict  [200 OK, 2.9 s]     √ 200      √ label AI
3 Predict real image        POST /predict  [200 OK, 546 ms]    √ 200      √ label Real
4 Predict WITHOUT API key   POST /predict  [401 Unauthorized]  √ 401 unauthorized
```

![label=postman; Postman collection run report: 7 assertions, 0 failed](screenshots/extra_postman_newman.png)

**Tested from other networks:** check-host.net requested `/health` from servers in 4 countries – all got **200 OK in ~0.35 s**.

![label=networks; API reached from Poland, Russia, Serbia and Sweden](screenshots/extra_other_networks.png)

## Task 8 – Integrating Flask and the API
- Flask runs **on localhost** (Docker Compose with PostgreSQL) with `INFERENCE_MODE=remote`, `MODEL_API_URL` and `MODEL_API_KEY` (`scripts/set_mode.ps1 remote`).
- In remote mode Flask validates the upload, sends it to the Azure API with the key, saves the row with `inference_mode = remote` and shows the result; if the API fails the user sees "Model unavailable".

![label=healthremote; /health in remote mode, including the cloud API status](screenshots/extra_health_remote.png)

![label=remoteresult; Result page in remote mode (Inference: remote)](screenshots/extra_remote_result.png)

**Multiple inputs** (form → Flask → Azure → PostgreSQL), 8 unseen test images – 7 correct:

| Image | Truth | Prediction (remote) | Confidence |
|---|---|---|---|
| images71.jpg | AI | AI-generated | 89.8 % |
| Nov20_17_498430921.jpg | AI | Real ✗ (half-real composite) | 89.7 % |
| il_570xN.4588037671_ak6c.jpg | AI | AI-generated | 99.9 % |
| 63964cc7…google-not-against-ai.jpg | AI | AI-generated | 99.8 % |
| Portrait197a-1280x640.jpg | Real | Real | 98.3 % |
| 1032249.jpg | Real | Real | 97.1 % |
| gettyimages-1238924365_wide….jpg | Real | Real | 66.4 % |
| pexels-photo-2422290.jpeg | Real | Real | 74.0 % |

The same requests arriving in the Azure log (model time ~0.2 s; the rest is the upload from Pakistan to US East):
```
11:48:29 INFO api: predict file=ai_1.png label=AI-generated prob_ai=0.999 228.8ms
11:48:31 INFO api: predict file=ai_2.jpg label=AI-generated prob_ai=0.999 189.6ms
11:48:34 INFO api: predict file=real_1.jpg label=Real prob_ai=0.008 178.2ms
11:48:35 INFO api: predict file=real_2.jpg label=Real prob_ai=0.017 186.7ms
```
The History menu ({fig:hist1}) logs these rows with Mode = remote next to the local ones.

## Conclusion
- Built the full path from raw data to a cloud API: versioned data, two models, tracked experiments, a registered champion, a validated Flask app with a PostgreSQL trail, a lean Docker setup with CI, and an Azure API used by the local app.
- Final model: test ROC-AUC 0.86 with precision 0.87 – matching my decision that falsely accusing a real photo is the worse error.
- Lessons: share preprocessing between training and serving; choose models on validation data; run-to-run noise can be as large as the effect being measured; read the logs – they solved every problem I hit (BatchNorm, TensorBoard, MLflow 3 models, Windows console, Azure trial limits, wrong DB password).
