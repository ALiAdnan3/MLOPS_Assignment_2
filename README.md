# 23L0839_A2_Ops – VisionGuard

[![CI](https://github.com/ALiAdnan3/MLOPS_Assignment_2/actions/workflows/ci.yml/badge.svg)](https://github.com/ALiAdnan3/MLOPS_Assignment_2/actions/workflows/ci.yml)

VisionGuard tells whether an uploaded image is **AI-generated** or **real** (human-made photo / art).
TensorFlow/Keras model · MLflow + DagsHub tracking · DVC data versioning · Flask app · database request log · Docker Compose · cloud API on Azure.

| | |
|---|---|
| DagsHub (code, DVC data, MLflow runs, model registry) | https://dagshub.com/ALiAdnan3/23L0839_A2_Ops |
| Live model API (Azure Container Apps) | https://vg-23l0839-a2-ops-api.gentleglacier-7f267617.eastus.azurecontainerapps.io/health |
| Champion model | `models:/23L0839_A2_Ops-visionguard@champion` (v2, MobileNetV2, threshold 0.80) |

Dataset: [AI Generated Images vs Real Images (Kaggle)](https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images) – versioned with DVC (`data/raw.dvc`, remote = DagsHub storage).

## Layout
| Path | Purpose |
|---|---|
| `visionguard/` | shared package: preprocessing, tf.data pipeline, models, inference (used by training **and** serving) |
| `src/` | audit, EDA, splits, training (`train.py`), tuner, registry, error analysis, Grad-CAM, MLflow toy |
| `webapp/` | Flask app (upload / result / history / health), SQLAlchemy database trail, templates + CSS |
| `api/` | model API deployed to Azure (`POST /predict` with `X-API-Key`, `GET /health`) |
| `tests/` | pytest suite run in CI |
| `scripts/` | experiment runner, demos, DVC / Docker lab scripts |
| `Dockerfile`, `Dockerfile.api`, `docker-compose.yml` | containers: web app, cloud API, web + Postgres stack |
| `.github/workflows/ci.yml` | lint + tests + Docker build on every push |
| `postman/` | Postman collection for the cloud API |
| `reports/` | audit table, figures, experiment comparison, report notes, report builder |

## Reproduce
```bash
python -m venv .venv && .venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env                                   # fill in DagsHub token, Postgres password, API key
dvc pull                                                 # fetch the images from DagsHub
python -m src.audit_dataset && python -m src.eda && python -m src.make_splits
python -m src.train --model mobilenetv2 --mlflow manual  # logs to DagsHub MLflow
python -m src.registry fetch                             # champion model -> models/champion.keras
```

## Run the app
```bash
docker compose up -d --build        # web on http://localhost:8000 + Postgres (internal only)
```
`INFERENCE_MODE=local` serves the mounted champion model inside the container; `INFERENCE_MODE=remote` (with `MODEL_API_URL` and `MODEL_API_KEY` in `.env`) makes the local Flask app call the cloud API instead.

## Call the cloud API
```bash
curl https://vg-23l0839-a2-ops-api.gentleglacier-7f267617.eastus.azurecontainerapps.io/health
curl -H "X-API-Key: <key>" -F image=@photo.jpg https://vg-23l0839-a2-ops-api.gentleglacier-7f267617.eastus.azurecontainerapps.io/predict
```
The first request after a quiet period can take ~15 s (the app scales to zero and cold-starts).
