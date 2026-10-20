# 23L0839_A2_Ops – VisionGuard

[![CI](https://github.com/ALiAdnan3/MLOPS_Assignment_2/actions/workflows/ci.yml/badge.svg)](https://github.com/ALiAdnan3/MLOPS_Assignment_2/actions/workflows/ci.yml)

VisionGuard tells whether an uploaded image is **AI-generated** or **real** (human-made photo / art).
TensorFlow/Keras model · MLflow + DagsHub tracking · DVC data versioning · Flask app · database request log · Docker Compose · cloud API.

Dataset: [AI Generated Images vs Real Images (Kaggle)](https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images) – versioned with DVC (`data/raw.dvc`).

## Layout
| Path | Purpose |
|---|---|
| `visionguard/` | shared package: preprocessing, tf.data pipeline, models (used by training **and** serving) |
| `src/` | scripts: dataset audit, EDA, splits, training |
| `scripts/` | small utilities (environment check, …) |
| `data/raw.dvc` | DVC pointer to the image folder (data itself is in DVC remote storage) |
| `data/splits/` | train / val / test lists |
| `reports/` | audit table, figures, report notes |

## Quick start
```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
dvc pull                                             # fetch the images
python -m src.audit_dataset && python -m src.eda && python -m src.make_splits
python -m src.train --model mobilenetv2 --mlflow manual
```
