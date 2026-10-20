"""Task 3.6 - MLflow Model Registry on DagsHub.

  python -m src.registry register     # v1 = reference run, v2 = best run (alias champion), v3 = runner-up (challenger)
  python -m src.registry fetch        # download champion -> models/champion.keras + models/champion.json (for serving)

The serving app never talks to DagsHub per request: `fetch` is run ONCE at deploy
time and the .keras file is mounted into / baked into the container.
"""
import json
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow  # noqa: E402

MODEL_NAME = "23L0839_A2_Ops-visionguard"
EXPERIMENT = "23L0839_A2_Ops-VisionGuard"
# chosen in report 3.5 on VALIDATION metrics (val AUC, val loss, val F0.5)
VERSIONS = [("R1_mnv2_reference", None), ("R4_mnv2_no_augment", "champion"), ("R8_effnetb0", "challenger")]


def run_by_name(client, name):
    exp = client.get_experiment_by_name(EXPERIMENT)
    runs = client.search_runs([exp.experiment_id], f"attributes.run_name = '{name}'", order_by=["start_time DESC"])
    return runs[0]


def register():
    client = mlflow.MlflowClient()
    for run_name, alias in VERSIONS:
        run = run_by_name(client, run_name)
        # MLflow 3 stores models as "LoggedModels"; DagsHub cannot resolve runs:/<id>/model, so use the model's own URI
        logged = [m for m in client.search_logged_models(experiment_ids=[run.info.experiment_id])
                  if m.source_run_id == run.info.run_id]
        mv = mlflow.register_model(logged[0].model_uri, MODEL_NAME,
                                   tags={"run_name": run_name, "threshold": str(run.data.metrics["threshold"])})
        client.set_model_version_tag(MODEL_NAME, mv.version, "val_auc", str(run.data.metrics["val_auc"]))
        if alias:
            client.set_registered_model_alias(MODEL_NAME, alias, mv.version)
        print(f"registered {MODEL_NAME} v{mv.version} <- {run_name}" + (f"  alias @{alias}" if alias else ""))
    client.update_registered_model(MODEL_NAME, "VisionGuard AI-vs-real classifier. "
                                               "@champion = model served in production.")


def fetch(out_dir="models"):
    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
    run = client.get_run(mv.run_id)
    tmp = mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="keras_file/model.keras")
    Path(out_dir).mkdir(exist_ok=True)
    shutil.copy(tmp, Path(out_dir) / "champion.keras")
    meta = {"model_name": MODEL_NAME, "version": mv.version, "alias": "champion", "run_id": mv.run_id,
            "run_name": run.info.run_name, "threshold": run.data.metrics["threshold"]}
    (Path(out_dir) / "champion.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    {"register": register, "fetch": fetch}[sys.argv[1]]()
