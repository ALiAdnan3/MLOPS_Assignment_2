"""Task 3.6 - load the registered model BY ALIAS from DagsHub and classify one image.

  python scripts/predict_champion.py data/raw/RealArt/<some image>.jpg
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(".env")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow  # noqa: E402
import mlflow.tensorflow  # noqa: E402

from visionguard.inference import to_result  # noqa: E402
from visionguard.preprocess import load_image_path  # noqa: E402

MODEL_URI = "models:/23L0839_A2_Ops-visionguard@champion"

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
client = mlflow.MlflowClient()
mv = client.get_model_version_by_alias("23L0839_A2_Ops-visionguard", "champion")
threshold = float(mv.tags.get("threshold", 0.5))
print(f"Tracking server : {mlflow.get_tracking_uri()}")
print(f"Loading         : {MODEL_URI}  (version {mv.version}, run {mv.tags.get('run_name')})")
model = mlflow.tensorflow.load_model(MODEL_URI)

path = sys.argv[1]
x = load_image_path(path).astype("float32")[None]
prob_ai = float(model(x, training=False).numpy().ravel()[0])
r = to_result(prob_ai, threshold, "23L0839_A2_Ops-visionguard", f"v{mv.version}@champion")
print(f"Image           : {path}")
print(f"Prediction      : {r['label']}  (confidence {r['confidence']:.1%}, "
      f"P(AI)={r['prob_ai']:.3f}, threshold {threshold})")
