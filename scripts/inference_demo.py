"""Task 4.1 - console demo: load the champion ONCE, classify two known images (one AI, one real)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from dotenv import load_dotenv  # noqa: E402

load_dotenv(".env")
from visionguard.inference import Predictor  # noqa: E402

p = Predictor(os.getenv("MODEL_PATH", "models/champion.keras"), float(os.getenv("THRESHOLD", "0.5")),
              os.getenv("MODEL_NAME", "23L0839_A2_Ops-visionguard"), os.getenv("MODEL_VERSION", "champion"))
print(f"model loaded once in {p.load_seconds}s  (version {p.model_version}, threshold {p.threshold})")
for path, truth in [(sys.argv[1], "AI-generated"), (sys.argv[2], "Real")]:
    t0 = time.perf_counter()
    with open(path, "rb") as f:
        r = p.predict_bytes(f.read())
    print(f"{os.path.basename(path)[:45]:45s} truth={truth:12s} -> {r['label']:12s} "
          f"confidence={r['confidence']:.1%}  P(AI)={r['prob_ai']:.3f}  ({(time.perf_counter() - t0) * 1000:.0f} ms)")
