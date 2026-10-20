"""VisionGuard model API (Task 7) - deployed to the cloud, called by the Flask web app (Task 8).

  GET  /health    public: model name/version, threshold
  POST /predict   multipart field "image"; header X-API-Key required
                  -> {"label", "confidence", "prob_ai", "threshold", "model_name", "model_version", "latency_ms"}

Settings from environment: MODEL_PATH, THRESHOLD, MODEL_NAME, MODEL_VERSION, API_KEY, MAX_UPLOAD_MB, PORT.
"""
import hmac
import logging
import os
import time

from flask import Flask, jsonify, request
from PIL import UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge

from visionguard.inference import Predictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s api: %(message)s")
log = logging.getLogger("api")

API_KEY = os.getenv("API_KEY", "")
ALLOWED = {"jpg", "jpeg", "png", "webp", "gif", "bmp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(float(os.getenv("MAX_UPLOAD_MB", "5")) * 1024 * 1024)
predictor = Predictor(os.getenv("MODEL_PATH", "models/champion.keras"), float(os.getenv("THRESHOLD", "0.5")),
                      os.getenv("MODEL_NAME", "23L0839_A2_Ops-visionguard"), os.getenv("MODEL_VERSION", "champion"))
log.info("model loaded in %ss (version %s)", predictor.load_seconds, predictor.model_version)


def error(status, message):
    return jsonify({"error": message}), status


@app.get("/")
@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "23L0839_A2_Ops visionguard model API",
                    "model_name": predictor.model_name, "model_version": predictor.model_version,
                    "threshold": predictor.threshold, "input_size": predictor.img_size})


@app.post("/predict")
def predict():
    # constant-time comparison so the key cannot be guessed from response timing
    if not API_KEY or not hmac.compare_digest(request.headers.get("X-API-Key", ""), API_KEY):
        return error(401, "missing or invalid X-API-Key")
    f = request.files.get("image")
    if f is None or not f.filename:
        return error(400, "multipart field 'image' is required")
    ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
    if ext not in ALLOWED:
        return error(415, f"file type '.{ext}' not allowed")
    data = f.read()
    t0 = time.perf_counter()
    try:
        result = predictor.predict_bytes(data)
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        return error(422, "file could not be decoded as an image")
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    log.info("predict file=%s label=%s prob_ai=%.3f %sms", f.filename, result["label"], result["prob_ai"],
             result["latency_ms"])
    return jsonify(result)


@app.errorhandler(RequestEntityTooLarge)
def too_large(_):
    return error(413, "file too large")


if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")), threads=4)
