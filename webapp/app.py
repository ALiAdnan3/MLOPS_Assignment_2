"""VisionGuard Flask web app (Tasks 4, 5, 8).

Routes
  GET  /            upload page
  POST /predict     validate -> predict -> log to DB -> result page
  GET  /history     table of past predictions (from the database)
  GET  /health      JSON health check (model + database)

Run (production WSGI server, debug off):  python -m webapp.app
"""
import base64
import hashlib
import logging
import time

import requests
from flask import Flask, abort, jsonify, render_template, request
from PIL import UnidentifiedImageError
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from werkzeug.utils import secure_filename

from visionguard.inference import CAVEAT, to_result
from visionguard.preprocess import load_image
from webapp.config import Config
from webapp.db import Database

log = logging.getLogger("visionguard")


class RemotePredictor:
    """Task 8 - calls the model API hosted in the cloud instead of loading TensorFlow locally."""

    def __init__(self, cfg):
        if not cfg.MODEL_API_URL:
            raise RuntimeError("INFERENCE_MODE=remote but MODEL_API_URL is not set")
        self.url = cfg.MODEL_API_URL.rstrip("/")
        self.headers = {"X-API-Key": cfg.MODEL_API_KEY}
        self.timeout = cfg.MODEL_API_TIMEOUT
        self.model_name, self.model_version = "remote", "unknown"

    def health(self):
        r = requests.get(f"{self.url}/health", timeout=self.timeout)
        r.raise_for_status()
        info = r.json()
        self.model_name, self.model_version = info.get("model_name", "remote"), info.get("model_version", "?")
        return info

    def predict_bytes(self, data, filename="upload"):
        r = requests.post(f"{self.url}/predict", headers=self.headers, timeout=self.timeout,
                          files={"image": (filename, data)})
        if r.status_code >= 400:
            raise RuntimeError(f"model API returned {r.status_code}: {r.text[:200]}")
        j = r.json()
        return to_result(j["prob_ai"], j["threshold"], j["model_name"], j["model_version"])


class UploadError(Exception):
    def __init__(self, title, message, status=400):
        super().__init__(message)
        self.title, self.message, self.status = title, message, status


def validate_upload(cfg):
    """Returns (safe_filename, bytes). The bytes stay in memory - nothing is written to disk."""
    f = request.files.get("image")
    if f is None or not f.filename:
        raise UploadError("No file selected", "Please choose an image to upload.")
    name = secure_filename(f.filename) or "upload"          # strips paths like ../../etc/passwd
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in cfg.ALLOWED_EXTENSIONS:
        raise UploadError("Unsupported file type",
                          f"'.{ext or '?'}' is not allowed. Upload one of: "
                          f"{', '.join(sorted(cfg.ALLOWED_EXTENSIONS))}.", 415)
    data = f.read()
    if not data:
        raise UploadError("Empty file", "The uploaded file is empty.")
    try:
        load_image(data)                                     # real decode, not just the extension
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        raise UploadError("Not a readable image",
                          f"'{name}' has an image extension but its content could not be decoded. "
                          "It may be corrupt or not an image at all.", 422)
    return name, data


def create_app(predictor=None, db=None):
    cfg = Config()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = Flask(__name__)
    app.config.from_object(cfg)

    # ---- created ONCE at start-up, shared by all requests
    if predictor is None:
        if cfg.INFERENCE_MODE == "remote":
            predictor = RemotePredictor(cfg)
            try:
                log.info("model API health: %s", predictor.health())
            except Exception as e:  # API may be cold-starting; requests will retry
                log.warning("model API not reachable yet: %s", e)
        else:
            from visionguard.inference import Predictor
            predictor = Predictor(cfg.MODEL_PATH, cfg.THRESHOLD, cfg.MODEL_NAME, cfg.MODEL_VERSION)
            log.info("model %s loaded from %s in %ss", cfg.MODEL_VERSION, cfg.MODEL_PATH, predictor.load_seconds)
    db = db or Database(cfg.DATABASE_URL)
    db.create_tables()
    app.extensions["predictor"], app.extensions["db"] = predictor, db

    @app.get("/")
    def index():
        return render_template("index.html", max_mb=cfg.MAX_UPLOAD_MB, allowed=sorted(cfg.ALLOWED_EXTENSIONS))

    @app.post("/predict")
    def predict():
        name, data = validate_upload(cfg)
        t0 = time.perf_counter()
        try:
            if cfg.INFERENCE_MODE == "remote":
                result = predictor.predict_bytes(data, name)
            else:
                result = predictor.predict_bytes(data)
        except Exception as e:
            log.exception("inference failed")
            raise UploadError("Model unavailable", f"The model could not process the image right now ({e}).", 503)
        latency = round((time.perf_counter() - t0) * 1000, 1)
        row = db.add(filename=name, image_sha256=hashlib.sha256(data).hexdigest(), label=result["label"],
                     confidence=result["confidence"], prob_ai=result["prob_ai"], threshold=result["threshold"],
                     model_name=result["model_name"], model_version=result["model_version"],
                     inference_mode=cfg.INFERENCE_MODE, latency_ms=latency)
        log.info("prediction id=%s file=%s label=%s conf=%.3f mode=%s %.0fms", row.id, name, result["label"],
                 result["confidence"], cfg.INFERENCE_MODE, latency)
        # preview is sent back inline (base64) from memory - the upload is never stored on disk
        preview = f"data:image/jpeg;base64,{base64.b64encode(data).decode()}" if len(data) < 3_000_000 else None
        return render_template("result.html", r=result, row=row, preview=preview, caveat=CAVEAT)

    @app.get("/history")
    def history():
        return render_template("history.html", rows=db.recent(200))

    @app.get("/history/<int:pred_id>")
    def history_item(pred_id):
        row = db.get(pred_id) or abort(404)
        return render_template("history.html", rows=[row], single=True)

    @app.get("/health")
    def health():
        db_ok = db.ping()
        body = {"status": "ok" if db_ok else "degraded", "database": "ok" if db_ok else "unreachable",
                "inference_mode": cfg.INFERENCE_MODE, "model_name": predictor.model_name,
                "model_version": predictor.model_version, "threshold": cfg.THRESHOLD,
                "debug": app.debug}
        if cfg.INFERENCE_MODE == "remote":
            try:
                body["model_api"] = predictor.health()
            except Exception as e:
                body["model_api"], body["status"] = f"unreachable: {e}", "degraded"
        return jsonify(body), 200 if body["status"] == "ok" else 503

    # ---- friendly error pages
    @app.errorhandler(UploadError)
    def upload_error(e):
        log.warning("rejected upload: %s - %s", e.title, e.message)
        return render_template("error.html", title=e.title, message=e.message, status=e.status), e.status

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(e):
        log.warning("rejected upload: larger than %s MB", cfg.MAX_UPLOAD_MB)
        return render_template("error.html", title="File too large",
                               message=f"The maximum upload size is {cfg.MAX_UPLOAD_MB:g} MB.", status=413), 413

    @app.errorhandler(HTTPException)
    def http_error(e):
        return render_template("error.html", title=e.name, message=e.description, status=e.code), e.code

    return app


if __name__ == "__main__":
    from waitress import serve

    application = create_app()
    application.debug = False   # never run debug mode in production (interactive debugger = remote code execution)
    log.info("VisionGuard listening on %s:%s (debug=%s)", Config.HOST, Config.PORT, application.debug)
    serve(application, host=Config.HOST, port=Config.PORT, threads=4)
