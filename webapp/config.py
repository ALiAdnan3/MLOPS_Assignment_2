"""Task 4.4 - every setting comes from environment variables (12-factor style).

Nothing environment-specific is hard-coded: the same image runs locally, in
Docker Compose and in the cloud, only the environment changes.
"""
import os


def _float(name, default):
    return float(os.getenv(name, default))


class Config:
    # inference
    INFERENCE_MODE = os.getenv("INFERENCE_MODE", "local")          # local | remote
    MODEL_PATH = os.getenv("MODEL_PATH", "models/champion.keras")
    MODEL_NAME = os.getenv("MODEL_NAME", "23L0839_A2_Ops-visionguard")
    MODEL_VERSION = os.getenv("MODEL_VERSION", "champion")
    THRESHOLD = _float("THRESHOLD", 0.5)
    MODEL_API_URL = os.getenv("MODEL_API_URL", "")                  # remote mode: cloud model API
    MODEL_API_KEY = os.getenv("MODEL_API_KEY", "")
    MODEL_API_TIMEOUT = _float("MODEL_API_TIMEOUT", 30)

    # database: postgresql+psycopg2://user:pass@db:5432/visionguard  or  sqlite:///visionguard.db
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///visionguard.db")

    # uploads
    MAX_UPLOAD_MB = _float("MAX_UPLOAD_MB", 5)
    MAX_CONTENT_LENGTH = int(MAX_UPLOAD_MB * 1024 * 1024)            # Flask rejects bigger bodies with 413
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "gif", "bmp"}

    SECRET_KEY = os.getenv("SECRET_KEY", os.urandom(16).hex())
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
