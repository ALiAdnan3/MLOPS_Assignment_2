"""Web app tests - run without TensorFlow or a real model (fake predictor + temporary SQLite)."""
import io

import pytest
from PIL import Image

from visionguard.inference import to_result
from webapp.app import create_app
from webapp.db import Database


class FakePredictor:
    model_name, model_version = "fake-model", "test"

    def predict_bytes(self, data, *_):
        return to_result(0.9, 0.72, self.model_name, self.model_version)


def png_bytes(size=(64, 48)):
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 30, 30)).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    app = create_app(predictor=FakePredictor(), db=Database(f"sqlite:///{tmp_path / 'test.db'}"))
    app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024
    return app.test_client()


def upload(client, data, name):
    return client.post("/predict", data={"image": (io.BytesIO(data), name)}, content_type="multipart/form-data")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json["database"] == "ok" and r.json["debug"] is False


def test_valid_upload_is_predicted_and_logged(client):
    r = upload(client, png_bytes(), "cat.png")
    assert r.status_code == 200 and b"AI-generated" in r.data
    h = client.get("/history")
    assert b"cat.png" in h.data and b"fake-model" in h.data


def test_wrong_file_type_rejected(client):
    r = upload(client, b"hello", "notes.txt")
    assert r.status_code == 415 and b"Unsupported file type" in r.data


def test_undecodable_image_rejected(client):
    r = upload(client, b"this is not really a jpeg", "fake.jpg")
    assert r.status_code == 422 and b"Not a readable image" in r.data


def test_too_large_rejected(client):
    r = upload(client, b"0" * (2 * 1024 * 1024), "big.png")
    assert r.status_code == 413 and b"File too large" in r.data


def test_filename_is_sanitized(client):
    upload(client, png_bytes(), "../../etc/passwd.png")
    assert b"etc_passwd.png" in client.get("/history").data


def test_confidence_is_for_chosen_class():
    assert to_result(0.9, 0.5, "m", "v")["label"] == "AI-generated"
    r = to_result(0.2, 0.5, "m", "v")
    assert r["label"] == "Real" and r["confidence"] == pytest.approx(0.8)
