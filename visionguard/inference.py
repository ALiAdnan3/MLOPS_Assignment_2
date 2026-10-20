"""Task 4.1 - inference module.

The Keras model is loaded ONCE when the Predictor is created (at app start-up)
and reused for every request. Preprocessing is the exact function used in the
tf.data training pipeline (visionguard.preprocess.load_image); model-specific
scaling lives inside the saved model.
"""
import time

import numpy as np

from visionguard.preprocess import CLASS_NAMES, IMG_SIZE, load_image

CAVEAT = ("VisionGuard is a statistical model trained on ~700 web images. It can be wrong, especially for "
          "paintings, heavily edited photos, screenshots, collages and small or re-compressed images. "
          "Treat the result as a hint, not as proof.")


def to_result(prob_ai: float, threshold: float, model_name: str, model_version: str) -> dict:
    is_ai = prob_ai >= threshold
    label = CLASS_NAMES[int(is_ai)]
    # confidence = probability the model assigns to the class it chose
    confidence = prob_ai if is_ai else 1.0 - prob_ai
    return {"label": label, "confidence": round(float(confidence), 4), "prob_ai": round(float(prob_ai), 4),
            "threshold": threshold, "model_name": model_name, "model_version": model_version}


class Predictor:
    def __init__(self, model_path: str, threshold: float = 0.5, model_name: str = "visionguard",
                 model_version: str = "unknown"):
        import tensorflow as tf  # imported here so the web app can run in remote mode without TF

        t0 = time.perf_counter()
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.img_size = int(self.model.input_shape[1] or IMG_SIZE)
        self.threshold = threshold
        self.model_name = model_name
        self.model_version = model_version
        self.load_seconds = round(time.perf_counter() - t0, 2)
        self.model(np.zeros((1, self.img_size, self.img_size, 3), np.float32))  # warm-up: build graph once

    def predict_bytes(self, data: bytes) -> dict:
        x = load_image(data, self.img_size).astype(np.float32)[None]  # same function as training
        prob_ai = float(self.model(x, training=False).numpy().ravel()[0])
        return to_result(prob_ai, self.threshold, self.model_name, self.model_version)
