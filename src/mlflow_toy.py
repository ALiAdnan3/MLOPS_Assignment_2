"""Task 3.1 / 3.2 - MLflow tool lab on Fashion-MNIST.

  python -m src.mlflow_toy --target local     # 3.1: ./mlflow_local.db, view with `mlflow ui`
  python -m src.mlflow_toy --target dagshub   # 3.2: same runs on DagsHub (URI + token from .env)

Two runs that differ only in learning rate. Each logs parameters, per-epoch
metrics, one plot (loss curves) and the trained model.
"""
import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.tensorflow  # noqa: E402
import tensorflow as tf  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

LOCAL_URI = "sqlite:///mlflow_local.db"


class PerEpoch(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=epoch)


def run(lr, x, y, xv, yv, epochs=5, units=64):
    tf.keras.utils.set_random_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Input((28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    with mlflow.start_run(run_name=f"toy_lr_{lr}"):
        mlflow.set_tags({"lab": "3.1 toy", "dataset": "fashion_mnist", "student": "23L0839"})
        mlflow.log_params({"learning_rate": lr, "units": units, "epochs": epochs, "batch_size": 128})
        h = model.fit(x, y, validation_data=(xv, yv), epochs=epochs, batch_size=128, callbacks=[PerEpoch()], verbose=2)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(h.history["loss"], label="train")
        ax.plot(h.history["val_loss"], label="val")
        ax.set_title(f"loss, lr={lr}")
        ax.set_xlabel("epoch")
        ax.legend()
        mlflow.log_figure(fig, "loss_curve.png")
        mlflow.log_metric("final_val_accuracy", h.history["val_accuracy"][-1])
        mlflow.tensorflow.log_model(model, name="model")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["local", "dagshub"], default="local")
    a = p.parse_args()
    if a.target == "local":
        mlflow.set_tracking_uri(LOCAL_URI)
    else:
        load_dotenv()   # MLFLOW_TRACKING_URI / _USERNAME / _PASSWORD come from .env, never from code
        assert os.getenv("MLFLOW_TRACKING_URI"), "MLFLOW_TRACKING_URI missing in .env"
    print("Tracking URI:", mlflow.get_tracking_uri())
    mlflow.set_experiment("23L0839_A2_Ops-toy-fashion-mnist")

    (x, y), (xv, yv) = tf.keras.datasets.fashion_mnist.load_data()
    x, xv = x[:20000] / 255.0, xv[:5000] / 255.0   # subset: keeps the toy fast on CPU
    y, yv = y[:20000], yv[:5000]
    for lr in (1e-3, 1e-2):
        run(lr, x, y, xv, yv)


if __name__ == "__main__":
    main()
