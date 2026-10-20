"""Train + evaluate a VisionGuard model (Tasks 2.2-2.5, 3.3-3.5).

Examples (run from repo root):
  python -m src.train --model cnn --mlflow off
  python -m src.train --model mobilenetv2 --mlflow manual
  python -m src.train --model mobilenetv2 --mlflow autolog --run-name tl_autolog

Everything for one run is written to outputs/<run_name>/ and, if MLflow is on,
logged to the tracking server given by MLFLOW_TRACKING_URI (.env).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from sklearn import metrics as skm  # noqa: E402

from visionguard.data import make_dataset  # noqa: E402
from visionguard.models import (build_baseline_cnn, build_transfer_model,  # noqa: E402
                                compile_model, unfreeze_last)

SPLITS = Path("data/splits")


# ----------------------------------------------------------------------------- helpers
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mobilenetv2", choices=["cnn", "mobilenetv2", "efficientnetb0"])
    p.add_argument("--run-name", default=None)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30, help="max epochs (stage 1 for transfer)")
    p.add_argument("--fine-tune-epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--fine-tune-lr-factor", type=float, default=0.1)
    p.add_argument("--unfreeze-layers", type=int, default=None, help="default: last block")
    p.add_argument("--no-fine-tune", action="store_true")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--augment", default="light", choices=["none", "light", "strong"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--mlflow", default="off", choices=["off", "manual", "autolog"])
    p.add_argument("--experiment", default="23L0839_A2_Ops-VisionGuard")
    return p.parse_args()


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def dvc_data_version():
    """The md5 inside data/raw.dvc identifies the exact dataset version."""
    try:
        for line in Path("data/raw.dvc").read_text().splitlines():
            if "md5:" in line:
                return line.split("md5:")[1].strip()
    except FileNotFoundError:
        pass
    return "untracked"


def callbacks(ckpt_path, patience, best_so_far=None):
    # best_so_far: stage 2 only overwrites the checkpoint if it beats stage 1's best val_loss
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, verbose=0,
                                           initial_value_threshold=best_so_far),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7, verbose=1),
    ]


class MLflowEpochLogger(tf.keras.callbacks.Callback):
    """Manual MLflow logging: one metric point per epoch (train + val + learning rate)."""

    def __init__(self, offset=0):
        super().__init__()
        self.offset = offset

    def on_epoch_end(self, epoch, logs=None):
        import mlflow
        logs = dict(logs or {})
        logs["lr"] = float(self.model.optimizer.learning_rate.numpy())
        mlflow.log_metrics({k: float(v) for k, v in logs.items()}, step=epoch + self.offset)


def merge_histories(histories):
    merged = {}
    for h in histories:
        for k, v in h.history.items():
            merged.setdefault(k, []).extend(v)
    return merged


def plot_curves(hist, out_path, title, stage_boundary=None):
    best = int(np.argmin(hist["val_loss"]))
    epochs = np.arange(1, len(hist["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key in zip(axes, ["loss", "accuracy"]):
        ax.plot(epochs, hist[key], label=f"train {key}")
        ax.plot(epochs, hist[f"val_{key}"], label=f"val {key}")
        ax.axvline(best + 1, color="red", ls="--", label=f"best epoch = {best + 1}")
        if stage_boundary:
            ax.axvline(stage_boundary + 0.5, color="gray", ls=":", label="fine-tuning starts")
        ax.set_xlabel("epoch")
        ax.set_title(key)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return best + 1


def predict(model, csv, args):
    ds = make_dataset(csv, training=False, batch_size=args.batch_size, img_size=args.img_size)
    y = pd.read_csv(csv)["label"].values
    p = model.predict(ds, verbose=0).ravel()
    return y, p


def choose_threshold(y_val, p_val, beta=0.5):
    """Pick the threshold on VALIDATION data that maximises F-beta with beta=0.5.

    beta < 1 weights precision more than recall: for VisionGuard, calling a real
    photo 'AI-generated' (false positive) is the worse error, so we prefer to be
    sure before saying 'AI'.
    """
    best_t, best_s = 0.5, -1
    for t in np.linspace(0.05, 0.95, 91):
        s = skm.fbeta_score(y_val, (p_val >= t).astype(int), beta=beta, zero_division=0)
        if s > best_s:
            best_t, best_s = float(t), s
    return round(best_t, 2), best_s


def evaluate(y, p, threshold, out_dir, tag):
    pred = (p >= threshold).astype(int)
    m = {
        "accuracy": skm.accuracy_score(y, pred),
        "precision": skm.precision_score(y, pred, zero_division=0),
        "recall": skm.recall_score(y, pred, zero_division=0),
        "f1": skm.f1_score(y, pred, zero_division=0),
        "roc_auc": skm.roc_auc_score(y, p),
    }
    cm = skm.confusion_matrix(y, pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    skm.ConfusionMatrixDisplay(cm, display_labels=["Real", "AI"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{tag} confusion matrix (t={threshold})")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    fpr, tpr, _ = skm.roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {m['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate (real called AI)")
    ax.set_ylabel("True positive rate (AI caught)")
    ax.set_title(f"{tag} ROC curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=120)
    plt.close(fig)
    return {k: round(float(v), 4) for k, v in m.items()}, cm


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    load_dotenv()
    tf.keras.utils.set_random_seed(args.seed)  # seeds python, numpy and tensorflow
    run_name = args.run_name or f"{args.model}_{time.strftime('%Y%m%d-%H%M%S')}"
    out = Path("outputs") / run_name
    out.mkdir(parents=True, exist_ok=True)
    ckpt = str(out / "best.keras")

    mlflow = None
    if args.mlflow != "off":
        import mlflow
        import mlflow.tensorflow  # noqa: F401
        mlflow.set_experiment(args.experiment)
        if args.mlflow == "autolog":
            mlflow.tensorflow.autolog(log_models=True, checkpoint=False)
        mlflow.start_run(run_name=run_name)
        mlflow.set_tags({"git_commit": git_commit(), "dvc_data_version": dvc_data_version(),
                         "seed": args.seed, "logging": args.mlflow, "model_family": args.model})
        if args.mlflow == "manual":
            mlflow.log_params({k: v for k, v in vars(args).items() if k not in ("mlflow", "experiment")})

    train_ds = make_dataset(SPLITS / "train.csv", True, args.batch_size, args.img_size, args.augment, args.seed)
    val_ds = make_dataset(SPLITS / "val.csv", False, args.batch_size, args.img_size)

    histories, stage_info = [], {}
    if args.model == "cnn":
        model = compile_model(build_baseline_cnn(args.img_size, dropout=args.dropout), args.lr)
        cbs = callbacks(ckpt, args.patience) + ([MLflowEpochLogger()] if args.mlflow == "manual" else [])
        histories.append(model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cbs))
    else:
        model = compile_model(build_transfer_model(args.model, args.img_size, dropout=args.dropout), args.lr)
        stage_info["stage1_trainable"] = [w.path for w in model.trainable_weights]
        cbs = callbacks(ckpt, args.patience) + ([MLflowEpochLogger()] if args.mlflow == "manual" else [])
        histories.append(model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cbs))
        if not args.no_fine_tune:
            n1 = len(histories[0].history["loss"])
            stage_info["stage2_unfrozen_backbone_layers"] = unfreeze_last(model, args.model, args.unfreeze_layers)
            compile_model(model, args.lr * args.fine_tune_lr_factor)  # recompile after changing trainable
            stage_info["stage2_trainable"] = [w.path for w in model.trainable_weights]
            best1 = min(histories[0].history["val_loss"])
            cbs = callbacks(ckpt, args.patience, best1) +([MLflowEpochLogger(offset=n1)] if args.mlflow == "manual" else [])
            histories.append(model.fit(train_ds, validation_data=val_ds, initial_epoch=n1,
                                       epochs=n1 + args.fine_tune_epochs, callbacks=cbs))
    (out / "trainable_layers.json").write_text(json.dumps(stage_info, indent=1))

    # Best weights (lowest val_loss) were restored by EarlyStopping; the checkpoint holds the same model.
    model = tf.keras.models.load_model(ckpt)
    model.save(out / "model.keras")
    hist = merge_histories(histories)
    boundary = len(histories[0].history["loss"]) if len(histories) > 1 else None
    best_epoch = plot_curves(hist, out / "training_curves.png", f"{run_name}: training curves", boundary)

    # ---- threshold from validation, then ONE evaluation on the test set
    y_val, p_val = predict(model, SPLITS / "val.csv", args)
    threshold, fbeta = choose_threshold(y_val, p_val)
    y_test, p_test = predict(model, SPLITS / "test.csv", args)
    test_m, cm = evaluate(y_test, p_test, threshold, out, run_name)
    pd.DataFrame({"path": pd.read_csv(SPLITS / "test.csv")["path"], "label": y_test, "prob_ai": p_test,
                  "pred": (p_test >= threshold).astype(int)}).to_csv(out / "test_predictions.csv", index=False)

    summary = {"run_name": run_name, "best_epoch": best_epoch, "threshold": threshold, "val_fbeta0.5": round(fbeta, 4),
               "val_auc": round(float(skm.roc_auc_score(y_val, p_val)), 4),
               "best_val_loss": round(float(min(hist["val_loss"])), 4),
               **{f"test_{k}": v for k, v in test_m.items()}, "confusion_matrix": cm.tolist(),
               "params": model.count_params()}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))

    if mlflow:
        mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items()})
        mlflow.log_metrics({"best_epoch": best_epoch, "threshold": threshold, "val_auc": summary["val_auc"],
                            "best_val_loss": summary["best_val_loss"]})
        if args.mlflow == "autolog":  # autolog records fit() args but not our own CLI choices
            mlflow.log_params({"augment": args.augment, "img_size": args.img_size, "model_family": args.model})
        for f in ["confusion_matrix.png", "roc_curve.png", "training_curves.png", "summary.json",
                  "trainable_layers.json", "test_predictions.csv"]:
            mlflow.log_artifact(str(out / f), artifact_path="evaluation")
        for split in ["train", "val", "test"]:
            mlflow.log_artifact(str(SPLITS / f"{split}.csv"), artifact_path="splits")
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        (out / "requirements-freeze.txt").write_text(freeze)
        mlflow.log_artifact(str(out / "requirements-freeze.txt"), artifact_path="environment")
        mlflow.log_artifact(str(out / "model.keras"), artifact_path="keras_file")
        if args.mlflow == "manual":
            mlflow.tensorflow.log_model(model, name="model")
        mlflow.end_run()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # MLflow prints emoji; Windows console is cp1252
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    main()
