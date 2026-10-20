"""Bonus - Grad-CAM heat maps for 4 correct and 4 wrong test predictions.

Usage: python -m src.gradcam outputs/<run_name>
Grad-CAM: gradient of the predicted-class score w.r.t. the last convolutional
feature map, averaged per channel = channel importance; weighted sum of the
channels (ReLU) = where in the image the evidence came from.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402

from visionguard.preprocess import load_image_path  # noqa: E402


def feature_and_head(model):
    """Return f(x) -> (last conv feature map, p(AI)) that is differentiable w.r.t. the feature map."""
    names = [layer.name for layer in model.layers]
    if "backbone" in names:                                   # transfer model: nested backbone
        pre = model.get_layer("preprocess") if "preprocess" in names else (lambda t: t)
        base, gap, out = model.get_layer("backbone"), model.get_layer("gap"), model.get_layer("ai_probability")

        def run(x, tape):
            f = base(pre(x), training=False)
            tape.watch(f)
            return f, out(gap(f))
    else:                                                     # baseline CNN: plain functional model
        sub = tf.keras.Model(model.inputs, [model.get_layer("block4_relu").output, model.output])

        def run(x, tape):
            return sub(x, training=False)
    return run


def gradcam(run, x, predicted_ai):
    with tf.GradientTape() as tape:
        f, p = run(x, tape)
        score = p[:, 0] if predicted_ai else 1 - p[:, 0]
    g = tape.gradient(score, f)[0]
    weights = tf.reduce_mean(g, axis=(0, 1))
    cam = tf.nn.relu(tf.reduce_sum(f[0] * weights, axis=-1)).numpy()
    cam = cam / (cam.max() + 1e-8)
    return tf.image.resize(cam[..., None], x.shape[1:3]).numpy()[..., 0]


def main(run_dir):
    run_dir = Path(run_dir)
    model = tf.keras.models.load_model(run_dir / "model.keras")
    t = json.loads((run_dir / "summary.json").read_text())["threshold"]
    pred = pd.read_csv(run_dir / "test_predictions.csv")
    correct = pred[pred.label == pred.pred]
    wrong = pred[pred.label != pred.pred]
    # 2 AI + 2 real for each group where possible
    pick = pd.concat([correct[correct.label == 1].head(2), correct[correct.label == 0].head(2),
                      wrong[wrong.label == 1].head(2), wrong[wrong.label == 0].head(2)])
    pick = pd.concat([pick, wrong.drop(pick.index, errors="ignore")]).drop_duplicates("path")
    pick = pd.concat([pick[pick.label == pick.pred].head(4), pick[pick.label != pick.pred].head(4)])

    run = feature_and_head(model)
    fig, axes = plt.subplots(2, 8, figsize=(22, 6))
    for j, r in enumerate(pick.itertuples()):
        img = load_image_path(r.path)
        cam = gradcam(run, tf.constant(img[None].astype("float32")), r.prob_ai >= t)
        ok = "CORRECT" if r.label == r.pred else "WRONG"
        axes[0, j].imshow(img)
        axes[0, j].set_title(f"{ok}\ntrue={'AI' if r.label else 'Real'} p(AI)={r.prob_ai:.2f}", fontsize=9,
                             color="green" if ok == "CORRECT" else "darkred")
        axes[1, j].imshow(img)
        axes[1, j].imshow(cam, cmap="jet", alpha=0.45)
        for a in axes[:, j]:
            a.axis("off")
    fig.suptitle(f"Grad-CAM ({run_dir.name}): top = input, bottom = regions driving the predicted class")
    fig.tight_layout()
    fig.savefig(run_dir / "gradcam.png", dpi=100)
    print("saved", run_dir / "gradcam.png")


if __name__ == "__main__":
    main(sys.argv[1])
