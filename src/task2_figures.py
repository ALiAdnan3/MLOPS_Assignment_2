"""Evidence for Task 2.1 (augmented batch figure) and 2.2 (per-layer table)."""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402

from visionguard.data import make_dataset  # noqa: E402
from visionguard.models import build_baseline_cnn  # noqa: E402

FIG = Path("reports/figures")


def augmented_batch():
    tf.keras.utils.set_random_seed(42)
    plain = make_dataset("data/splits/train.csv", training=True, batch_size=8, augment="none")
    aug = make_dataset("data/splits/train.csv", training=True, batch_size=8, augment="light")
    # same seed + same shuffle -> same 8 images, with and without augmentation
    tf.keras.utils.set_random_seed(42)
    x0, y = next(iter(plain))
    tf.keras.utils.set_random_seed(42)
    x1, _ = next(iter(aug))
    fig, axes = plt.subplots(2, 8, figsize=(18, 5))
    for i in range(8):
        for row, x in enumerate([x0, x1]):
            axes[row, i].imshow(x[i].numpy().astype("uint8"))
            axes[row, i].axis("off")
        axes[0, i].set_title("AI" if y[i] == 1 else "Real")
    axes[0, 0].text(-0.15, 0.5, "decoded +\nresized", transform=axes[0, 0].transAxes, ha="right", va="center")
    axes[1, 0].text(-0.15, 0.5, "augmented\n(train only)", transform=axes[1, 0].transAxes, ha="right", va="center")
    fig.suptitle("One training batch (224x224): before vs after random flip / rotation / zoom")
    fig.tight_layout()
    fig.savefig(FIG / "augmented_batch.png", dpi=110)


def layer_table():
    model = build_baseline_cnn()
    rows = [{"layer": ly.name, "type": type(ly).__name__, "output_shape": str(ly.output.shape),
             "params": ly.count_params()} for ly in model.layers]
    df = pd.DataFrame(rows)
    df.to_csv("reports/cnn_layers.csv", index=False)
    print(df.to_markdown(index=False))
    print("Total params:", model.count_params(),
          "| trainable:", sum(int(tf.size(w)) for w in model.trainable_weights))
    model.summary()


if __name__ == "__main__":
    augmented_batch()
    layer_table()
