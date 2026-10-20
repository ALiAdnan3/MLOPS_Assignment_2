"""Task 2.6 - grid of the 10 most confident misclassified test images.

Usage: python -m src.error_analysis outputs/<run_name>
Writes <run>/errors.png and <run>/errors.csv (with size/format/mode from the audit table).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

N = 10


def main(run_dir):
    run = Path(run_dir)
    t = json.loads((run / "summary.json").read_text())["threshold"]
    pred = pd.read_csv(run / "test_predictions.csv")
    audit = pd.read_csv("reports/audit.csv")[["path", "width", "height", "file_size_kb", "format", "mode"]]
    wrong = pred[pred.label != pred.pred].merge(audit, on="path")
    wrong["confidence_of_error"] = (wrong.prob_ai - t).abs()
    wrong = wrong.sort_values("confidence_of_error", ascending=False).head(N)
    wrong.to_csv(run / "errors.csv", index=False)

    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    for ax, (i, r) in zip(axes.ravel(), enumerate(wrong.itertuples(), 1)):
        with Image.open(r.path) as im:
            ax.imshow(im.convert("RGB"))
        truth = "AI" if r.label == 1 else "Real"
        ax.set_title(f"#{i} true={truth}  p(AI)={r.prob_ai:.2f}\n{r.width}x{r.height} {r.format} {r.file_size_kb:.0f}KB",
                     fontsize=9, color="darkred")
        ax.axis("off")
    fig.suptitle(f"{run.name}: 10 most confident test errors (threshold {t})")
    fig.tight_layout()
    fig.savefig(run / "errors.png", dpi=110)
    print(wrong[["path", "label", "prob_ai", "width", "height", "format", "file_size_kb"]].to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
