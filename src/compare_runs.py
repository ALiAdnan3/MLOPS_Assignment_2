"""Task 3.5 - one comparison table + chart of all experiment runs (from outputs/R*/summary.json)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

CHANGE = {"R1": "reference (MobileNetV2, lr 1e-3, light aug, 224, last block)", "R2": "same as R1, autologging",
          "R3": "lr 3e-4", "R4": "no augmentation", "R5": "strong augmentation", "R6": "frozen backbone only",
          "R7": "image size 160", "R8": "backbone EfficientNetB0", "R9": "baseline CNN from scratch"}
rows = []
for f in sorted(Path("outputs").glob("R*/summary.json")):
    s = json.loads(f.read_text())
    rows.append({"run": s["run_name"], "change vs R1": CHANGE[s["run_name"][:2]], "best_epoch": s["best_epoch"],
                 "val_loss": s["best_val_loss"], "val_auc": s["val_auc"], "val_F0.5": s["val_fbeta0.5"],
                 "threshold": s["threshold"], "test_acc": s["test_accuracy"], "test_precision": s["test_precision"],
                 "test_recall": s["test_recall"], "test_f1": s["test_f1"], "test_auc": s["test_roc_auc"]})
df = pd.DataFrame(rows)
df.to_csv("reports/experiment_comparison.csv", index=False)
print(df.drop(columns="change vs R1").to_markdown(index=False))

fig, ax = plt.subplots(figsize=(11, 4.5))
x = range(len(df))
w = 0.27
ax.bar([i - w for i in x], df["val_auc"], w, label="val ROC-AUC (selection)")
ax.bar(list(x), df["val_F0.5"], w, label="val F0.5 (selection)")
ax.bar([i + w for i in x], df["test_auc"], w, label="test ROC-AUC (reported only)", alpha=0.6)
ax.set_xticks(list(x), [r.replace("_", "\n", 1) for r in df["run"]], fontsize=8)
ax.set_ylim(0.6, 0.95)
ax.axhline(df["val_auc"].max(), color="gray", ls=":", lw=0.8)
ax.set_title("23L0839_A2_Ops VisionGuard - experiment comparison (champion chosen on validation metrics)")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig("reports/figures/experiment_comparison.png", dpi=120)
