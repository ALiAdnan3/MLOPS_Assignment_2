"""Task 1.3 - Visual EDA. Reads reports/audit.csv (from audit_dataset.py).

Figures written to reports/figures/:
  sample_grid.png       8 random images per class
  hist_width.png, hist_height.png, hist_file_size.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

FIG_DIR = Path("reports/figures")
SEED = 42
N_PER_CLASS = 8


def sample_grid(df):
    classes = sorted(df["class"].unique())
    fig, axes = plt.subplots(len(classes), N_PER_CLASS, figsize=(2.2 * N_PER_CLASS, 2.6 * len(classes)))
    for i, c in enumerate(classes):
        sample = df[df["class"] == c].sample(N_PER_CLASS, random_state=SEED)
        for j, (_, r) in enumerate(sample.iterrows()):
            ax = axes[i, j]
            with Image.open(r["path"]) as im:
                ax.imshow(im.convert("RGB"))
            ax.set_title(f"{r['width']}x{r['height']}", fontsize=8)
            ax.axis("off")
        axes[i, 0].text(-0.1, 0.5, c, transform=axes[i, 0].transAxes, rotation=90,
                        va="center", ha="right", fontsize=12, fontweight="bold")
    fig.suptitle("Random samples per class (title = width x height)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sample_grid.png", dpi=120)
    plt.close(fig)


def histogram(df, column, label, fname, log=False):
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [df[df["class"] == c][column] for c in sorted(df["class"].unique())]
    bins = np.logspace(np.log10(max(1, df[column].min())), np.log10(df[column].max()), 40) if log else 40
    for c, d in zip(sorted(df["class"].unique()), data):
        ax.hist(d, bins=bins, alpha=0.55, label=f"{c} (median {d.median():.0f})")
    if log:
        ax.set_xscale("log")
    ax.set_xlabel(label)
    ax.set_ylabel("Number of images")
    ax.set_title(f"Distribution of {label} per class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=120)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv("reports/audit.csv")
    df = df[df["readable"] & ~df["is_duplicate"]]
    sample_grid(df)
    histogram(df, "width", "width (px)", "hist_width.png")
    histogram(df, "height", "height (px)", "hist_height.png")
    histogram(df, "file_size_kb", "file size (KB, log scale)", "hist_file_size.png", log=True)
    print("Saved figures to", FIG_DIR)


if __name__ == "__main__":
    main()
