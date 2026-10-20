"""Create stratified train / val / test split lists from the audit table.

- drops unreadable files
- drops exact duplicates (keeps one copy); if the same bytes appear in BOTH
  classes the label is ambiguous, so every copy is dropped
- 70 / 15 / 15 stratified split with a fixed seed

Outputs data/splits/{train,val,test}.csv with columns: path,label,class
label: 1 = AI-generated (AiArtData), 0 = real (RealArt)
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
CLASS_TO_LABEL = {"AiArtData": 1, "RealArt": 0}
OUT = Path("data/splits")


def main():
    df = pd.read_csv("reports/audit.csv")
    df = df[df["readable"]]
    n_classes_per_hash = df.groupby("sha256")["class"].transform("nunique")
    ambiguous = df[n_classes_per_hash > 1]
    df = df[n_classes_per_hash == 1].drop_duplicates("sha256", keep="first")
    df = df.assign(label=df["class"].map(CLASS_TO_LABEL))[["path", "label", "class"]]

    train, rest = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=SEED)

    OUT.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.sort_values("path").to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name:5s} {len(part):4d}  AI={int(part.label.sum()):3d}  real={int((1 - part.label).sum()):3d}")
    print(f"Dropped {len(ambiguous)} files whose bytes appear in both classes")


if __name__ == "__main__":
    main()
