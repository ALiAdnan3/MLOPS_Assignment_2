"""Task 1.2 - Dataset audit.

Walks data/raw/<class>/ and records, for every file:
class, file name, file size, width, height, colour mode, format, SHA-256.
Flags unreadable (corrupt) files and exact duplicates (same SHA-256).

Outputs:
  reports/audit.csv          one row per file
  reports/audit_summary.txt  counts / stats printed for the report
"""
import hashlib
import os
from pathlib import Path

import pandas as pd
from PIL import Image

DATA_DIR = Path("data/raw")
OUT_DIR = Path("reports")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect(path: Path) -> dict:
    """Open the image twice: verify() checks the file structure, load() decodes pixels."""
    row = {"width": None, "height": None, "mode": None, "format": None, "readable": True, "error": ""}
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
            row.update(width=im.width, height=im.height, mode=im.mode, format=im.format)
    except Exception as e:  # any decode failure = corrupt / not an image
        row.update(readable=False, error=f"{type(e).__name__}: {e}"[:200])
    return row


def main():
    rows = []
    for class_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        for path in sorted(class_dir.iterdir()):
            if not path.is_file():
                continue
            row = {
                "class": class_dir.name,
                "path": path.as_posix(),
                "file": path.name,
                "extension": path.suffix.lower().lstrip("."),
                "file_size_kb": round(os.path.getsize(path) / 1024, 2),
                "sha256": sha256_of(path),
            }
            row.update(inspect(path))
            rows.append(row)

    df = pd.DataFrame(rows)
    # Exact duplicates: identical bytes -> identical hash. Keep first, mark the rest.
    df["is_duplicate"] = df.duplicated("sha256", keep="first")
    dup_groups = df[df.duplicated("sha256", keep=False)].groupby("sha256")["class"].agg(lambda s: sorted(set(s)))
    cross_class_dups = int(sum(len(c) > 1 for c in dup_groups))

    OUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUT_DIR / "audit.csv", index=False)

    ok = df[df.readable]
    lines = []
    lines.append("=== Files per class ===")
    lines.append(df.groupby("class").agg(total=("file", "count"),
                                         readable=("readable", "sum"),
                                         duplicates=("is_duplicate", "sum")).to_string())
    lines.append("\n=== Width / height / file size (readable files) ===")
    stats = ok.groupby("class")[["width", "height", "file_size_kb"]].agg(["min", "median", "max"])
    lines.append(stats.to_string())
    lines.append("\n=== Overall ===")
    lines.append(ok[["width", "height", "file_size_kb"]].agg(["min", "median", "max"]).to_string())
    lines.append("\n=== Colour modes ===")
    lines.append(pd.crosstab(ok["class"], ok["mode"]).to_string())
    lines.append("\n=== Formats ===")
    lines.append(pd.crosstab(ok["class"], ok["format"]).to_string())
    lines.append(f"\nCorrupt / unreadable files: {int((~df.readable).sum())}")
    for _, r in df[~df.readable].iterrows():
        lines.append(f"  - {r['path']}  ({r['error']})")
    lines.append(f"Exact duplicate files (extra copies): {int(df.is_duplicate.sum())}")
    lines.append(f"Duplicate groups spanning BOTH classes: {cross_class_dups}")

    text = "\n".join(lines)
    print(text)
    (OUT_DIR / "audit_summary.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
