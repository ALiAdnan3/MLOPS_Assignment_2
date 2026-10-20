# VisionGuard – report notes (draft text per sub-task)

Name: Ali Adnan · Roll: 23L0839 · Project: 23L0839_A2_Ops
GitHub: https://github.com/ALiAdnan3/MLOPS_Assignment_2 · DagsHub: TODO · Live API URL: TODO
Graded commit: TODO
AI disclosure: Claude Code (Anthropic) was used as a tutor/pair-programmer to explain concepts, scaffold code and draft report text; all commands were run, checked and screenshotted by me.

---

## Task 1 – Data Understanding and Versioning

### 1.1 Environment
**What:** Created an isolated virtual environment (`python -m venv .venv`) and installed TensorFlow and the other
libraries there. Exact versions are pinned in `requirements.txt` and printed by `scripts/check_env.py`.

| Package | Version |
|---|---|
| Python | 3.13.15 (Windows 11) |
| tensorflow / keras | 2.21.0 / 3.15.1 |
| numpy / pandas / pillow | 2.5.4 / 3.0.6 / 13.0.0 |
| scikit-learn / matplotlib | 1.9.1 / 3.11.2 |
| mlflow / dagshub / dvc | 3.17.1 / 0.7.3 / 3.67.1 |
| keras-tuner / flask | 1.4.8 / 3.1.3 |

**GPU:** `tf.config.list_physical_devices("GPU")` returns `[]`. TensorFlow ≥ 2.11 has no GPU support on native Windows
(only via WSL2), and this laptop has no NVIDIA GPU, so all training ran on the CPU. This is fine for ~1 000 images
with a small CNN and a frozen/lightly fine-tuned MobileNetV2.

**Why a venv:** packages for this project cannot conflict with other projects, and anyone can rebuild the same environment from `requirements.txt`.

📸 Fig 1.1 – terminal running `python scripts\check_env.py` (prompt shows user + path).

### 1.2 Dataset audit
**What:** `src/audit_dataset.py` walks `data/raw/<class>/`, and for every file records class, file size, width, height,
colour mode, format and SHA-256 → `reports/audit.csv`. Each file is opened twice with PIL: `verify()` checks the file
structure, `load()` actually decodes the pixels – a file that fails either is counted as corrupt. Exact duplicates are
files with identical SHA-256 (identical bytes).

| Class | Files | Corrupt | Exact duplicates |
|---|---|---|---|
| AiArtData (AI) | 539 | 0 | 1 (image33.jpg = image34.jpg) |
| RealArt (real) | 436 | 0 | 0 |

| | width min / median / max | height min / median / max | file size KB min / median / max |
|---|---|---|---|
| AI | 225 / 1024 / 6000 | 135 / 800 / 4001 | 11.5 / 195 / 5494 |
| Real | 183 / 920 / 8495 | 158 / 720 / 7091 | 11.1 / 141 / 9956 |

Colour modes: mostly RGB, plus RGBA (35 AI / 14 real), palette `P` (14 / 9) and greyscale `L` (1 / 6) → every image is converted to RGB before training.
Formats by content: AI 429 JPEG + 110 PNG; Real 403 JPEG + 33 PNG.

**Findings:**
- **20 file extensions lie** (e.g. 11 `.png` files are really JPEG, the 3 `.webp` and the 1 `.gif` are JPEG). The audit uses the
  *content* format reported by PIL, not the extension – and the serving app must do the same.
- No duplicate appears in both classes (no contradicting labels). The one duplicate is removed before splitting so the same image cannot land in train and test (leakage).
- **Possible shortcuts:** 20 % of AI images are PNG vs 8 % of real; 14 % of AI images are exactly 512/1024/2048 px wide
  (native generator sizes) vs 3 % of real; 13 % of real files are tiny (< 30 KB, thumbnails) vs 4 % of AI. A model
  could learn format/compression/size instead of visual content. Resizing everything to 224×224 removes the size cue
  but not compression artefacts.

**Splits:** `src/make_splits.py` → stratified 70/15/15 with seed 42 (974 unique images):
train 681 (376 AI / 305 real), val 146 (81/65), test 147 (81/66). Saved as `data/splits/*.csv` (path,label,class) and committed to Git.

### 1.3 Visual EDA
- **Fig 1.3a sample grid** (`reports/figures/sample_grid.png`): AI images are mostly polished, highly saturated
  portraits, fantasy scenes and "perfect" animals. The *RealArt* class is not only photographs: it also contains human-made
  paintings and digital art. So the real task is *AI-generated vs human-made*, which is harder than AI vs photo.
- **Fig 1.3b width histogram:** AI widths pile up at 512, 1024 and ~2048 px – typical native output sizes of generators.
  Real images are more spread out and have a long tail up to 8 495 px (camera photos).
- **Fig 1.3c height histogram:** same pattern; AI heights concentrate at a few values (e.g. 768, 1024), real heights are more varied.
- **Fig 1.3d file-size histogram (log scale):** both classes overlap heavily around 100–1000 KB, but real images have a
  cluster of tiny 10–20 KB thumbnails. File size alone is therefore a weak but non-zero signal – something we do not want the model to rely on.
