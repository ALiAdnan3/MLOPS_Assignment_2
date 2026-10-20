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

---

## Task 2 – Model Development with TensorFlow/Keras

### 2.1 Input pipeline (`visionguard/data.py`, `visionguard/preprocess.py`)
Stages, in order:
1. **Read split list** – `pd.read_csv("data/splits/train.csv")` → `tf.data.Dataset.from_tensor_slices((paths, labels))`. Using the saved lists (not a random split at load time) means every run and every model sees exactly the same images.
2. **Decode + resize** – `tf.numpy_function` calls `load_image_path`, which uses PIL to decode, takes frame 0 of animated files, converts P/RGBA/L/CMYK → RGB and resizes to **224×224** (bilinear). PIL is used instead of `tf.io.decode_image` because TF cannot decode WEBP and handles palette/CMYK images poorly; the *same* function is imported by the Flask app, so serving preprocessing is identical by construction. Runs in parallel (`num_parallel_calls=AUTOTUNE`).
3. **Cache** – decoded 224×224 images are kept in RAM after the first epoch (~100 MB for train), so JPEG decoding happens only once.
4. **Shuffle** (train only) – full-buffer shuffle with seed 42, reshuffled every epoch.
5. **Batch** – 32 images per batch.
6. **Augment** (train only) – Keras layers `RandomFlip("horizontal")`, `RandomRotation(0.05)` (±18°), `RandomZoom(0.1)`. Validation/test are never augmented, so their scores measure the real images. Augmentation is deliberately mild: blur, noise, JPEG re-compression or strong colour jitter would erase exactly the fine texture/colour artefacts that betray generated images. (A "strong" option with contrast/brightness/translation exists for the comparison in 3.5.)
7. **Prefetch** – `prefetch(AUTOTUNE)` prepares the next batch on the CPU while the model trains on the current one.

**Image size 224:** the native input size of MobileNetV2/EfficientNetB0 ImageNet weights; 224² is also small enough for CPU training of ~700 images. Smaller sizes (e.g. 160) are tested in 3.5.

**Aspect ratio:** images are *squashed* to a square, so aspect ratio is **not** preserved (a 1500×750 banner is compressed 2× horizontally). Alternatives: center-crop (loses edges – for collages and wide scenes the cropped part may be the only AI-looking region) or pad with black bars (keeps geometry, but bar size would encode the original aspect ratio, which differs between classes = a shortcut). Squashing keeps all content and applies the same distortion at training and serving time. Aspect-ratio (w/h) distributions of both classes are almost identical (median 1.5, quartiles 1.0–1.78 vs 1.0–1.75), so the distortion is similar for both classes and is not a label shortcut.

📸 Fig 2.1 – `reports/figures/augmented_batch.png`: top row = decoded + resized, bottom = the same images after augmentation.

### 2.2 Baseline CNN (`build_baseline_cnn` in `visionguard/models.py`)
Architecture: `Rescaling(1/255)` → 4 × [Conv2D 3×3 (32/64/128/256 filters) → BatchNorm → ReLU → MaxPool 2×2] → GlobalAveragePooling → Dropout 0.4 → Dense 64 ReLU → Dense 1 sigmoid. ~406 k parameters.

Why each layer:
- **Rescaling 1/255** – puts pixels in [0,1] so gradients are well-scaled; inside the model so serving cannot forget it.
- **Conv2D 3×3** – learns local patterns (edges → textures → parts); doubling filters each block while the map halves keeps compute roughly constant and lets deeper layers represent more complex patterns. `use_bias=False` because BatchNorm adds its own shift.
- **BatchNormalization** – normalises activations per channel; faster, more stable training and mild regularisation. Momentum set to 0.9 (see "what went wrong").
- **ReLU** – cheap non-linearity without vanishing gradients.
- **MaxPooling 2×2** – halves resolution, adds small translation invariance, grows the receptive field.
- **GlobalAveragePooling** – averages each feature map to one number: far fewer parameters than Flatten (256 vs 200 k inputs to the dense layer) → less overfitting on 681 images.
- **Dropout 0.4** – randomly drops 40 % of features during training so the head cannot rely on a few features.
- **Dense 64 + Dense 1 sigmoid** – small classifier head; sigmoid outputs P(AI-generated) in [0,1].

**Loss:** binary cross-entropy – the negative log-likelihood of a Bernoulli label, the correct pairing with one sigmoid output; it penalises confident wrong answers heavily. **Optimizer:** Adam (lr 1e-3) – per-parameter adaptive steps, works well without much tuning. **Metrics:** accuracy (easy to read; classes are only mildly imbalanced 55/45), precision and recall (the two error types matter differently, see 2.5) and AUC (threshold-independent ranking quality).

📸 Fig 2.2 – `model.summary()` screenshot; table in `reports/cnn_layers.csv`.

**What went wrong (first CNN run):** training loss fell 0.76 → 0.60 but validation accuracy was stuck at exactly 0.4452 (= 65/146, every validation image predicted "real") and val AUC 0.41; early stopping ended the run at epoch 8. Cause: BatchNorm uses batch statistics in training but *moving averages* at validation time; with the default momentum 0.99 and only 22 batches/epoch, the moving averages were still near their initial values, so validation activations were mis-scaled. Fix: BN momentum 0.9, longer patience (10) and LR-reduction patience 3. Log kept in `outputs/baseline_cnn_failed_bn/`.
