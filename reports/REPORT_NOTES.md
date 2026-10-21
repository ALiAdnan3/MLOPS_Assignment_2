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

**Result of the fixed baseline:** best epoch 16 (val loss 0.572, val AUC 0.762); test accuracy 0.640, precision 0.726, recall 0.556, F1 0.629, ROC-AUC 0.759. 📸 Fig 2.2b `reports/figures/baseline_cnn_training_curves.png`.

### 2.3 Transfer learning (MobileNetV2)
**What:** `keras.applications.MobileNetV2(include_top=False, weights="imagenet")` – the ImageNet classifier is removed, leaving a 1280-channel feature extractor (2.26 M params). Our head: GlobalAveragePooling → Dropout 0.3 → Dense(1, sigmoid).
- **Stage 1 (frozen):** `base.trainable = False`, Adam lr 1e-3. Trainable: only `ai_probability/kernel` + `ai_probability/bias` = **1 281 parameters**. Freezing means the randomly initialised head cannot send large gradients back into (and wreck) the pretrained filters.
- **Stage 2 (fine-tune last block):** unfreeze `block_16_expand`, `block_16_depthwise`, `block_16_project`, `Conv_1`. BatchNorm layers stay frozen and the backbone is called with `training=False`, so BN keeps its ImageNet statistics (with batches of 32 they would otherwise be destroyed). Trainable: **880 321** of 2 259 265 params. Learning rate 10× smaller (1e-4) so the pretrained weights are only nudged, not overwritten. The model is re-compiled after changing `trainable`.
- **Preprocessing:** MobileNetV2's ImageNet weights were trained on pixels scaled to **[-1, 1]** (`mobilenet_v2.preprocess_input` = x/127.5 − 1). Feeding [0,1] or [0,255] would shift every activation away from what the filters expect, so the pretrained features would be meaningless. It is implemented as a `Rescaling(1/127.5, offset=-1)` layer *inside* the model, so it is saved with the model and serving cannot forget it. (EfficientNetB0 in run R8 has its rescaling built in and expects raw [0,255].)

Result (run `mobilenetv2_tl`): stage 1 plateaued at val AUC ≈ 0.82 (best stage-1 epoch 12); fine-tuning raised it to **0.87** and val loss 0.515 → 0.461. Test: accuracy 0.789, precision 0.891, recall 0.704, F1 0.786, ROC-AUC 0.872.
📸 Fig 2.3 `reports/figures/mobilenetv2_tl_training_curves.png` – both stages in one figure (dotted line = start of fine-tuning). Trainable-layer lists: `reports/trainable_layers.json`.

**What went wrong:** (1) In Keras 3 `weight.name` is only `kernel`/`bias`; switched to `weight.path` to get readable names. (2) A fresh ModelCheckpoint in stage 2 would overwrite the stage-1 checkpoint even if fine-tuning were worse; fixed with `initial_value_threshold` = best stage-1 val_loss.

### 2.4 Training discipline
- `tf.keras.utils.set_random_seed(42)` seeds Python, NumPy and TensorFlow; splits and shuffling also use seed 42.
- `EarlyStopping(monitor="val_loss", patience=5–10, restore_best_weights=True)` – stops when validation loss stops improving and rolls back to the best weights.
- `ModelCheckpoint(monitor="val_loss", save_best_only=True)` – the best model is on disk even if training crashes.
- `ReduceLROnPlateau(factor=0.3, patience=3)` – when val loss stalls, smaller steps can still find a lower minimum.
- Best epochs: baseline CNN **16**, MobileNetV2 **24** (8th epoch of fine-tuning), champion R4 **26**; marked by the red dashed line in every training-curve figure.
- Fixed seeds do not make CPU/oneDNN kernels and parallel data loading bit-for-bit deterministic. Evidence: R1 and R2 have identical settings but val AUC 0.819 vs 0.854.

### 2.5 Evaluation
**Which error is worse?** Positive class = AI-generated. A **false positive** (real photo called fake) accuses a real photographer, journalist or artist of faking: reputational damage and loss of trust in the tool. A false negative (fake called real) is also bad, but VisionGuard is a hint, not a gatekeeper, and a missed fake is the status quo without the tool. So **calling a real photo fake is the worse error** → we want high precision for the AI class.

**Threshold:** chosen on the **validation** set (never the test set) by maximising **F0.5** (β = 0.5 weights precision twice as much as recall) over thresholds 0.05–0.95. Results: CNN 0.67, MobileNetV2 0.72, champion R4 0.80. The test set was then evaluated **once** per model with that fixed threshold.

| Test set | Baseline CNN | MobileNetV2 TL | Champion R4 (MNV2, no aug) |
|---|---|---|---|
| Threshold (from val) | 0.67 | 0.72 | 0.80 |
| Accuracy | 0.640 | 0.789 | 0.714 |
| Precision | 0.726 | 0.891 | 0.868 |
| Recall | 0.556 | 0.704 | 0.568 |
| F1 | 0.629 | 0.786 | 0.687 |
| ROC-AUC | 0.759 | 0.872 | 0.860 |

📸 Figs: `reports/figures/*_confusion_matrix.png`, `*_roc_curve.png`. The high thresholds trade recall for precision exactly as intended: e.g. only 7 of 66 real test images were wrongly flagged by MobileNetV2 (confusion [[59, 7], [24, 57]]).

### 2.6 Error analysis (champion, `reports/figures/error_analysis_champion.png`)
All 10 most confident errors are AI images predicted "Real" – the error type we chose to tolerate. Hypotheses:
1. `Screen_Shot…png` – a *screenshot* of AI art with a Matrix-code overlay and Van-Gogh brush style; painterly texture resembles the human paintings in RealArt.
2. `240201-airec…jpg` – flat pop-art collage / article banner; no photographic texture for the model to judge.
3. `Bu69…jpg` – saturated HDR-like landscape; RealArt contains similar HDR travel photos.
4. `Nov20_17…jpg` – a real forest photo with an AI "pixel" overlay on half the image (composite) – arguably label noise.
5. `Midjourney-River.jpg` – soft photorealistic Midjourney landscape, re-compressed JPEG.
6. `ai-cartoon-portraits-01.jpg` – photoreal face next to a cartoon character on studio-white background; the photographic face dominates.
7. `Boris-Eldagsen-AI-art.jpg` – deliberately photorealistic "vintage photo" that famously won a photography prize; it fooled human judges too.
8. `Kai-painting-2023.png` – painterly brush strokes typical of human paintings in the real class.
9. `ai-generated-8067660_1280.jpg` – lush digital-painting landscape, very similar to stock illustrations.
10. `HD-wallpaper-ai-landscape…jpg` – anime-style wallpaper, only 49 KB: heavy JPEG compression destroys fine generator artefacts.
Themes: painting/illustration styles, photorealism, composites/banners (label noise from web scraping) and strong compression.

### Bonus – Grad-CAM (`reports/figures/gradcam_champion.png`)
Grad-CAM on MobileNetV2's last conv feature map (`out_relu`). For correct AI predictions the heat concentrates on the **main rendered subject** (castle, moon/mountain); for correct real predictions on the hand-painted canvas and on faces in a photo grid. Wrong predictions: *Girl with a Pearl Earring* (real painting) is called AI because the model focuses on the smooth, idealised face – the typical look of AI portraits; a black-and-white triptych is called AI from its high-contrast central face. **Conclusion:** the model looks at subject content and rendering style (smoothness, saturation, idealised faces), not at background noise or edges. It learned "looks like typical AI art" rather than low-level generator fingerprints, which explains both its accuracy and its failures on paintings and photoreal fakes.

### 2.7 Keras Tuner (`src/tuning.py`)
**Toy part** – Fashion-MNIST (15 000 train / 3 000 val), Flatten → Dense(`units`) → Dense(10). `RandomSearch`, 10 trials, 4 epochs each, objective `val_accuracy`. Search space: `units` 32–256 (step 32), `learning_rate` 1e-4–1e-1 (log scale). Best: **units 256, lr 0.0071 → val acc 0.860**.

|   trial |   learning_rate |   units |   val_accuracy |
|--------:|----------------:|--------:|---------------:|
|       5 |     0.00710272  |     256 |         0.86   |
|       9 |     0.00304412  |      96 |         0.8503 |
|       8 |     0.000470497 |     160 |         0.8447 |
|       1 |     0.000654073 |     128 |         0.8447 |
|       2 |     0.00113642  |     256 |         0.8413 |
|       4 |     0.000537716 |     128 |         0.837  |
|       6 |     0.000186646 |     256 |         0.8277 |
|       3 |     0.000158727 |     160 |         0.8153 |
|       7 |     0.000134183 |     256 |         0.8147 |
|       0 |     0.000130514 |     192 |         0.8123 |

Reading: every trial with lr < 2e-4 is in the bottom four (0.81–0.83) regardless of units; with 4 epochs a too-small LR simply has not converged. Units matter less.

**Apply** – `VisionGuardHyperModel` reproduces our two-stage recipe inside each trial: 4 epochs head-only on the frozen MobileNetV2, then unfreeze the last `unfreeze_layers` backbone layers and fine-tune 6 epochs with `fine_tune_lr`. Search space: `fine_tune_lr` 1e-5–1e-3 (log), `dropout` 0.1–0.5, `unfreeze_layers` ∈ {0, 10, 20, 40, 60}. RandomSearch, **10 trials**, objective **val AUC**.

|   trial |   fine_tune_lr |   dropout |   unfreeze_layers |   val_auc |
|--------:|---------------:|----------:|------------------:|----------:|
|       9 |    0.000233605 |       0.1 |                40 |    0.8931 |
|       5 |    1.21655e-05 |       0.1 |                60 |    0.8756 |
|       2 |    0.000124829 |       0.5 |                10 |    0.8518 |
|       3 |    3.06927e-05 |       0.1 |                20 |    0.8484 |
|       7 |    7.76745e-05 |       0.3 |                60 |    0.846  |
|       1 |    3.49744e-05 |       0.1 |                20 |    0.846  |
|       8 |    1.38554e-05 |       0.3 |                20 |    0.8267 |
|       6 |    4.1268e-05  |       0.3 |                10 |    0.8014 |
|       0 |    1e-05       |       0.4 |                 0 |    0.7774 |
|       4 |    0.000671871 |       0.5 |                40 |    0.7026 |

**Which setting mattered most:** the **fine-tune learning rate**, in interaction with how much is unfrozen. With 40 unfrozen layers, lr 2.3e-4 gave the best AUC (0.893) while lr 6.7e-4 gave the worst (0.703): a large step size on many unfrozen layers destroys the ImageNet features ("catastrophic forgetting"). Unfreezing nothing (trial 00) caps AUC at ≈0.78, confirming that adapting the top of the backbone is necessary. Dropout had no consistent effect (0.1 appears in the top trials, but it is confounded with the other two settings).

**Why the best validation score overfits the validation set:** every trial is scored on the same 146 validation images, and we keep the maximum of 10 noisy scores. The maximum is biased upward – part of the winner's advantage is luck on those particular 146 images (we measured ±0.035 AUC noise between identical runs R1/R2). The more trials, the more likely a configuration "wins" by chance. That is why the tuned score (0.893) must not be reported as the expected performance; an untouched test set (or cross-validation / a fresh hold-out) is needed for an honest estimate. Consequence for us: we did not replace the champion based on the tuner alone; the tuner's suggestion (unfreeze ≈40 layers, lr ≈2e-4, dropout 0.1) is a candidate for the next training run.
📸 Fig 2.7: terminal output of both tuner searches (tables in `reports/tuner_toy.csv`, `reports/tuner_visionguard.csv`).

---

## Task 3 – Experiment Tracking with MLflow and DagsHub

### 3.1 MLflow locally (`src/mlflow_toy.py --target local`)
Tiny Dense network (Flatten → Dense 64 → Dense 10 softmax) on 20 000 Fashion-MNIST images, 5 epochs, two runs: lr 0.001 vs 0.01. Logged: params (learning_rate, units, epochs, batch_size), per-epoch metrics (loss, accuracy, val_loss, val_accuracy via a Keras callback), one plot (`loss_curve.png` via `mlflow.log_figure`) and the model (`mlflow.tensorflow.log_model`). Backend: local SQLite `mlflow_local.db`; UI: `mlflow ui --backend-store-uri sqlite:///mlflow_local.db`. Result: lr 0.001 → val acc 0.840; lr 0.01 → 0.838 with lower train loss (0.355 vs 0.401), i.e. it starts to overfit sooner.
📸 Figs 3.1a–d: run list, run page, artifacts, compare view.

Definitions (own words):
- **Experiment** – a named group of all attempts at one problem (e.g. `23L0839_A2_Ops-VisionGuard`) so they can be compared side by side.
- **Run** – one execution of training code; records everything about that single attempt.
- **Parameter** – an input chosen before the run (learning rate, image size); logged once and cannot change.
- **Metric** – a number measured during/after the run (loss, AUC); can be logged per step/epoch to draw curves.
- **Artifact** – any file produced by the run (plots, model, split lists, requirements) stored with it.
- **Tag** – free-form metadata about the run (git commit, data version, seed, logging type) used for search and provenance.

What went wrong: the MLflow 3.17 UI opens in "GenAI" mode – switch to "Model training". Keras 3 `compile(opt, loss, ["accuracy"])` passes the metrics list positionally as `loss_weights` → `ValueError: dtype='string'`; fixed with keyword arguments.

### 3.2 DagsHub setup
DagsHub repo `ALiAdnan3/23L0839_A2_Ops` was created **connected to (mirroring) the GitHub repo** `ALiAdnan3/MLOPS_Assignment_2`. MLflow tracking address: `https://dagshub.com/ALiAdnan3/23L0839_A2_Ops.mlflow`. The access token lives in `.env` as `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` / `MLFLOW_TRACKING_URI`; MLflow reads these environment variables itself, so no secret appears in code. `python-dotenv` loads `.env`; `.gitignore` contains `.env` (`git check-ignore -v .env` → `.gitignore:7:.env`). DVC credentials are stored with `dvc remote modify --local`, i.e. in `.dvc/config.local`, which DVC git-ignores. The toy run was repeated against DagsHub (`--target dagshub`).
What went wrong: MLflow prints a 🏃 emoji at the end of a run; the Windows console (cp1252) crashed with `UnicodeEncodeError`. Fixed with `PYTHONUTF8=1` / `sys.stdout.reconfigure(encoding="utf-8")`; the half-finished run was deleted.
📸 Figs 3.2a–c: DagsHub project page, toy runs on DagsHub, `.env` ignored.

### 3.3 Instrumenting VisionGuard training (`src/train.py --mlflow manual|autolog`)
**Manual** (`MLflowEpochLogger` Keras callback + explicit calls): all CLI hyper-parameters (model, lr, fine-tune factor, unfreeze, dropout, augment, img size, seed …), per-epoch train/val loss, accuracy, precision, recall, AUC and learning rate (step counter continues across both stages), final test metrics, threshold, best epoch; tags `git_commit`, `dvc_data_version` (md5 from `data/raw.dvc`), `seed`.
**Autolog** (`mlflow.tensorflow.autolog()`): automatically records `fit()` arguments, optimizer config (`opt_*`), callback settings (EarlyStopping patience, monitor…), per-epoch metrics, `stopped_epoch`/`restored_epoch`, a model summary, TensorBoard logs and the model.

| | Manual | Autolog |
|---|---|---|
| Design params (augment, img_size, backbone, unfreeze, dropout) | ✅ | ❌ not `fit()` arguments – added by hand |
| Optimizer / callback internals | partial | ✅ very detailed |
| Two-stage training | ✅ one continuous curve | ❌ stage-2 params rejected with `INVALID_PARAMETER_VALUE` (a param cannot change inside a run); the error was swallowed, so the fine-tune LR is not recorded |
| Test metrics, threshold, confusion matrix, tags | ✅ | ❌ only what happens inside `fit()` |
| Effort | more code | one line |

Conclusion: autolog gives "free" low-level detail; manual logging is needed for everything outside `fit()` and for the business-relevant numbers. Best practice: autolog + manual additions.
What went wrong: the first autolog run crashed with `TBNotInstalledError` – MLflow's Keras autologging writes TensorBoard summaries but TF 2.21 no longer installs `tensorboard`; installed `tensorboard==2.21.0`.
📸 Figs 3.3a/b: R1 run page (params, metric curves, tags) and R2 autolog run page.

### 3.4 Artifacts
Every manual run has `evaluation/` (confusion_matrix.png, roc_curve.png, training_curves.png, summary.json, test_predictions.csv, trainable_layers.json), `splits/` (train/val/test.csv), `environment/requirements-freeze.txt` (full `pip freeze`), `keras_file/model.keras`, and the MLflow model `model` (logged with `mlflow.tensorflow.log_model`). 📸 Fig 3.4 artifacts tab of R4.

### 3.5 Experiment comparison
Nine runs, each differing from the reference R1 in **one** setting (`scripts/run_experiments.sh`, table `reports/experiment_comparison.csv`, chart `reports/figures/experiment_comparison.png`):

| Run | Change vs R1 | best ep | val loss | val AUC | val F0.5 | thr | test acc | test P | test R | test AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| R1 | reference: MobileNetV2, lr 1e-3, light aug, 224 px, last block | 23 | 0.517 | 0.819 | 0.797 | 0.63 | 0.742 | 0.853 | 0.642 | 0.834 |
| R2 | same as R1, autologging | 22 | 0.494 | 0.854 | 0.840 | 0.65 | 0.782 | 0.818 | 0.778 | 0.863 |
| R3 | lr 3e-4 | 30 | 0.532 | 0.815 | 0.825 | 0.75 | 0.762 | 0.897 | 0.642 | 0.848 |
| **R4** | **no augmentation** | 26 | **0.445** | **0.875** | **0.853** | 0.80 | 0.714 | 0.868 | 0.568 | 0.860 |
| R5 | strong augmentation | 23 | 0.452 | 0.859 | 0.842 | 0.75 | 0.748 | 0.867 | 0.642 | 0.846 |
| R6 | frozen backbone only | 16 | 0.589 | 0.767 | 0.761 | 0.46 | 0.728 | 0.753 | 0.753 | 0.808 |
| R7 | image size 160 | 23 | 0.500 | 0.840 | 0.821 | 0.68 | 0.769 | 0.822 | 0.741 | 0.837 |
| R8 | backbone EfficientNetB0 | 27 | 0.461 | 0.865 | 0.831 | 0.77 | 0.776 | 1.000 | 0.593 | 0.884 |
| R9 | baseline CNN | 16 | 0.552 | 0.774 | 0.753 | 0.47 | 0.673 | 0.667 | 0.815 | 0.742 |

**Choice: R4.** Selection uses **validation** metrics only – val ROC-AUC (threshold-free ranking quality), val F0.5 (our precision-weighted product metric) and val loss (calibration). R4 is best on all three. Test numbers are reported but not used for selection; picking the run with the best *test* score (R8) would turn the test set into a second validation set and make the reported test score optimistic.
**Honest caveats:** R1 and R2 have identical settings yet differ by 0.035 val AUC, so seed/ordering noise is about as large as most single-setting differences (R4, R5, R8 are statistically tied on 146 validation images). Robust findings: fine-tuning > frozen (R1 vs R6), transfer learning > CNN from scratch (R1 vs R9). R8 (EfficientNetB0, 0 false positives on test) is registered as `@challenger` for future evaluation on more data.
📸 Fig 3.5: DagsHub/MLflow compare view of the runs + our chart.

### 3.6 Model registry
`src/registry.py register` registered `23L0839_A2_Ops-visionguard` **v1** ← R1 (reference), **v2** ← R4 with alias **`@champion`**, **v3** ← R8 with alias `@challenger`; each version is tagged with its run name, threshold and val AUC. `scripts/predict_champion.py` loads `models:/23L0839_A2_Ops-visionguard@champion` from DagsHub and classifies one image (output: version 2, `Real`, confidence 90.9 %, P(AI)=0.091, threshold 0.8).
What went wrong: MLflow 3 logs models as "LoggedModels"; `register_model("runs:/<id>/model")` failed on DagsHub ("Unable to find a logged_model with artifact_path model"). Fixed by looking up the run's LoggedModel and registering its own `models:/m-…` URI.
**Getting the champion to the server without calling DagsHub per request:** `python -m src.registry fetch` downloads the `@champion` .keras file **once** at deploy time to `models/champion.keras` (+ `champion.json` with version and threshold). Docker Compose mounts it read-only; the cloud API image has it baked in. Promoting a new model = move the alias, re-run `fetch`, restart/redeploy – no code change.
📸 Fig 3.6a registry page (versions + aliases), 3.6b terminal output of `predict_champion.py`.

---

## Task 4 – Flask Web Application

### 4.1 Inference module (`visionguard/inference.py`)
`Predictor.__init__` loads the champion `.keras` file **once** (plus one warm-up call that builds the TF graph); `predict_bytes()` then reuses it for every request. Pre-processing is `visionguard.preprocess.load_image` – literally the same function the `tf.data` training pipeline calls – and the model-specific scaling layer lives inside the saved model. The result is a label, the **confidence of the chosen class** (P(AI) if the label is AI, otherwise 1 − P(AI)), P(AI), the threshold and the model name/version.

**Why not load per request:** loading the model took **3.9–8.2 s** (16 MB file, Keras deserialisation, graph build), while one prediction takes **0.3–0.5 s**. Loading per request would make every upload 10–20× slower, multiply memory use under concurrent requests and hit the disk constantly.
**Why preprocessing must match:** the network only learned the input distribution it was trained on. A different resize method, channel order (RGB vs BGR), scaling ([0,1] vs [-1,1]) or forgetting to drop the alpha channel silently shifts every pixel – the model still returns a confident number, just a wrong one ("training/serving skew"). Sharing one function and keeping the scaling inside the model removes that risk by construction.

Console demo (`scripts/inference_demo.py`):
```
model loaded once in 3.89s  (version v2@champion, threshold 0.8)
15G9ADdejiQlTQc0UxNt3Cg.png       truth=AI-generated -> AI-generated confidence=99.9%  P(AI)=0.999  (523 ms)
people-taking-selfie-together-…   truth=Real         -> Real         confidence=99.2%  P(AI)=0.008  (344 ms)
```
📸 Fig 4.1 terminal output.

### 4.2 Routes and validation (`webapp/app.py`)
Routes: `GET /` upload page, `POST /predict` → result page, `GET /history` (+ `/history/<id>`), `GET /health` (JSON).
Validation, in order:
1. A file must be present → 400 "No file selected".
2. **File type:** extension must be in jpg/jpeg/png/webp/gif/bmp → **415 "Unsupported file type"**.
3. **Size limit:** `MAX_CONTENT_LENGTH` = 5 MB (env `MAX_UPLOAD_MB`); Flask aborts larger bodies before reading them → **413 "File too large"**.
4. **Looks like an image but is not:** the bytes are actually decoded with the shared `load_image` (PIL) → **422 "Not a readable image"** (e.g. a text file renamed `.jpg`).
- File names are passed through `werkzeug.utils.secure_filename` (`../../etc/passwd.png` → `etc_passwd.png`) before they are logged or shown.
- **The upload never touches the disk:** `request.files["image"].read()` keeps the bytes in memory; the result page shows a preview as an inline base64 `data:` URI built from those bytes; only metadata + SHA-256 go to the database.
- Friendly HTML error pages for all of the above, for 404/405 and for 503 "Model unavailable".
Tests (`tests/test_webapp.py`, 7 passing) cover all three failure cases and the sanitising.
📸 Figs 4.2a–c: the three failure pages (wrong type, too large, undecodable).

### 4.3 HTML and CSS (`webapp/templates/`, `webapp/static/style.css`)
Jinja2 templates extending one `base.html` with a top menu **Upload · History · Health**. Upload page with a drop-zone file picker; result page with the image preview, verdict (red AI / green Real), confidence bar, P(AI) and threshold, model name/version, inference mode and latency, the DB row id/time, and an honest caveat ("statistical model… can be wrong for paintings, heavily edited photos, screenshots… treat as a hint, not proof"). History page reads the database table. Plain CSS, responsive at phone width.
📸 Figs 4.3a–c: upload page, a result page, history page.

### 4.4 Configuration and production server
All settings are read from environment variables in `webapp/config.py`: `INFERENCE_MODE`, `MODEL_PATH`, `MODEL_NAME`, `MODEL_VERSION`, `THRESHOLD`, `DATABASE_URL`, `MAX_UPLOAD_MB`, `MODEL_API_URL`, `MODEL_API_KEY`, `PORT`… The same code/image therefore runs locally (SQLite, local model), in Compose (Postgres, mounted model) and against the cloud API without code changes, and no secret is in the code.
The app is served by **waitress** (a production WSGI server, multi-threaded) with **debug off** (`/health` reports `"debug": false`). Why: Flask's built-in dev server is single-process and not hardened; debug mode shows full tracebacks with source and configuration to any visitor and enables the Werkzeug interactive debugger, which allows **arbitrary code execution** from the browser – it must never be reachable in production. Debug mode also auto-reloads, which would reload the model.
Health check response: `{"status":"ok","database":"ok","debug":false,"inference_mode":"local","model_name":"23L0839_A2_Ops-visionguard","model_version":"v2@champion","threshold":0.8}`
📸 Fig 4.4 `/health` in the browser.

---

## Task 5 – Database Trail (`webapp/db.py`)
SQLAlchemy model `predictions`, one row per successful request:

| Column | Answers |
|---|---|
| `id`, `created_at` (UTC, indexed) | **when** |
| `filename` (sanitised), `image_sha256` | which input (without storing the image) |
| `label`, `prob_ai`, `threshold` | **what was decided** and why |
| `confidence` | **how confidently** |
| `model_name`, `model_version`, `inference_mode` | **by which model** (local container or cloud API) |
| `latency_ms` | how fast |

The engine URL comes from `DATABASE_URL`: SQLite for local runs/tests, **PostgreSQL 16** in Docker Compose (`postgresql+psycopg2://…@db:5432/visionguard`). Tables are created at start-up (`create_all`); `pool_pre_ping` re-opens dropped connections. The `/predict` view inserts the row right after the model returns; `/history` shows the latest 200 rows (newest first) in the History menu; `/history/<id>` shows one. Because Postgres stores the data in the named volume `pgdata`, the history survives `docker compose down`/`up` and container re-creation (verified during 6.5: rows 1–2 were still there after the db container was recreated).
Example (psql inside the db container):
```
 id |          created_at           |    label     | conf  | model_version
  2 | 2026-10-20 19:59:47.327487+00 | Real         | 0.992 | v2@champion
  1 | 2026-10-20 19:59:46.755581+00 | AI-generated | 0.999 | v2@champion
```
📸 Fig 5.1 History page; Fig 5.2 psql query.

---

## Task 6 – Containerization with Docker and Docker Compose

### 6.1 Docker warm-up (`scripts/docker_warmup.ps1`, nginx:alpine)
Commands: `docker pull`, `docker run -d -p 8081:80 -v <folder>:/usr/share/nginx/html:ro`, `docker ps`, `docker logs`, `docker exec … sh -c …`, `docker stop`, `docker rm`, `docker images`, `docker rmi`.
Definitions (own words):
- **Image** – a read-only, versioned template (file system + metadata + start command) from which containers are created; like a class.
- **Container** – a running (or stopped) instance of an image with its own writable layer, process space and network; like an object.
- **Layer** – each Dockerfile instruction that changes files produces one read-only, content-addressed layer; images are stacks of layers that are cached and shared between images.
- **Volume** – storage managed by Docker (or a mounted host folder) that lives outside the container's writable layer, so data survives when the container is removed.
- **Port mapping** – `-p host:container` forwards traffic arriving on a host port into a port inside the container's isolated network (8081 → 80).
📸 Figs 6.1a–d (one per pause of the script).

### 6.2 Dockerfile for the web app (`Dockerfile`)
Order of instructions and why:
1. `FROM python:3.13-slim` – Debian slim base (~120 MB) instead of the full image (~1 GB); same Python as training.
2. `ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 …` – no .pyc clutter, logs flushed immediately (visible in `docker logs`), no pip cache in the image.
3. `COPY requirements-serve.txt` + `RUN pip install --no-cache-dir -r …` **before** copying the source. **Layer cache:** Docker reuses a layer if the instruction and its inputs are unchanged. Dependencies change rarely, code changes often – so editing `app.py` only rebuilds the last tiny COPY layers (a rebuild took **7 s**), instead of re-installing ~1.6 GB of packages. Copying the source first would invalidate the pip layer on every code change.
4. `RUN useradd --uid 10001 appuser` … `USER appuser` – the app runs **non-root**, so a compromised process cannot modify the image's files or install software. Proof: `docker run --rm --entrypoint id 23l0839_a2_ops-web` → `uid=10001(appuser) gid=10001(appuser)`.
5. `COPY visionguard/ webapp/` – only the code the server needs.
6. `HEALTHCHECK` – calls `/health` every 30 s; Docker/Compose mark the container `unhealthy` if the model or DB breaks.
7. `CMD ["python", "-m", "webapp.app"]` – exec form, so the process gets signals (clean shutdown).
`.dockerignore` excludes `.env`, `.git`, `.venv`, `data/`, `outputs/`, DVC cache, reports… → small build context and **no secrets in the image**; secrets are injected at run time as environment variables.
**TensorFlow package:** `tensorflow-cpu==2.21.0` – wheel 274 MB vs 573 MB for `tensorflow` (which bundles GPU kernels that a CPU-only container can never use). CPU inference of MobileNetV2 takes ~0.3 s per image, so a GPU is unnecessary. The serving image also leaves out everything that is only for training (MLflow, DVC, keras-tuner, matplotlib, scikit-learn) via a separate `requirements-serve.txt`.
📸 Fig 6.2 `Dockerfile` on GitHub + the `id` output.

### 6.3 Compose stack (`docker-compose.yml`)
- Two services, **web** and **db** (postgres:16-alpine), on Compose's default internal network; web reaches the database by **service name** (`@db:5432` in `DATABASE_URL`).
- **Named volume** `pgdata` → `/var/lib/postgresql/data`.
- **Readiness:** db has a `pg_isready` health check; web uses `depends_on: db: condition: service_healthy`, so it starts only once Postgres accepts connections (log: `db-1 Waiting → Healthy → web-1 Starting`).
- **No published DB port:** db has no `ports:` section → `docker compose ps` shows `5432/tcp` without a host mapping, and a TCP connect to `127.0.0.1:5432` from the host fails ("port 5432 closed from host"). Only web publishes 8000.
- **Secrets:** `POSTGRES_USER/PASSWORD/DB`, thresholds etc. are interpolated from the untracked `.env`; `.env.example` documents every variable with placeholder values.
- **Model file: mounted**, read-only: `./models:/app/models:ro`. `python -m src.registry fetch` places the `@champion` from the DagsHub registry in `./models` once. Justification: the model changes independently of the code – promoting a new champion is "fetch + `docker compose restart web`" without rebuilding a 1.6 GB image, and the same image can serve any model version; `:ro` prevents the app from altering it. (For the cloud API the opposite choice – baking it in – is made, see Task 7.)
📸 Figs 6.3a compose file on GitHub, 6.3b `docker compose ps` (both healthy), 6.3c port 5432 closed.
