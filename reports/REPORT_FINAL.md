## Introduction
In this assignment I built **VisionGuard**, a small web service that tells whether an uploaded image is AI-generated or real. I trained two TensorFlow/Keras models, tracked every experiment with MLflow on DagsHub, versioned the dataset with DVC, served the best model through a Flask web app that logs every request into PostgreSQL, packaged everything with Docker Compose, set up GitHub Actions CI, and finally deployed the model as an API on Microsoft Azure that my local Flask app calls.

The dataset is *AI Generated Images vs Real Images* (Kaggle). The label of an image is the folder it is in: `AiArtData` (AI-generated) or `RealArt` (real). Everything in this project is named **23L0839_A2_Ops**.

| Item | Link / value |
|---|---|
| GitHub repository | https://github.com/ALiAdnan3/MLOPS_Assignment_2 |
| DagsHub project (code, data, experiments, registry) | https://dagshub.com/ALiAdnan3/23L0839_A2_Ops |
| Live model API | https://vg-23l0839-a2-ops-api.gentleglacier-7f267617.eastus.azurecontainerapps.io |
| Final model | `23L0839_A2_Ops-visionguard` version 2, alias `@champion` (MobileNetV2, threshold 0.80) |

## Task 1 – Data Understanding and Versioning

### 1.1 Environment
**What I did:** I created an isolated virtual environment with `python -m venv .venv`, activated it and installed TensorFlow and all the other libraries inside it. I pinned the exact versions in `requirements.txt` and wrote a small script `scripts/check_env.py` that prints the versions and asks TensorFlow which GPUs it can see.

**Why:** a virtual environment keeps this project's packages separate from everything else on my laptop, so an upgrade for another project cannot break this one, and anyone can rebuild the same environment with `pip install -r requirements.txt`.

**Result:** Python 3.13.15, TensorFlow 2.21.0, Keras 3.15.1, NumPy 2.5.4, pandas 3.0.6, Pillow 13.0.0, scikit-learn 1.9.1, MLflow 3.17.1, DagsHub 0.7.3, DVC 3.67.1, Keras Tuner 1.4.8, Flask 3.1.3. **TensorFlow does not see a GPU.** I found out that TensorFlow stopped supporting GPUs on native Windows after version 2.10 (it only works through WSL2), and my laptop has no NVIDIA card anyway, so I trained everything on the CPU. With only ~1 000 images this was fine – one MobileNetV2 run took about 10 minutes.

![label=env; Library versions printed by scripts/check_env.py inside the virtual environment](screenshots/Task1_img_1.png)

### 1.2 Dataset audit
**What I did:** I wrote `src/audit_dataset.py`, which walks through `data/raw/<class>/` and for every file records the class, file size, width, height, colour mode, format and a SHA-256 hash. Each file is opened twice with Pillow: `verify()` checks the file structure and `load()` actually decodes the pixels, so a broken file fails one of the two and is counted as corrupt. Two files with the same SHA-256 have exactly the same bytes, so that is how I found exact duplicates. The full table is saved as `reports/audit.csv`.

| Class | Files | Corrupt | Exact duplicates |
|---|---|---|---|
| AiArtData (AI) | 539 | 0 | 1 (image33.jpg = image34.jpg) |
| RealArt (real) | 436 | 0 | 0 |

| | Width min / median / max | Height min / median / max | File size KB min / median / max |
|---|---|---|---|
| AI | 225 / 1024 / 6000 | 135 / 800 / 4001 | 11.5 / 195 / 5494 |
| Real | 183 / 920 / 8495 | 158 / 720 / 7091 | 11.1 / 141 / 9956 |

**Things I noticed:**
- **20 file extensions are wrong.** For example 11 `.png` files are really JPEGs and the 3 `.webp` files and the one `.gif` are also JPEGs. So I never trust the extension – I use the format that Pillow detects from the content (the web app does the same).
- Colour modes are mostly RGB, but there are also RGBA (35 AI / 14 real), palette `P` (14 / 9) and greyscale `L` (1 / 6) images, so every image has to be converted to RGB before training.
- No duplicate appears in both classes, so there are no contradicting labels. I removed the one duplicate before splitting so that the same image cannot end up in both train and test.
- **Possible shortcuts:** 20 % of AI images are PNG but only 8 % of real ones; 14 % of AI images are exactly 512, 1024 or 2048 pixels wide (typical generator output sizes) against 3 % of real ones; and 13 % of real files are tiny thumbnails under 30 KB. A model could learn "PNG" or "1024 px" instead of what the image looks like. Resizing everything to 224×224 removes the size hint but not compression differences.

**Splits:** `src/make_splits.py` makes a stratified 70 / 15 / 15 split with seed 42 from the 974 unique images: train 681 (376 AI / 305 real), validation 146 (81 / 65), test 147 (81 / 66). The lists are saved in `data/splits/` and committed to Git, so every model sees exactly the same images.

### 1.3 Visual EDA
I wrote `src/eda.py` to draw a grid of random samples and histograms of width, height and file size per class.

![label=grid; Eight random images per class (the title is width × height)](reports/figures/sample_grid.png)

In {fig:grid} the AI images are mostly very polished, saturated portraits, fantasy scenes and "too perfect" animals. The surprising part is that the *RealArt* class is not only photos – it also contains human paintings and digital art (for example the blue face and the ballerina). So the real task is "AI-generated vs made by a human", which is harder than "AI vs photo".

![label=hw; Width distribution per class](reports/figures/hist_width.png)

In {fig:hw} the AI widths pile up at 512, 1024 and around 2048 pixels, which are the native output sizes of image generators. Real images are spread out more and have a long tail up to 8 495 px from camera photos.

![label=hh; Height distribution per class](reports/figures/hist_height.png)

{fig:hh} shows the same pattern for height: the AI images sit on a few fixed values (like 768 and 1024), while the real heights are much more varied.

![label=hs; File size distribution per class (log scale)](reports/figures/hist_file_size.png)

In {fig:hs} both classes overlap a lot between 100 KB and 1 MB, but the real class has a cluster of tiny 10–20 KB thumbnails. File size alone is a weak signal, and it is not something I want the model to rely on.

### 1.4 DVC (data version control)
**Toy part.** I wrote `scripts/dvc_toy_lab.ps1` so that every step is printed before it runs. In a new empty folder `D:\dvc_toy_23L0839` I ran `git init` and `dvc init`, created a small `students.csv` with two rows and tracked it with `dvc add students.csv`. DVC moved the real file into its cache, wrote a small pointer file `students.csv.dvc` and added `students.csv` to `.gitignore`, so Git only stores the pointer.

![label=dvc1; Toy: git init, dvc init and the small CSV](screenshots/Task1_img_2.png)

![label=dvc2; Toy: the pointer file students.csv.dvc (md5, size, path) and the first commit](screenshots/Task1_img_3.png)

Then I added a third row, ran `dvc add` again (the MD5 in the pointer changed) and made a second commit. To go back to the old version I checked out the old pointer with `git checkout HEAD~1 -- students.csv.dvc` and ran `dvc checkout`, which put the 2-row version of the data back.

![label=dvc3; Toy: version 2, git log with both commits, and returning to version 1](screenshots/Task1_img_4.png)

Next I configured DagsHub as remote storage (`dvc remote add -d origin s3://dvc` with the DagsHub endpoint). The access token was saved with `dvc remote modify --local`, which writes it into `.dvc/config.local` – a file DVC never commits. I pushed the Git commits to DagsHub and the data with `dvc push`.

![label=dvc4; Toy: back to version 2, DagsHub remote configured and git push](screenshots/Task1_img_5.png)

To prove the remote works I deleted `students.csv` **and** the whole `.dvc/cache` folder (like a fresh computer) and ran `dvc pull`. The file came back with all three rows.

![label=dvc5; Toy: local data and cache deleted, dvc pull brings the file back](screenshots/Task1_img_6.png)

![label=dvc6; The toy repository on DagsHub with its commits and the DVC-tracked students.csv](screenshots/Task1_img_7.png)

**Apply part.** I tracked the real dataset the same way: `dvc add data/raw` created `data/raw.dvc`, which I committed to Git, and `dvc push` uploaded all 975 images (480 MB) to the DagsHub storage of my main project. `dvc status -c` confirms that my cache and the remote are in sync. Every training run also logs the dataset hash as the MLflow tag `dvc_data_version`, so I always know which data a model was trained on.

![label=dvc7; The pointer file of the real dataset and dvc status -c (cache and DagsHub remote in sync)](screenshots/Task1_img_8.png)

**What the pointer file holds:** the MD5 hash of the data (for a folder it is the hash of a list of all file hashes, which is why it ends in `.dir`), the total size in bytes (503 783 378), the number of files (975) and the path. It is a tiny text file, so Git can version it easily, and the hash tells DVC exactly which bytes to download for any commit.

**Why large data does not belong in Git:** Git keeps every version of every file forever, inside every clone. A 480 MB image folder changed a few times would make every clone several GB, Git's diff and compression are made for text, not images, and GitHub refuses files over 100 MB. With DVC the images live in object storage and Git only versions the pointer.

## Task 2 – Model Development with TensorFlow/Keras

### 2.1 Input pipeline
I built the pipeline in `visionguard/data.py` with `tf.data`. The stages are:
1. **Read the split list** – `train.csv` is turned into a dataset of (path, label) pairs. Using saved lists means every run uses exactly the same images.
2. **Decode and resize** – each path goes through `load_image_path()` from `visionguard/preprocess.py`. It uses Pillow to decode the file, takes the first frame of animated files, converts palette/RGBA/greyscale/CMYK to RGB and resizes to 224×224. I used Pillow instead of `tf.io.decode_image` because TensorFlow cannot read WEBP and struggles with palette and CMYK images. The most important point: **the Flask app imports exactly the same function**, so the images are prepared identically in training and in production.
3. **Cache** – after the first epoch the decoded images stay in memory (~100 MB), so decoding happens only once.
4. **Shuffle** (training only) – with seed 42, reshuffled every epoch.
5. **Batch** – 32 images.
6. **Augment** (training only) – random horizontal flip, rotation up to ±18° and zoom up to 10 %. Validation and test images are never augmented. I kept augmentation mild on purpose: blur, noise or strong colour changes could destroy the small texture details that give AI images away.
7. **Prefetch** – the CPU prepares the next batch while the model trains on the current one.

**Image size:** 224×224 is the size MobileNetV2 and EfficientNetB0 were pre-trained on, and it is small enough to train on a CPU. **Aspect ratio:** I squash the image to a square, so the aspect ratio is not kept (a 1500×750 banner gets squeezed horizontally). Cropping would cut off parts of wide images and padding would add black bars whose size depends on the original shape. I checked that both classes have almost the same width/height ratio (median 1.5 for both), so the squashing affects both classes equally and does not become a hint for the label.

![label=aug; One training batch: top row decoded and resized, bottom row the same images after random augmentation](reports/figures/augmented_batch.png)

### 2.2 Baseline CNN
I designed a small CNN myself (`build_baseline_cnn` in `visionguard/models.py`): `Rescaling(1/255)` → four blocks of [Conv2D 3×3 → BatchNormalization → ReLU → MaxPooling 2×2] with 32, 64, 128 and 256 filters → GlobalAveragePooling → Dropout 0.4 → Dense 64 (ReLU) → Dense 1 (sigmoid). It has 406 369 parameters.

Why each layer is there:
- **Rescaling** puts the pixels into [0, 1] so the gradients are well scaled. It is inside the model, so the app cannot forget it.
- **Conv2D** layers learn local patterns: edges first, then textures, then parts. I double the number of filters every time the image is halved so deeper layers can represent more complex patterns.
- **BatchNormalization** normalises the activations, which makes training faster and more stable.
- **ReLU** is a cheap non-linearity that does not suffer from vanishing gradients.
- **MaxPooling** halves the resolution and makes the network a bit tolerant to small shifts.
- **GlobalAveragePooling** turns each feature map into one number. Compared to `Flatten` this keeps the dense layer tiny (256 inputs instead of 50 176), which reduces overfitting on only 681 images.
- **Dropout 0.4** randomly switches off 40 % of the features during training so the model cannot rely on a few of them.
- **Dense 1 with sigmoid** outputs one number between 0 and 1: the probability that the image is AI-generated.

**Loss, optimizer, metrics:** binary cross-entropy is the right loss for one sigmoid output and punishes confident wrong answers hard. Adam adapts the step size for every weight and works well without much tuning (learning rate 1e-3). As metrics I used accuracy (easy to read, the classes are 55/45), precision and recall (the two kinds of error matter differently, see 2.5) and AUC (how well the model ranks images, independent of the threshold).

![label=cnnsum; model.summary() of the baseline CNN (output shape and parameters per layer)](screenshots/Task2_img_1.png)

CNN_TABLE

**What went wrong:** my first CNN run looked like it was learning (training loss went from 0.76 to 0.60), but the validation accuracy was stuck at exactly 0.4452 – that is 65/146, so it predicted "real" for every validation image – and early stopping ended the run at epoch 8. The cause was **BatchNormalization**: during training it uses the statistics of the current batch, but during validation it uses moving averages. With the default momentum of 0.99 and only 22 batches per epoch, those averages were still close to their starting values, so validation images were scaled completely differently. I changed the momentum to 0.9 and gave early stopping more patience. After that the model trained normally: best epoch 16, test accuracy 0.640 and ROC-AUC 0.759.

![label=cnncurves; Baseline CNN training curves (the red line marks the best epoch, 16)](reports/figures/baseline_cnn_training_curves.png)

### 2.3 Transfer learning (MobileNetV2)
I loaded `keras.applications.MobileNetV2(include_top=False, weights="imagenet")`, which removes the ImageNet classifier and leaves a feature extractor, and added my own head: GlobalAveragePooling → Dropout 0.3 → Dense(1, sigmoid).

- **Stage 1 – frozen:** I set `base.trainable = False` and trained only my head with learning rate 1e-3. Only **1 281** parameters were trainable. Freezing first is important because the new head starts random and would otherwise send large gradients back and damage the pretrained filters.
- **Stage 2 – fine-tune the last block:** I unfroze only the last block (`block_16_expand`, `block_16_depthwise`, `block_16_project` and `Conv_1`) and trained again with a 10× smaller learning rate (1e-4), so the pretrained weights are only adjusted a little. BatchNorm layers stay frozen. Now **880 321** of 2 259 265 parameters were trainable. The model has to be compiled again after changing `trainable`.

| Stage | Trainable layers | Trainable parameters |
|---|---|---|
| 1 – frozen backbone | only my Dense head (`ai_probability`) | 1 281 |
| 2 – fine-tuning | `block_16_expand`, `block_16_depthwise`, `block_16_project`, `Conv_1` + head | 880 321 |

**Preprocessing:** MobileNetV2 was pre-trained on pixels scaled to the range [-1, 1] (`mobilenet_v2.preprocess_input` computes x/127.5 − 1). If I gave it [0, 1] or [0, 255] the pretrained filters would see values they never saw during training and the features would be useless. I put this scaling inside the model as a `Rescaling(1/127.5, offset=-1)` layer, so it is saved with the model.

**Result:** stage 1 reached a validation AUC of about 0.82; fine-tuning raised it to 0.87 and lowered the validation loss from 0.515 to 0.461. On the test set: accuracy 0.789, precision 0.891, recall 0.704, F1 0.786, ROC-AUC 0.872 – much better than my CNN.

![label=tlcurves; MobileNetV2 training curves: stage 1 left of the dotted line, fine-tuning right of it (best epoch 24 in red)](reports/figures/mobilenetv2_tl_training_curves.png)

**What went wrong:** in Keras 3 the weight names are only `kernel` and `bias`, so my first list of trainable layers was useless; I switched to `weight.path`. I also noticed that the checkpoint callback in stage 2 started from scratch, so it could have overwritten the better stage-1 model; I fixed it by giving stage 2 the best stage-1 validation loss as a starting threshold.

### 2.4 Training discipline
- I fixed all random seeds with `tf.keras.utils.set_random_seed(42)`; the splits and the shuffling also use seed 42.
- **EarlyStopping** watches the validation loss and stops when it does not improve any more, restoring the best weights.
- **ModelCheckpoint** saves the model with the lowest validation loss to disk.
- **ReduceLROnPlateau** cuts the learning rate by 70 % when the validation loss stalls.

The best epochs were 16 for the CNN, 24 for MobileNetV2 and 26 for my final model R4. {fig:r4curves} shows the curves of the final model with the best epoch marked. Even with fixed seeds the runs are not 100 % repeatable on the CPU: R1 and R2 used exactly the same settings but got a validation AUC of 0.819 and 0.854.

![label=r4curves; Loss and accuracy curves of the final model R4 with the best epoch (26) marked](reports/figures/R4_mnv2_no_augment_training_curves.png)

### 2.5 Evaluation
**Which error is worse?** I treat "AI-generated" as the positive class. Calling a **real photo fake** (a false positive) means accusing a real photographer, artist or journalist of faking their work, which can hurt their reputation and makes people stop trusting the tool. Calling a fake real is also bad, but without VisionGuard that would happen anyway. So I decided that **calling a real photo fake is the worse error**, and I want high precision for the AI class.

**Threshold:** I chose the threshold on the **validation set**, never on the test set. I tried thresholds from 0.05 to 0.95 and kept the one with the best **F0.5 score**, which weighs precision twice as much as recall. This gave 0.67 for the CNN, 0.72 for MobileNetV2 and 0.80 for my final model. Only then did I evaluate each model **once** on the test set.

| Test set | Baseline CNN | MobileNetV2 | Final model R4 |
|---|---|---|---|
| Threshold (from validation) | 0.67 | 0.72 | 0.80 |
| Accuracy | 0.640 | 0.789 | 0.714 |
| Precision | 0.726 | 0.891 | 0.868 |
| Recall | 0.556 | 0.704 | 0.568 |
| F1 | 0.629 | 0.786 | 0.687 |
| ROC-AUC | 0.759 | 0.872 | 0.860 |

![label=cm; Test confusion matrices: baseline CNN, MobileNetV2, final model R4 (each at its validation threshold)](reports/figures/eval_confusion_matrix_3models.png)

![label=roc; Test ROC curves: baseline CNN, MobileNetV2, final model R4](reports/figures/eval_roc_curve_3models.png)

The high thresholds do exactly what I wanted: for example MobileNetV2 wrongly flagged only 7 of the 66 real test images. The price is a lower recall – some AI images get through as "real".

### 2.6 Error analysis
{fig:errors} shows the 10 test images where my final model was most confidently wrong. All of them are AI images that it called "real", which is the error type I chose to accept.

![label=errors; The 10 most confident test errors of the final model (threshold 0.8)](reports/figures/error_analysis_champion.png)

My hypotheses for each one:
1. A **screenshot** of AI art with a "Matrix code" overlay in Van Gogh style – it looks like a human painting.
2. A flat **pop-art collage / article banner** – there is no photographic texture to judge.
3. A very saturated landscape that looks like an **HDR travel photo**, which the real class also contains.
4. A real forest photo where only half has an AI "pixel" effect – a **composite**, so the label is questionable.
5. A soft, **photorealistic** Midjourney landscape that was re-compressed as JPEG.
6. A realistic face next to a cartoon character on a white studio background – the **photographic face dominates**.
7. Boris Eldagsen's famous AI "vintage photo" that won a photography prize – it **fooled human judges too**.
8. A **painting style** with visible brush strokes, like the human paintings in the real class.
9. A lush **digital-painting** landscape that looks like a stock illustration.
10. An anime wallpaper of only 49 KB – **heavy compression** removes the fine generator details.

So the model struggles with painting styles, photorealistic fakes, collages and strongly compressed images.

### Bonus – Grad-CAM
I implemented Grad-CAM (`src/gradcam.py`) on the last convolutional layer of MobileNetV2 for 4 correct and 4 wrong predictions.

![label=gradcam; Grad-CAM for 4 correct and 4 wrong test predictions (top: input, bottom: regions that drove the prediction)](reports/figures/gradcam_champion.png)

For correct AI predictions the heat is on the **main subject** (the castle, the moon and mountains). For correct real predictions it is on the hand-painted canvas or on the faces in a photo grid. The famous real painting *Girl with a Pearl Earring* is wrongly called AI because the model looks at the smooth, perfect face – exactly how AI portraits look. My conclusion is that the model judges the subject and the "style" (smoothness, saturation, perfect faces) and not tiny pixel-level traces of the generator. That explains both why it works and why it fails on paintings and photorealistic fakes.

### 2.7 Keras Tuner
**Toy part:** I tuned a tiny network (Flatten → Dense(units) → Dense 10) on Fashion-MNIST with `RandomSearch` (10 trials, 4 epochs each), searching the number of units (32–256) and the learning rate (1e-4 to 1e-1, log scale).

TOY_TABLE

The best setting was **256 units with learning rate 0.0071** (validation accuracy 0.860). All trials with a learning rate below 2e-4 ended at the bottom, no matter how many units – with only 4 epochs a too-small learning rate simply does not converge.

**Apply part:** I wrote `VisionGuardHyperModel`, which repeats my two-stage recipe inside every trial (4 epochs with the frozen backbone, then 6 epochs of fine-tuning). I searched the fine-tuning learning rate (1e-5 to 1e-3), the dropout (0.1–0.5) and the number of unfrozen layers (0, 10, 20, 40 or 60) with 10 trials, maximising validation AUC.

APPLY_TABLE

**What mattered most:** the **fine-tuning learning rate together with the number of unfrozen layers**. With 40 unfrozen layers, a learning rate of 2.3e-4 gave the best result (0.893) but 6.7e-4 gave the worst (0.703) – a big learning rate on many unfrozen layers destroys what the network learned on ImageNet. Not unfreezing anything (trial 0) stayed at 0.78. Dropout did not show a clear effect.

**Why the best validation score can overfit the validation set:** all 10 trials are scored on the same 146 validation images and I pick the maximum. Part of the winner's advantage is just luck on those particular images (I saw ±0.035 AUC between two identical runs), and the more trials I run, the more likely one wins by chance. That is why 0.893 is not an honest estimate of real performance – only an untouched test set can give that. For this reason I did not replace my final model just because of the tuner; its suggestion is a candidate for a future run.

## Task 3 – Experiment Tracking with MLflow and DagsHub

### 3.1 MLflow locally
**What I did:** `src/mlflow_toy.py --target local` trains a tiny Dense network on 20 000 Fashion-MNIST images twice, once with learning rate 0.001 and once with 0.01. Each run logs the parameters (learning rate, units, epochs, batch size), the loss and accuracy after every epoch (with a small Keras callback), one plot (`loss_curve.png`) and the trained model. The local tracking store is a SQLite file, and I opened the UI with `mlflow ui --backend-store-uri sqlite:///mlflow_local.db` to compare the two runs. Result: lr 0.001 reached 0.840 validation accuracy, lr 0.01 reached 0.838 but with a lower training loss, which means it starts to overfit earlier.

[[OPT extra_mlflow_local_runs.png | Local MLflow UI with the two toy runs]]
[[OPT extra_mlflow_local_compare.png | Local MLflow compare view of the two learning rates]]

**My definitions:**
- **Experiment** – a named folder that groups all attempts at one problem, so they can be compared side by side.
- **Run** – one execution of the training code, with everything recorded about that single attempt.
- **Parameter** – a setting I choose before the run, like the learning rate; it is logged once and cannot change.
- **Metric** – a number measured during or after the run, like loss or AUC; it can be logged every epoch to draw curves.
- **Artifact** – any file the run produces, like a plot, the model or the split lists.
- **Tag** – a label with extra information about the run (git commit, data version, seed) that helps me find and trust runs later.

**What went wrong:** MLflow 3.17 opens in a "GenAI" view – I had to switch to "Model training" to see normal runs. My first script also crashed with `dtype='string'` because in Keras 3 the third positional argument of `compile()` is `loss_weights`, not `metrics`; using keyword arguments fixed it.

### 3.2 DagsHub setup
I created my DagsHub account with GitHub and created the project **ALiAdnan3/23L0839_A2_Ops** connected to my GitHub repository (it mirrors it). Its MLflow tracking address is `https://dagshub.com/ALiAdnan3/23L0839_A2_Ops.mlflow`. I created an access token and put the address and the token into a `.env` file as `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD`. MLflow reads these environment variables by itself, so the token never appears in my code. `.env` is listed in `.gitignore`, so it can never be committed. Then I repeated the toy runs against DagsHub (`--target dagshub`).

![label=dagshub; My DagsHub project (connected to GitHub) with the code, Dockerfiles and requirements](screenshots/Task1_img_9.png)

![label=toyruns; The two toy runs on the DagsHub MLflow server](screenshots/Task2_img_3.png)

![label=gitignore; .gitignore of the project: .env (secrets) and the DVC credentials file are never committed](screenshots/Task2_img_4.png)

**What went wrong:** at the end of every run MLflow prints a 🏃 emoji, and the Windows console (cp1252) crashed with a `UnicodeEncodeError`. I fixed it by switching Python's output to UTF-8 and deleted the half-finished run.

### 3.3 Instrumenting VisionGuard training
My training script `src/train.py` has a switch `--mlflow manual` or `--mlflow autolog`.
- **Manual logging:** a custom Keras callback logs training and validation loss, accuracy, precision, recall, AUC and the learning rate after every epoch (continuing the step counter across both stages). I also log all my hyper-parameters, the final test metrics, the threshold, the best epoch and the tags `git_commit`, `dvc_data_version` and `seed`.
- **Autologging:** `mlflow.tensorflow.autolog()` automatically records the arguments of `fit()`, the optimizer settings, the callback settings, the per-epoch metrics, a model summary, TensorBoard logs and the model.

![label=runs; All VisionGuard runs on DagsHub (R2 is the autolog run, the others use manual logging)](screenshots/Task2_img_2.png)

![label=autolog; Autolog run R2: metrics recorded automatically, plus my own test metrics, threshold and best epoch](screenshots/Task2_img_5.png)

[[OPT extra_r1_run_page.png | Manual run R1: parameters, metric curves and tags]]

| | Manual logging | Autologging |
|---|---|---|
| My design choices (augmentation, image size, backbone, unfrozen layers) | ✅ | ❌ not arguments of `fit()`, I had to add them myself |
| Optimizer and callback details | partly | ✅ very detailed |
| Two training stages | ✅ one continuous curve | ❌ the second `fit()` tried to change parameters that were already logged and MLflow rejected it, so the fine-tuning learning rate is missing |
| Test metrics, threshold, confusion matrix, tags | ✅ | ❌ only what happens inside `fit()` |
| Effort | more code | one line |

My conclusion: autolog gives a lot of low-level detail for free, but anything outside `fit()` – like the test results and the business decisions – has to be logged by hand. The best is to combine both.

**What went wrong:** the first autolog run crashed with `TBNotInstalledError`, because MLflow's Keras autologging writes TensorBoard summaries and TensorFlow 2.21 no longer installs TensorBoard. Installing `tensorboard==2.21.0` fixed it.

### 3.4 Artifacts
Every run stores its evaluation (confusion matrix, ROC curve, training curves, a summary, the test predictions and the list of trainable layers), the three split lists, the full package list (`pip freeze`), the `.keras` model file and the MLflow model.

[[OPT extra_r4_artifacts.png | Artifacts of run R4]]

### 3.5 Experiment comparison
I ran a series of experiments with `scripts/run_experiments.sh`, where each run changes **one** thing compared to the reference run R1.

![label=compare; DagsHub/MLflow compare view of the parameters of the runs (one change per run)](screenshots/Task2_img_6.png)

| Run | What changed vs R1 | Best epoch | Val loss | Val AUC | Val F0.5 | Test acc | Test AUC |
|---|---|---|---|---|---|---|---|
| R1 | reference: MobileNetV2, lr 1e-3, light augmentation, 224 px, last block | 23 | 0.517 | 0.819 | 0.797 | 0.742 | 0.834 |
| R2 | same as R1 but autologging | 22 | 0.494 | 0.854 | 0.840 | 0.782 | 0.863 |
| R3 | learning rate 3e-4 | 30 | 0.532 | 0.815 | 0.825 | 0.762 | 0.848 |
| **R4** | **no augmentation** | 26 | **0.445** | **0.875** | **0.853** | 0.714 | 0.860 |
| R5 | strong augmentation | 23 | 0.452 | 0.859 | 0.842 | 0.748 | 0.846 |
| R6 | frozen backbone only | 16 | 0.589 | 0.767 | 0.761 | 0.728 | 0.808 |
| R7 | image size 160 | 23 | 0.500 | 0.840 | 0.821 | 0.769 | 0.837 |
| R8 | EfficientNetB0 backbone | 27 | 0.461 | 0.865 | 0.831 | 0.776 | 0.884 |
| R9 | baseline CNN | 16 | 0.552 | 0.774 | 0.753 | 0.673 | 0.742 |

![label=comparechart; Validation metrics I used to choose the model, and test AUC (only reported)](reports/figures/experiment_comparison.png)

**My choice: R4.** I chose using **validation** metrics only: validation ROC-AUC (how well it ranks images), validation F0.5 (my precision-focused product metric) and validation loss. R4 is the best on all three. I did not choose by test score – R8 has the best test AUC, but if I picked by the test set, the test set would become a second validation set and my reported test score would be too optimistic.

To be honest, R1 and R2 have identical settings and still differ by 0.035 validation AUC, so random noise is about as big as most of these differences – R4, R5 and R8 are basically tied. The results I trust are: fine-tuning beats a frozen backbone (R1 vs R6), and transfer learning beats my CNN from scratch (R1 vs R9). I registered R8 as `@challenger` so it can be tested again on more data later.

### 3.6 Model registry
With `src/registry.py` I registered the model `23L0839_A2_Ops-visionguard` three times: version 1 from R1, version 2 from R4 with the alias **`@champion`**, and version 3 from R8 with the alias `@challenger`. Each version is tagged with its run name, threshold and validation AUC.

![label=registry; Model registry on DagsHub: three versions, @champion on version 2 and @challenger on version 3](screenshots/Task2_img_7.png)

Then I wrote a separate small script `scripts/predict_champion.py` that loads the model by its alias (`models:/23L0839_A2_Ops-visionguard@champion`) and classifies one image.

![label=champion; Loading the model by its alias from DagsHub and classifying one image](screenshots/Task2_img_8.png)

**What went wrong:** MLflow 3 stores models as "LoggedModels", and registering with `runs:/<id>/model` failed on DagsHub ("Unable to find a logged_model"). I fixed it by looking up the logged model of each run and registering its own `models:/m-…` address.

**How the model reaches the server:** the web app must not call DagsHub for every request. So `python -m src.registry fetch` downloads the `@champion` model **once** into `models/champion.keras` (together with its version and threshold). Docker Compose mounts this file into the web container, and the cloud API has it built into its image. To promote a new model I only move the alias, run `fetch` again and restart – no code change needed.

## Task 4 – Flask Web Application

### 4.1 Inference module
`visionguard/inference.py` contains a `Predictor` class. When the app starts, it loads the champion `.keras` file **once** and runs one warm-up prediction. After that, every request only calls `predict_bytes()`, which uses the same `load_image()` function as training, runs the model and returns the label, the confidence, P(AI), the threshold and the model version. The confidence is the probability of the class that was chosen: P(AI) if the answer is "AI-generated", otherwise 1 − P(AI).

![label=demo; The inference module loads the model once and classifies two known images (one AI, one real)](screenshots/Task4_img_1.png)

**Why loading per request is unacceptable:** loading the model takes about 1–8 seconds (reading 16 MB, rebuilding the network), while one prediction takes only 0.1–0.5 s. Loading it for every request would make each upload many times slower and use a lot more memory when several people upload at once.

**Why training and serving preprocessing must match:** the model only knows the kind of input it was trained on. If the app resized differently, used BGR instead of RGB, scaled to [0, 1] instead of [-1, 1] or kept the alpha channel, the model would still return a confident number – just a wrong one. Using one shared function and keeping the scaling inside the model makes this mistake impossible.

### 4.2 Routes and validation
The app (`webapp/app.py`) has these routes: `/` (upload page), `/predict` (result), `/history` (database history) and `/health` (JSON health check). Before the model sees anything, every upload is checked:
- **Wrong file type** – only jpg, jpeg, png, webp, gif and bmp are allowed, otherwise error **415**.
- **Too large** – the limit is 5 MB (`MAX_UPLOAD_MB`); Flask rejects bigger requests with **413** before reading them.
- **Looks like an image but is not** – the bytes are really decoded with Pillow; if that fails the answer is **422**.

File names are cleaned with `secure_filename` (for example `../../etc/passwd.png` becomes `etc_passwd.png`). **The upload is never saved to disk:** the bytes stay in memory, the preview on the result page is sent back inside the HTML, and only the file name and a SHA-256 hash go into the database. All errors have friendly pages. These cases are also covered by my automated tests.

![label=e413; Friendly error page for a file over the size limit (413)](screenshots/Task4_img_4.png)

[[OPT extra_error_415.png | Friendly error page for a wrong file type (415)]]
[[OPT extra_error_422.png | Friendly error page for a file that cannot be decoded (422)]]

The other two failure cases, tested with curl against the running app:
```
> curl -F image=@notes.txt http://localhost:8000/predict
Error 415 - Unsupported file type
'.txt' is not allowed. Upload one of: bmp, gif, jpeg, jpg, png, webp.
> curl -F image=@fake_image.jpg http://localhost:8000/predict
Error 422 - Not a readable image
'fake_image.jpg' has an image extension but its content could not be decoded. It may be corrupt or not an image at all.
```

### 4.3 HTML and CSS
The templates in `webapp/templates/` share one `base.html` with a menu: **Upload · History · Health**. The result page shows the image, the verdict in red (AI) or green (Real), a confidence bar, P(AI) with the threshold, the model version, how the prediction was made (local or remote) and how long it took, the database row number and time, and an honest warning that the model can be wrong and should be treated as a hint. The CSS is plain and also works on a phone.

[[OPT extra_upload_page.png | Upload page]]

![label=resai; Result page for an AI-generated image](screenshots/Task4_img_2.png)

![label=resreal; Result page for a real image](screenshots/Task4_img_3.png)

### 4.4 Configuration and production server
All settings come from environment variables (`webapp/config.py`): the inference mode, model path, model name and version, threshold, database address, upload limit, cloud API address and key. That means the same code runs on my laptop, in Docker and against the cloud just by changing the environment, and no password is written in the code.

I run the app with **waitress**, a production WSGI server, and **debug mode off**. Flask's built-in server is only meant for development, and debug mode shows the source code and settings to anyone who causes an error – and even gives them an interactive console that can run code on the server. It would also restart and reload the model whenever a file changes.

![label=health; /health response: database ok, debug false, local inference with the champion model](screenshots/Task4_img_6.png)

## Task 5 – Database Trail
Every successful prediction writes one row into the `predictions` table (`webapp/db.py`, using SQLAlchemy):

| Column | Answers the question |
|---|---|
| `id`, `created_at` (UTC) | **when** |
| `filename`, `image_sha256` | which image (without storing the image itself) |
| `label`, `prob_ai`, `threshold` | **what was decided** and why |
| `confidence` | **how confidently** |
| `model_name`, `model_version`, `inference_mode` | **by which model** (inside the container or in the cloud) |
| `latency_ms` | how long it took |

The database address comes from `DATABASE_URL`: SQLite when I test locally and **PostgreSQL 16** in Docker Compose. The `/predict` view inserts the row right after the model answers, and the **History** menu shows the newest rows first. Because PostgreSQL stores its files in a named Docker volume, the history survives restarts and even re-creating the containers.

![label=hist1; History menu showing what was decided, when, by which model, how confidently, and whether it ran locally or in the cloud](screenshots/Task4_img_5.png)

![label=hist2; History page with row numbers (each number links to a single record)](screenshots/Task8_img_1.png)

## Task 6 – Containerization with Docker and Docker Compose

### 6.1 Docker warm-up
Before touching my project I practised with the ready-made `nginx:alpine` image (`scripts/docker_warmup.ps1`). I ran a container that maps port 8081 on my laptop to port 80 inside it and mounts a folder from my laptop as the web root, so nginx served my own `index.html`. Then I read its logs, opened a shell inside it, and finally stopped and removed the container and the image.

![label=w1; docker run with port mapping (-p 8081:80) and a mounted volume (-v); the page comes from the mounted folder](screenshots/Task6_img_1.png)

![label=w2; docker logs: nginx start-up and my GET request](screenshots/Task6_img_2.png)

![label=w3; A shell inside the running container with docker exec](screenshots/Task6_img_3.png)

![label=w4; docker ps -a, stopping and removing the container](screenshots/Task6_img_5.png)

![label=w5; Listing images and removing the nginx image with docker rmi](screenshots/Task6_img_4.png)

**My definitions:**
- **Image** – a read-only template (files + settings + start command) that containers are created from, like a class.
- **Container** – a running (or stopped) copy of an image with its own writable layer, processes and network, like an object of that class.
- **Layer** – every Dockerfile instruction that changes files creates one read-only layer; images are stacks of layers that are cached and shared.
- **Volume** – storage outside the container's own layer, so the data survives when the container is deleted.
- **Port mapping** – `-p host:container` forwards traffic from a port on my laptop to a port inside the container's private network.

### 6.2 Dockerfile for the web app
Dockerfile: https://github.com/ALiAdnan3/MLOPS_Assignment_2/blob/main/Dockerfile

The order of the instructions and why:
1. `FROM python:3.13-slim` – a slim Debian base instead of the full one (which has compilers I do not need).
2. `ENV ...` – no `.pyc` files, logs are printed immediately, and pip keeps no cache.
3. `COPY requirements-serve.txt` and `RUN pip install` **before** copying my code. Docker reuses a layer from cache if its instruction and its input files did not change. My dependencies rarely change but my code changes all the time, so with this order a code change only rebuilds the last tiny layers (a rebuild took about 7 seconds) instead of reinstalling 1.6 GB of packages.
4. `useradd ... appuser` and `USER appuser` – the app does **not run as root**, so even if someone broke into it they could not change the system.
5. `COPY visionguard/ webapp/` – only the code the server needs.
6. `HEALTHCHECK` – Docker calls `/health` every 30 seconds and marks the container unhealthy if the model or the database stops working.
7. `CMD ["python", "-m", "webapp.app"]`.

A `.dockerignore` file keeps `.env`, `.git`, the virtual environment, the dataset and the outputs out of the build, so the build is fast and **no secrets end up inside the image** – passwords are only given at run time as environment variables.

**TensorFlow package:** I deliberately used `tensorflow-cpu`. Its download is 274 MB compared with 573 MB for the normal `tensorflow` package, which includes GPU code that a CPU container can never use. One prediction takes well under a second on the CPU, so a GPU is not needed. I also made a separate `requirements-serve.txt` without the training-only libraries (MLflow, DVC, Keras Tuner, matplotlib, scikit-learn).

The user inside the container is `appuser`, as {fig:compexec} shows.

### 6.3 Compose stack
Compose file: https://github.com/ALiAdnan3/MLOPS_Assignment_2/blob/main/docker-compose.yml
- Two services, **web** and **db** (`postgres:16-alpine`). The web app reaches the database by its service name: `@db:5432` in `DATABASE_URL`.
- The database stores its files in the **named volume** `pgdata`.
- The database has a `pg_isready` health check, and the web service has `depends_on: db: condition: service_healthy`, so the app only starts when PostgreSQL is ready to accept connections.
- **The database publishes no port.** It has no `ports:` section, so `docker compose ps` shows `5432/tcp` without a mapping to my laptop – only other containers on the Compose network can reach it. Only the web app publishes port 8000.
- All passwords come from my untracked `.env` file; `.env.example` shows which variables are needed, with dummy values.
- **The model file is mounted**, read-only: `./models:/app/models:ro`. The model changes independently of the code, so promoting a new champion is just "fetch + restart" without rebuilding a 2 GB image, and `:ro` stops the app from changing it. (For the cloud API I made the opposite choice and built the model into the image – see Task 7.)

![label=compose; docker compose up / ps / stop / start / restart: both services healthy; the db shows 5432/tcp with no host port, only the web app publishes 8000](screenshots/Task6_img_6.png)

### 6.4 Image analysis
`docker history` showed that the biggest layer by far is `pip install` with **1.66 GB**; TensorFlow alone takes 1.3 GB of it. The slim Python base is about 120 MB and my own code is less than 0.1 MB.

To measure the effect of my optimisations I also built a deliberately bad version, `Dockerfile.unoptimized`: the full `python:3.13` base, the normal `tensorflow` package and pip's download cache left inside.

| Image | Base | TensorFlow package | pip cache | Size on disk | Download size |
|---|---|---|---|---|---|
| Before (`Dockerfile.unoptimized`) | `python:3.13` (full) | `tensorflow` | kept (629 MB) | **5.88 GB** | **1.74 GB** |
| After (`Dockerfile`) | `python:3.13-slim` | `tensorflow-cpu` | none | **2.22 GB** | **421 MB** |

![label=sizes; docker images: the optimised image (2.22 GB) next to the unoptimised one (5.88 GB)](screenshots/Task6_img_4.png)

**Conclusion:** the image got **62 % smaller on disk** and **76 % smaller to download**. Almost all of the remaining size is TensorFlow itself, so the biggest wins come from choosing the right TensorFlow package and base image. To go further I would need a lighter runtime for serving, like TensorFlow Lite.

### 6.5 Operations and troubleshooting
With `scripts/docker_ops_demo.ps1` I demonstrated `docker compose up -d`, `ps`, `stop`, `start`, `restart`, `logs` and a shell inside each service. {fig:compose} shows the start, status, stop, start and restart commands; after stopping, the containers show `Exited`, and after starting again both are `healthy`.

![label=compexec; A shell inside the web container (runs as appuser, model mounted) and psql inside the database showing the prediction records](screenshots/Task6_img_7.png)

**Breaking it on purpose:** I started the web service with a **wrong database password** (`$env:POSTGRES_PASSWORD="wrong-password"` – a variable in the shell overrides `.env` for Compose). The web container kept restarting. I diagnosed it only from the logs:
```
web-1 | sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) connection to server at "db"
        (172.18.0.2), port 5432 failed: FATAL:  password authentication failed for user "visionguard"
db-1  | 2026-10-20 20:03:07.738 UTC [47] FATAL:  password authentication failed for user "visionguard"
```
**Diagnosis:** the web log shows that the database server *answered* (so the network and the name `db` work) but refused the login, and the db log confirms that the password was wrong. An interesting detail: the db container also received the wrong password, but PostgreSQL only uses `POSTGRES_PASSWORD` when it creates a *new, empty* volume – my existing volume still had the old password, so the two did not match.

[[OPT extra_ops_failure.png | The failure in the logs]]

**Fix:** I removed the wrong variable so the password from `.env` was used again and recreated the web container. Both services became healthy again, `/health` reported the database as ok, and all the old history rows were still there.

![label=ops3; After the fix: both services healthy and /health reports the database as ok](screenshots/Task6_img_8.png)

### 6.6 GitHub Actions (CI)
**Toy part:** in a throwaway repository (`23L0839_A2_Ops_ci_toy`) I created a workflow that runs on every push, uses a **matrix of Python 3.11 and 3.12** and runs a small pytest file. My first commit was green. Then I introduced a bug on purpose (`add` returned `a - b`), the run turned **red** (`assert -1 == 5` failed on both Python versions), and after fixing it the run was **green** again.

![label=citoy; Toy repository: green, then red (bug on purpose), then green again (fixed)](screenshots/Task6_img_9.png)

[[OPT extra_ci_red_job.png | The failing pytest step of the red run]]

**Apply part:** the workflow `.github/workflows/ci.yml` of VisionGuard has two jobs. `lint-and-test` installs the dependencies, runs `flake8` and runs my 7 pytest tests (they use a fake model and a temporary SQLite database, so CI does not need TensorFlow or the real model). `docker-build` runs only if the first job passed; it builds the Docker image and prints its size and the non-root user. Every push so far has been green, and the status badge is at the top of my README.

![label=ciproj; VisionGuard CI: every push green](screenshots/Task6_img_10.png)

[[OPT extra_readme_badge.png | CI status badge in the README]]

**Where secrets go:** workflow secrets are stored in GitHub under *Settings → Secrets and variables → Actions*. They are encrypted, cannot be read back, are hidden as `***` in logs, and are used in a workflow as `${{ secrets.NAME }}`. They must never be written in the workflow file, because that file is part of the repository: anyone who can see the repository would see the secret, and it would stay in the Git history forever even after deleting it.

## Task 7 – API Deployment (Microsoft Azure)
I chose **pathway C** with Microsoft Azure. My model API is reachable at:

**https://vg-23l0839-a2-ops-api.gentleglacier-7f267617.eastus.azurecontainerapps.io**
- `GET /health` – public; returns the model name, version and threshold.
- `POST /predict` – needs the header `X-API-Key` and an image in the form field `image`; returns the label, confidence, P(AI), threshold, model version and the time the model needed.

**What I researched first:**
- A new Azure account gets USD 200 of credit. Azure Container Apps includes a monthly free amount of CPU time and requests, and with **minimum replicas = 0** the app shuts down when nobody uses it, so it costs almost nothing. The downside is a **cold start**: the first request after a quiet period took about 15 seconds (starting the container, importing TensorFlow and loading the model).
- On a free-trial subscription Azure **does not allow** building images in the cloud with `az acr build` (error `TasksOperationsNotAllowed`), so I built the image on my laptop and pushed it.
- How my Docker Compose setup maps to Azure:

| Docker Compose | Azure |
|---|---|
| image built locally | image in **Azure Container Registry** `acr23l0839a2ops` |
| service container | **Container App** `vg-23l0839-a2-ops-api` (1 vCPU, 2 GB, 0–1 replicas) |
| network and `ports:` | Container Apps **environment** + **ingress** (public HTTPS → port 8080, certificate managed by Azure) |
| `.env` values | Container App **secrets** and environment variables |
| `docker compose logs` | Log Analytics workspace / log stream |
| restart policy and health check | the platform restarts and scales the app |
| db service | stays on my laptop, because Task 8 runs Flask and the database locally |

- **The model is built into the API image** (`Dockerfile.api`), because Container Apps has no simple shared disk, and a fixed image guarantees that exactly the tested model runs. A new model means a new image tag.
- **Security:** `/predict` requires an API key, compared in constant time. The key is an Azure secret and is not in the image or in Git. Only HTTPS is used and uploads are limited to 5 MB.

**Step by step:**
1. `az login` – signed in with my Azure account.
2. `az provider register` for Microsoft.App, Microsoft.OperationalInsights and Microsoft.ContainerRegistry – enables these services on a new subscription.
3. `az group create -n 23L0839_A2_Ops -l eastus` – one resource group for the whole project.
4. `az acr create -n acr23l0839a2ops --sku Basic` – a private registry for my image.
5. `python -m src.registry fetch` – downloaded the `@champion` model from DagsHub.
6. `docker build -f Dockerfile.api -t acr23l0839a2ops.azurecr.io/23l0839-a2-ops-api:v2 .` – took only 7 seconds because it reused the cached TensorFlow layer. I tested it locally first.
7. `az acr login` + `docker push` – my home internet dropped once, but `docker push` skips layers that are already uploaded, so retrying worked.
8. `az containerapp env create -n env-23l0839-a2-ops` – the environment (Azure also created a Log Analytics workspace).
9. `az containerapp create -n vg-23l0839-a2-ops-api --image ...:v2 --target-port 8080 --ingress external --cpu 1.0 --memory 2.0Gi --min-replicas 0 --max-replicas 1 --secrets api-key=... --env-vars API_KEY=secretref:api-key THRESHOLD=0.8 MODEL_VERSION=v2@champion` – this printed the public address.

![label=azure; My Azure resource group 23L0839_A2_Ops: container registry, Container Apps environment, the Container App and the log workspace](screenshots/Task7_img_1.png)

[[OPT extra_azure_container_app.png | The Container App overview with its public URL]]

**Testing the API.** I tested the API with a Postman collection (`postman/23L0839_A2_Ops_VisionGuard.postman_collection.json`) that contains four requests with automatic tests:

| Request | Status | Result |
|---|---|---|
| GET /health | 200 (15 s cold start, then under 1 s) | status ok, model v2@champion, threshold 0.8 |
| POST /predict – AI image | 200 | AI-generated, confidence 0.9994, model time 256 ms |
| POST /predict – real image | 200 | Real, confidence 0.9919, P(AI) 0.0081 |
| POST /predict – without API key | **401** | `{"error":"missing or invalid X-API-Key"}` |

[[OPT extra_postman_health.png | Postman: GET /health]]
[[OPT extra_postman_predict.png | Postman: POST /predict with test results]]
[[OPT extra_postman_401.png | Postman: request without API key rejected with 401]]
[[OPT extra_phone_mobile_data.png | The API opened on my phone over mobile data (another network)]]

## Task 8 – Integrating Flask and the API
In this task my Flask app runs **on localhost** (in Docker Compose, together with the database) but no longer uses its own model. I set `INFERENCE_MODE=remote`, `MODEL_API_URL` (the Azure address) and `MODEL_API_KEY` in `.env` and restarted the web container (`scripts/set_mode.ps1 remote`). In remote mode the app creates a `RemotePredictor` instead of loading TensorFlow. When I upload an image, Flask first does all the checks from 4.2, then sends the image to the Azure API with the API key, turns the answer into the same result format, saves the row in PostgreSQL with `inference_mode = remote` and shows the result page. If the cloud API is down or too slow, the user gets a friendly "Model unavailable" page.

The health check in remote mode also shows the cloud API's status:
```
{ "database": "ok", "debug": false, "inference_mode": "remote", "status": "ok",
  "model_api": { "service": "23L0839_A2_Ops visionguard model API", "status": "ok",
                 "model_version": "v2@champion", "threshold": 0.8, "input_size": 224 } }
```

**Several inputs through the whole chain** (browser form → Flask on localhost → Azure → PostgreSQL). First batch, 8 test images the model had never seen:

| Image | Truth | VisionGuard (remote) | Confidence | Round trip |
|---|---|---|---|---|
| images71.jpg | AI | AI-generated | 89.8 % | 1.2 s |
| Nov20_17_498430921.jpg | AI | **Real** ✗ (the half-real composite from 2.6) | 89.7 % | 4.2 s |
| il_570xN.4588037671_ak6c.jpg | AI | AI-generated | 99.9 % | 1.8 s |
| 63964cc7…google-not-against-ai.jpg | AI | AI-generated | 99.8 % | 2.8 s |
| Portrait197a-1280x640.jpg | Real | Real | 98.3 % | 1.9 s |
| 1032249.jpg | Real | Real | 97.1 % | 2.9 s |
| gettyimages-1238924365_wide….jpg | Real | Real | 66.4 % | 2.2 s |
| pexels-photo-2422290.jpeg | Real | Real | 74.0 % | 3.3 s |

7 of 8 were correct. The **History** menu ({fig:hist1}) shows these rows with Mode = remote next to the rows from local mode, so the database log always tells which predictions came from the cloud.

A second batch of four images on 26 October gave the same answers as the local model (records #16–#19):
```
ai_1.png   -> AI-generated 99.9%  remote 5852 ms  #16 2026-10-26 11:48:27
ai_2.jpg   -> AI-generated 99.9%  remote 2193 ms  #17 2026-10-26 11:48:30
real_1.jpg -> Real         99.2%  remote 2110 ms  #18 2026-10-26 11:48:32
real_2.jpg -> Real         98.3%  remote 1798 ms  #19 2026-10-26 11:48:34
```
The same four requests arriving in the log of the Azure Container App. The model itself only needs about 0.2 s; the rest of the round trip is the upload from Pakistan to the US East data centre (and the first request also woke the app up from zero replicas):
```
11:47:57 INFO api: model loaded in 0.97s (version v2@champion)
11:48:29 INFO api: predict file=ai_1.png label=AI-generated prob_ai=0.999 228.8ms
11:48:31 INFO api: predict file=ai_2.jpg label=AI-generated prob_ai=0.999 189.6ms
11:48:34 INFO api: predict file=real_1.jpg label=Real prob_ai=0.008 178.2ms
11:48:35 INFO api: predict file=real_2.jpg label=Real prob_ai=0.017 186.7ms
```

[[OPT extra_remote_result.png | Result page in remote mode]]
[[OPT extra_azure_log_stream.png | Azure log stream with the incoming predict requests]]

## Conclusion
I built the complete path from raw data to a cloud service: a cleaned and versioned dataset, two models (a CNN from scratch and a fine-tuned MobileNetV2), tracked and compared experiments, a registered champion model, a validated Flask app with a PostgreSQL history, a small and secure Docker setup with CI, and a public API on Azure that my local app uses. The final model reaches a test ROC-AUC of 0.86 with high precision (0.87), which fits my decision that wrongly accusing a real photo is the worse error.

The most useful lessons for me were: the same preprocessing must be shared between training and serving; a model has to be chosen on validation data, not test data; random noise between runs can be as big as the effect I want to measure; and almost every problem I had (BatchNorm statistics, missing TensorBoard, MLflow 3 model paths, the Windows console, Azure's free-trial limits, a wrong database password) was found by reading the logs carefully. If I continue the project, I would collect more and cleaner data (no banners or collages), train the tuner's suggested settings, and test the EfficientNetB0 challenger on more images.
