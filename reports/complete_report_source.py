"""One-off edit of REPORT_FINAL.md: reference every figure in the text, add a 'Problems' line to every
sub-task and explain the Docker commands (requirements of section 11.1)."""
import re
from pathlib import Path

P = Path(__file__).resolve().parent / "REPORT_FINAL.md"

LEAD = {  # sentence placed right before a figure that is not yet referenced
    "env": "{fig:env} shows the versions printed inside my virtual environment.",
    "dvc1": "{fig:dvc1}: step 1 – Git and DVC initialised in the empty toy folder and the small CSV created.",
    "dvc2": "{fig:dvc2}: the pointer file `students.csv.dvc` holds only the MD5, size and path; Git commits the pointer and the `.gitignore`.",
    "dvc3": "{fig:dvc3}: after `dvc add` of the changed file the MD5 changes; `git log` shows both versions and `dvc checkout` restores the 2-row version.",
    "dvc4": "{fig:dvc4}: the DagsHub remote is configured (token hidden) and the commits are pushed with `git push`.",
    "dvc5": "{fig:dvc5}: after deleting the data and the cache, `dvc pull` downloads the file again from DagsHub.",
    "dvc6": "{fig:dvc6} shows the toy repository on DagsHub with all three commits and the DVC-tracked CSV.",
    "dvc7": "{fig:dvc7} shows the pointer of the real dataset and `dvc status -c` reporting that my cache and the DagsHub remote are in sync.",
    "aug": "{fig:aug} shows one training batch before and after augmentation (flips, small rotations and zoom).",
    "cnnsum": "{fig:cnnsum} is the `model.summary()` output; the full table of output shapes and parameters per layer follows.",
    "cnncurves": "{fig:cnncurves} shows the fixed baseline: validation loss is lowest at epoch 16 (red line), after which it starts to overfit.",
    "tlcurves": "{fig:tlcurves} shows both stages: fine-tuning (right of the dotted line) clearly lowers the validation loss.",
    "r4curves": "{fig:r4curves} shows the curves of my final model with the best epoch marked in red.",
    "cm": "{fig:cm} shows the test confusion matrices – the high thresholds keep the number of real images called AI low.",
    "roc": "{fig:roc} shows the ROC curves; MobileNetV2 and R4 are clearly above the CNN.",
    "errors": "{fig:errors} shows the 10 most confident test errors with their size and format.",
    "gradcam": "{fig:gradcam} shows the input (top) and the regions that drove each prediction (bottom).",
    "mlruns": "{fig:mlruns} shows the two toy runs in my local MLflow UI.",
    "mlcompare": "{fig:mlcompare} compares the two learning rates side by side (parameters and metrics).",
    "dagshub": "{fig:dagshub} shows my DagsHub project with the files mirrored from GitHub.",
    "toyruns": "{fig:toyruns} shows the same two toy runs logged to the DagsHub MLflow server.",
    "gitignore": "{fig:gitignore} proves that `.env` (and the DVC credentials file) are in `.gitignore`.",
    "runs": "{fig:runs} lists all VisionGuard runs on DagsHub, each with its logged model.",
    "autolog": "{fig:autolog} shows the autolog run R2 with the automatically captured metrics plus the ones I added.",
    "compare": "{fig:compare} shows the DagsHub compare view: each run differs in one parameter (augment, img_size, model, no_fine_tune …).",
    "registry": "{fig:registry} shows the registry with three versions and the aliases @champion and @challenger.",
    "champion": "{fig:champion} shows the separate script loading the model by alias from DagsHub and classifying one image.",
    "demo": "{fig:demo} shows the console output: the model is loaded once, then two known images are classified correctly.",
    "e415": "{fig:e415}, {fig:e413} and {fig:e422} show the three failure cases with friendly error pages.",
    "upload": "{fig:upload} shows the upload page; {fig:resai} and {fig:resreal} show result pages for an AI and a real image.",
    "health": "{fig:health} shows the `/health` response with `debug: false` and the database ok.",
    "hist1": "{fig:hist1} shows the History menu with all columns (including Mode local/remote); {fig:hist2} shows record numbers that open a single record.",
    "w1": "{fig:w1} shows `docker run` with port mapping and a volume, `docker ps`, and the page served from my mounted folder.",
    "w2": "{fig:w2} shows `docker logs` with nginx starting and my GET request.",
    "w3": "{fig:w3} shows `docker exec` running commands inside the container.",
    "w4": "{fig:w4} shows `docker ps -a`, `docker stop` and `docker rm`; the image removal with `docker rmi` is in {fig:sizes}.",
    "compose": "{fig:compose} shows both services healthy, the db with `5432/tcp` and no host mapping, and the stop/start/restart commands.",
    "sizes": "{fig:sizes} shows both images side by side in `docker images`.",
    "ops3": "{fig:ops3} shows the state after the fix.",
    "citoy": "{fig:citoy} shows the three toy runs: green, red (bug on purpose) and green again.",
    "ciproj": "{fig:ciproj} shows that every push of VisionGuard passed CI.",
    "azure": "{fig:azure} shows the resource group with all Azure resources of the project.",
    "postman": "{fig:postman} shows the report of the collection run.",
    "networks": "{fig:networks} shows the check from four other countries.",
    "healthremote": "{fig:healthremote} shows `/health` in remote mode, including the status of the cloud API.",
    "remoteresult": "{fig:remoteresult} shows a result page produced by the cloud model (Inference: remote).",
}

PROBLEMS = {  # heading -> honest "what went wrong" line
    "### 1.1 Environment": "None in the setup itself; the TensorFlow download (~500 MB) was slow on my connection, so I ran the install in the background.",
    "### 1.2 Dataset audit": "`ls` first showed 538 / 435 files because it hides file names starting with a dot; the script counted 539 / 436 – I now count with code, not by eye.",
    "### 1.3 Visual EDA": "None – but the sample grid changed my view of the task (paintings are in the real class).",
    "### 1.4 DVC": "PowerShell 5.1 treated git's normal progress messages on stderr as errors and stopped my script, so I switched it to check exit codes instead. Uploading 480 MB to DagsHub took a long time on my connection.",
    "### 2.1 Input pipeline (`visionguard/data.py`)": "TensorFlow could not decode WEBP/CMYK files, so I used Pillow; palette images with transparency gave a warning, fixed by converting them to RGBA first.",
    "### 2.3 Transfer learning (MobileNetV2)": "In Keras 3 weight names are only `kernel`/`bias`, so I used `weight.path`; the stage-2 checkpoint could have overwritten a better stage-1 model, fixed with `initial_value_threshold`.",
    "### 2.4 Training discipline": "Even with fixed seeds the CPU runs are not exactly repeatable (R1 and R2 differ by 0.035 AUC), so I treat small differences as noise.",
    "### 2.5 Evaluation": "None; the important rule was to choose the threshold on validation data and touch the test set only once.",
    "### 2.6 Error analysis": "None; several errors come from noisy labels (banners and collages scraped from articles).",
    "### Bonus – Grad-CAM": "The backbone is a nested model, so I computed Grad-CAM by calling the backbone and the head separately inside a `GradientTape`.",
    "### 2.7 Keras Tuner": "Each trial trains MobileNetV2 on the CPU, so I kept trials short (4 + 6 epochs); short training makes the scores noisier.",
    "### 3.2 DagsHub setup": "MLflow prints a 🏃 emoji after each run and the Windows console (cp1252) crashed; fixed by switching output to UTF-8. The repository first came out private and had to be made public.",
    "### 3.4 Artifacts": "None.",
    "### 3.5 Experiment comparison": "In my run script the shared options came after the per-run options, so R9 used 25 epochs / patience 6 instead of 50 / 10 (argparse keeps the last value); it still converged (best epoch 16).",
    "### 4.1 Inference module (`visionguard/inference.py`)": "None; the first prediction is slower, which is why one warm-up call runs at start-up.",
    "### 4.2 Routes and validation": "None – all cases are covered by 7 automated tests that run in CI.",
    "### 4.3 HTML and CSS": "None.",
    "### 4.4 Configuration and production server": "None.",
    "### 6.1 Docker warm-up (`scripts/docker_warmup.ps1`, nginx:alpine)": "None.",
    "### 6.2 Dockerfile for the web app": "The first build failed with `short read: unexpected EOF` because my connection dropped while downloading TensorFlow; retrying worked.",
    "### 6.3 Compose stack": "None.",
    "### 6.4 Image analysis": "The 'before' build was interrupted by a network drop and had to be restarted.",
    "### 6.6 GitHub Actions (CI)": "None – the project workflow was green on the first push.",
    "## Task 5 – Database Trail": "None; rows 1–2 survived re-creating the db container, which confirmed the named volume works.",
    "## Task 7 – API Deployment (Microsoft Azure, pathway C)": "The free trial blocks `az acr build`; `az acr build` also ignored my `.dockerignore` and tried to upload 1 GB, so I built locally from a minimal context instead. My connection dropped (DNS error) during `docker push`; retrying skipped the layers already uploaded.",
    "## Task 8 – Integrating Flask and the API": "After changing `INFERENCE_MODE` in `.env` I first forgot to recreate the web container, so some uploads still ran locally (records 13–15). I wrote `scripts/set_mode.ps1` to switch and restart in one step.",
}

COMMANDS_61 = """- **Commands:** `docker pull nginx:alpine` downloads the image; `docker run -d --name warmup-nginx -p 8081:80 -v <folder>:/usr/share/nginx/html:ro nginx:alpine` starts it in the background, maps port 8081 → 80 and mounts my folder read-only; `docker ps` lists running containers; `docker logs` prints the container's output; `docker exec … sh -c "…"` runs commands inside it; `docker stop` / `docker rm` stop and delete the container; `docker images` / `docker rmi` list and delete images.
"""
COMMANDS_65 = """- **Commands:** `docker compose up -d` creates and starts both services in the background; `ps` shows status and health; `stop` / `start` stop and restart the existing containers (data kept); `restart web` restarts one service; `logs --tail N <service>` shows recent output; `exec <service> sh -c …` runs a command inside a service (`whoami`, `psql`).
"""


def main():
    t = P.read_text(encoding="utf-8")
    # 1) lead-in sentence before every figure that is not referenced yet
    out = []
    for ln in t.splitlines():
        m = re.fullmatch(r"!\[label=([\w-]+);.*", ln.strip())
        if m and f"{{fig:{m.group(1)}}}" not in t and m.group(1) in LEAD:
            out += [LEAD[m.group(1)], ""]
            t += f"{{fig:{m.group(1)}}}"          # mark as referenced (avoid double lead-ins for grouped refs)
            for ref in re.findall(r"\{fig:([\w-]+)\}", LEAD[m.group(1)]):
                t += f"{{fig:{ref}}}"
        out.append(ln)
    text = "\n".join(out) + "\n"
    # 2) commands
    text = text.replace("- **Image** – read-only template", COMMANDS_61 + "- **Image** – read-only template", 1)
    text = text.replace("![label=compexec;", COMMANDS_65 + "\n![label=compexec;", 1)
    # 3) problems line at the end of each listed section
    lines, res, current = text.splitlines(), [], None

    def close():
        if current in PROBLEMS:
            while res and not res[-1].strip():
                res.pop()
            res.extend(["", f"- **Problems:** {PROBLEMS[current]}", ""])

    for ln in lines:
        if ln.startswith("## ") or ln.startswith("### "):
            close()
            current = ln.strip()
        res.append(ln)
    close()
    P.write_text("\n".join(res) + "\n", encoding="utf-8")
    missing = [h for h in PROBLEMS if h not in text]
    print("headings not found:", missing)


if __name__ == "__main__":
    main()
