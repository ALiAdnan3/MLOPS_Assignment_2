#!/usr/bin/env bash
# Task 3.3 / 3.5 - experiment series logged to DagsHub MLflow.
# Each run differs from the reference run (R1) in exactly ONE setting.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
COMMON="--epochs 25 --fine-tune-epochs 20 --patience 6"
run() { echo "=== $1 ($(date +%H:%M)) ==="; $PY -m src.train --run-name "$@" $COMMON > "outputs/$1.log" 2>&1 && grep -E '"test_(accuracy|roc_auc)"' "outputs/$1/summary.json" | tr -d '\n'; echo; }

run R1_mnv2_reference      --model mobilenetv2 --lr 1e-3 --augment light --img-size 224 --mlflow manual
run R2_mnv2_autolog        --model mobilenetv2 --lr 1e-3 --augment light --img-size 224 --mlflow autolog
run R3_mnv2_lr3e-4         --model mobilenetv2 --lr 3e-4 --augment light --img-size 224 --mlflow manual
run R4_mnv2_no_augment     --model mobilenetv2 --lr 1e-3 --augment none  --img-size 224 --mlflow manual
run R5_mnv2_strong_augment --model mobilenetv2 --lr 1e-3 --augment strong --img-size 224 --mlflow manual
run R6_mnv2_frozen_only    --model mobilenetv2 --lr 1e-3 --augment light --img-size 224 --mlflow manual --no-fine-tune
run R7_mnv2_img160         --model mobilenetv2 --lr 1e-3 --augment light --img-size 160 --mlflow manual
run R8_effnetb0            --model efficientnetb0 --lr 1e-3 --augment light --img-size 224 --mlflow manual
run R9_baseline_cnn        --model cnn --lr 1e-3 --dropout 0.4 --augment light --img-size 224 --mlflow manual --epochs 50 --patience 10
echo "ALL DONE $(date +%H:%M)"
