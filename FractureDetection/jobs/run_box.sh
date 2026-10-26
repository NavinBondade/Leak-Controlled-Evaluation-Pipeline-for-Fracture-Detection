#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export PYTHONPATH=. PYTHONWARNINGS=ignore HF_HUB_OFFLINE=1
PY=.venv/Scripts/python
run() {
  log=$1; shift
  for attempt in $(seq 1 20); do
    $PY -u scripts/train/train_cv.py fracatlas --epochs 12 "$@" >> "$log" 2>&1 && return 0
    echo "[run_box] attempt $attempt failed at $(date); retrying in 60 s" | tee -a "$log"
    sleep 60
  done
  return 1
}
run results/train_fracatlas_box100.log --box-frac 1.0  --tag _box100 || exit 1
run results/train_fracatlas_box10.log  --box-frac 0.10 --tag _box10  || exit 1
run results/train_fracatlas_box25.log  --box-frac 0.25 --tag _box25  || exit 1
echo "[run_box] all done $(date)"
