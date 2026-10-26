#!/usr/bin/env bash
# Fine (32x32 pyramid) localisation head: boxes on 100% of fractured training images, then its matching control
# (same head, negatives only). Starts when run_controls.sh has finished. Resumable, offline.
cd "$(dirname "$0")/.."
export PYTHONPATH=. PYTHONWARNINGS=ignore HF_HUB_OFFLINE=1
PY=.venv/Scripts/python
until grep -q "all done" results/run_controls.out 2>/dev/null; do sleep 30; done
run() {
  log=$1; shift
  for attempt in $(seq 1 20); do
    $PY -u scripts/train/train_cv.py fracatlas --epochs 12 "$@" >> "$log" 2>&1 && return 0
    echo "[run_fine] attempt $attempt failed at $(date); retrying in 60 s" | tee -a "$log"
    sleep 60
  done
  return 1
}
run results/train_fracatlas_fine100.log --aux-kind fine --box-frac 1.0 --tag _fine100 || exit 1
run results/train_fracatlas_fine0.log   --aux-kind fine --aux --box-frac 0 --tag _fine0 || exit 1
echo "[run_fine] all done $(date)"
