#!/usr/bin/env bash
# Controls for the box experiment, run after run_box.sh: (a) aux head supervised by negatives only (no box information),
# (b) baseline retrained with another seed (run-to-run noise). Resumable, offline.
cd "$(dirname "$0")/.."
export PYTHONPATH=. PYTHONWARNINGS=ignore HF_HUB_OFFLINE=1
PY=.venv/Scripts/python
until grep -q "all done" results/run_box.out 2>/dev/null; do sleep 30; done
run() {
  log=$1; shift
  for attempt in $(seq 1 20); do
    $PY -u scripts/train/train_cv.py fracatlas --epochs 12 "$@" >> "$log" 2>&1 && return 0
    echo "[run_controls] attempt $attempt failed at $(date); retrying in 60 s" | tee -a "$log"
    sleep 60
  done
  return 1
}
run results/train_fracatlas_box0.log --aux --box-frac 0 --tag _box0 || exit 1
run results/train_fracatlas_s43.log --seed 43 --tag _s43 || exit 1
echo "[run_controls] all done $(date)"
