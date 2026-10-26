#!/usr/bin/env bash
# Candidate 2: oversample smallest-quartile fractured images x3 (no aux head). Starts after run_fine.sh. Resumable, offline.
cd "$(dirname "$0")/.."
export PYTHONPATH=. PYTHONWARNINGS=ignore HF_HUB_OFFLINE=1
PY=.venv/Scripts/python
until grep -q "all done" results/run_fine.out 2>/dev/null; do sleep 30; done
for attempt in $(seq 1 20); do
  $PY -u scripts/train/train_cv.py fracatlas --epochs 12 --small-rep 3 --tag _small3 >> results/train_fracatlas_small3.log 2>&1 && break
  echo "[run_small] attempt $attempt failed at $(date); retrying in 60 s" | tee -a results/train_fracatlas_small3.log
  sleep 60
done
echo "[run_small] all done $(date)"
