#!/usr/bin/env bash
# Trains the controlled-split and naive-split 5-fold runs. Safe to re-run after any crash or reboot:
# finished folds are skipped and an interrupted fold resumes from its last finished epoch. Needs no internet.
# A crash (e.g. Windows running out of commit memory because other apps are open) is retried automatically.
cd "$(dirname "$0")/.."
export PYTHONPATH=. PYTHONWARNINGS=ignore HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/Scripts/python

run() {  # run <logfile> <extra args...>; retry up to 20 times, resuming each time
  log=$1; shift
  for attempt in $(seq 1 20); do
    $PY -u scripts/train/train_cv.py fracatlas --epochs 12 "$@" >> "$log" 2>&1 && return 0
    echo "[run_training] attempt $attempt failed at $(date); retrying in 60 s (resumes from checkpoint)" | tee -a "$log"
    sleep 60
  done
  return 1
}

run results/train_fracatlas.log                                          || exit 1
run results/train_fracatlas_naive.log --fold-col fold_naive --tag _naive || exit 1
echo "[run_training] all done $(date)"
