#!/usr/bin/env bash
# Shortcut audit: table analyses first (no GPU), then the three model-inference jobs, then the figures.
# Reads results/ext_graz_scores.csv and the saved fold models. No training, no network.
cd "$(dirname "$0")/.."
export PYTHONPATH=. PYTHONWARNINGS=ignore HF_HUB_OFFLINE=1
PY=.venv/Scripts/python
mkdir -p results/audit2
for s in a1_cast_within_patient a2_age_sex a3_device; do
  $PY -u scripts/analysis/$s.py > results/audit2/${s%%_*}.log 2>&1 || exit 1
done
for s in b1_feature_probe b2_cam_cast b3_perturb; do
  $PY -u scripts/audit/$s.py > results/audit2/${s%%_*}_log.txt 2>&1 || exit 1
done
$PY -u scripts/analysis/audit_figures.py || exit 1
echo "[run_audits] all done $(date)"
