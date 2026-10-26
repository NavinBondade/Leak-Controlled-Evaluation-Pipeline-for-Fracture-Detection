"""Shared helpers for the audit2 (B) jobs: GRAZ table with initial_exam, fold models, thresholds, patient bootstrap.
Inference only: models are loaded in eval mode and never trained."""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fracai.data import GRAZ
from fracai.metrics import thr_at_spec
from fracai.train import build_model, dev, load_cache

OUT = Path("results/audit2")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 0


def graz_table():
    """ext_graz_scores.csv + initial_exam (1 = first presentation, 0/blank = follow-up), flag = ensemble score >= 0.
    Row order matches the graz_512 cache."""
    raw = pd.read_csv(GRAZ / "dataset.csv", usecols=["filestem", "initial_exam"]).rename(
        columns={"filestem": "case_id"}
    )
    d = pd.read_csv("results/ext_graz_scores.csv")
    meta = pd.read_csv("results/graz_meta.csv")
    assert (d.case_id.values == meta.case_id.values).all()
    d = d.merge(raw, on="case_id", how="left")
    assert (d.case_id.values == meta.case_id.values).all()
    d["initial_exam"] = d.initial_exam.fillna(0).astype(int)
    d["followup"] = 1 - d.initial_exam
    d["flag"] = (d.score >= 0).astype(int)
    d["row"] = np.arange(len(d))
    return d


def thresholds():
    val = pd.read_csv("results/val_fracatlas.csv")
    return [thr_at_spec(val[val.test_fold == k].y, val[val.test_fold == k].score, 0.90) for k in range(5)]


def load_fold(k):
    m = build_model("convnext_tiny").to(dev)
    m.load_state_dict(torch.load(f"results/model_fracatlas_fold{k}.pt", map_location=dev))
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def graz_arr():
    return load_cache("graz", 512)


def stratified_sample(d, quotas, seed=SEED):
    """quotas: list of (mask, n). Takes min(n, available) rows from each stratum at random (fixed seed)."""
    rng = np.random.default_rng(seed)
    parts = []
    for mask, n in quotas:
        idx = d.index[mask].values
        parts.append(rng.choice(idx, size=min(n, len(idx)), replace=False))
    return d.loc[np.unique(np.concatenate(parts))].copy()


def patient_boot(df, fn, n=200, seed=SEED, col="patient_id"):
    """Resample patients with replacement. Returns (point, lo, hi)."""
    point = fn(df)
    codes, uniq = pd.factorize(df[col])
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([order[bounds[g] : bounds[g + 1]] for g in pick])
        v = fn(df.iloc[idx])
        if v is not None and not np.isnan(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)
    return point, lo, hi


def fmt(p, nd=3):
    return f"{p[0]:.{nd}f} ({p[1]:.{nd}f} to {p[2]:.{nd}f})"
