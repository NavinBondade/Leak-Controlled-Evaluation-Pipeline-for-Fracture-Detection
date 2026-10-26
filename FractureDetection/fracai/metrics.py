"""Sensitivity, specificity, AUC and group-bootstrap intervals."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def sens_spec(y, s, thr):
    y, pred = np.asarray(y), np.asarray(s) >= thr
    p, n = (y == 1).sum(), (y == 0).sum()
    return (pred[y == 1].sum() / p if p else np.nan), ((~pred)[y == 0].sum() / n if n else np.nan)


def thr_at_spec(y, s, spec=0.90):
    """Lowest threshold whose specificity on (y, s) is >= spec. Fit on validation data only."""
    neg = np.sort(np.asarray(s)[np.asarray(y) == 0])
    return np.quantile(neg, spec, method="higher")


def thr_youden(y, s):
    cand = np.unique(s)
    j = [sum(sens_spec(y, s, t)) - 1 for t in cand]
    return cand[int(np.argmax(j))]


def safe_auc(y, s):
    y = np.asarray(y)
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def group_bootstrap(df, stat_fn, group_col, n_boot=1000, seed=0):
    """Resample whole groups with replacement. stat_fn(sub_df) -> float. Returns (point, lo, hi)."""
    point = stat_fn(df)
    codes, uniq = pd.factorize(df[group_col])
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([order[bounds[g] : bounds[g + 1]] for g in pick])
        v = stat_fn(df.iloc[idx])
        if not np.isnan(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)
    return point, lo, hi


def ece_brier(y, logit, bins=10):
    p = 1 / (1 + np.exp(-np.asarray(logit)))
    y = np.asarray(y)
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece = sum((b == i).mean() * abs(y[b == i].mean() - p[b == i].mean()) for i in range(bins) if (b == i).any())
    return ece, float(np.mean((p - y) ** 2))
