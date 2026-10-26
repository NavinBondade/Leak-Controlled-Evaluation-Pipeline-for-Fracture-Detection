from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fracai.split import assign_folds, fracatlas_groups

RES = Path(__file__).resolve().parents[1] / "results"


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "case_id": [f"IMG{i:07d}" for i in range(n)],
            "y": rng.integers(0, 2, n),
            "bodypart": rng.choice(["hand", "leg"], n),
            "multiscan": 0,
        }
    )
    return df


def test_group_never_spans_folds():
    df = _toy()
    df["group"] = np.arange(len(df)) // 3  # 3 images per "patient"
    df["fold"] = assign_folds(df, "group")
    assert (df.groupby("group").fold.nunique() == 1).all()
    assert set(df.fold) == set(range(5))


def test_near_duplicates_share_group():
    df = _toy(50)
    rng = np.random.default_rng(1)
    h = rng.integers(0, 2, (50, 256)).astype(bool)
    h[7] = h[3]  # exact duplicate
    h[9] = h[3].copy()
    h[9][:10] ^= True  # 10 bits away, under threshold
    g = fracatlas_groups(df, h, max_dist=20)
    assert g[3] == g[7] == g[9]
    assert g[3] != g[4]


def test_consecutive_multiscan_same_bodypart_grouped():
    df = _toy(10)
    df["bodypart"] = "hand"
    df.loc[[2, 3, 4], "multiscan"] = 1
    h = np.random.default_rng(2).integers(0, 2, (10, 256)).astype(bool)
    g = fracatlas_groups(df, h)
    assert g[2] == g[3] == g[4]
    assert g[1] != g[2] and g[5] != g[4]


@pytest.mark.parametrize("name", ["fracatlas", "graz"])
def test_saved_splits_have_no_leakage(name):
    f = RES / f"{name}_meta.csv"
    if not f.exists():
        pytest.skip("run scripts/data/make_split.py first")
    df = pd.read_csv(f)
    assert (df.groupby("group").fold.nunique() == 1).all()
    prev = df.groupby("fold").y.mean()
    assert prev.max() - prev.min() < 0.05
    if name == "graz":  # patient is the group: no patient in two folds
        assert (df.groupby("patient_id").fold.nunique() == 1).all()
