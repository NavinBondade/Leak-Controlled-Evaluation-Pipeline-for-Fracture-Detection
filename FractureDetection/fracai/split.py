"""Leakage-controlled fold assignment."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


class _DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def phashes(paths, size=16):
    import imagehash
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True  # a few FracAtlas jpgs are truncated; see results/truncated_images.txt
    out = []
    for p in paths:
        with Image.open(p) as im:
            out.append(imagehash.phash(im.convert("L"), hash_size=size).hash.flatten())
    return np.array(out)


def fracatlas_groups(df: pd.DataFrame, hashes: np.ndarray, max_dist: int = 20) -> pd.Series:
    """hashes: (n, 256) bool array from phashes(). max_dist is in bits of 256."""
    n = len(df)
    dsu = _DSU(n)
    h = hashes.astype(np.uint8)
    for i in range(n):
        d = (h[i + 1 :] != h[i]).sum(axis=1)
        for j in np.nonzero(d <= max_dist)[0]:
            dsu.union(i, i + 1 + j)
    for i in range(n - 1):
        if df.multiscan.iat[i] and df.multiscan.iat[i + 1] and df.bodypart.iat[i] == df.bodypart.iat[i + 1]:
            dsu.union(i, i + 1)
    roots = pd.Series([dsu.find(i) for i in range(n)], index=df.index)
    return roots.map({r: k for k, r in enumerate(roots.unique())}).rename("group")


def assign_folds(df: pd.DataFrame, group_col: str, n_folds: int = 5, seed: int = 42) -> pd.Series:
    strat = df.y.astype(str) + "_" + df.bodypart.astype(str)
    fold = np.full(len(df), -1)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, test_idx) in enumerate(sgkf.split(df, strat, groups=df[group_col])):
        fold[test_idx] = k
    return pd.Series(fold, index=df.index, name="fold")


def fold_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("fold").agg(
        n=("y", "size"), pos=("y", "sum"), prevalence=("y", "mean"), hardware=("hardware", "sum")
    )
