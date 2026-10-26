"""Build the leakage-controlled fold assignments for both datasets."""
import numpy as np

from fracai.data import load_fracatlas, load_graz
from sklearn.model_selection import StratifiedKFold

from fracai.split import assign_folds, fold_summary, fracatlas_groups

SEED, MAX_DIST = 42, 30

fa = load_fracatlas()
h = np.load("results/fracatlas_phash.npy")
fa["group"] = fracatlas_groups(fa, h, max_dist=MAX_DIST)
fa["fold"] = assign_folds(fa, "group", seed=SEED)
# naive comparison split: random stratified by image, ignoring duplicate groups (the leaky way people often split)
naive = np.full(len(fa), -1)
for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(fa, fa.y.astype(str) + "_" + fa.bodypart)):
    naive[te] = k
fa["fold_naive"] = naive
fa.drop(columns="path").to_csv("results/fracatlas_meta.csv", index=False)

sz = fa.groupby("group").size()
print(f"FracAtlas: {len(fa)} images -> {sz.size} groups (phash max_dist={MAX_DIST}/256 bits + consecutive multiscan)")
print("group size: max", sz.max(), "| groups with >1 image:", (sz > 1).sum(), "| images in such groups:", sz[sz > 1].sum())
print("size quantiles:", sz.quantile([.5, .9, .99]).to_dict())
print(fold_summary(fa).round(3), "\n")

gz = load_graz()
gz["group"] = gz.patient_id
gz["fold"] = assign_folds(gz, "group", seed=SEED)
gz.drop(columns="path").to_csv("results/graz_meta.csv", index=False)
print(f"GRAZPEDWRI-DX: {len(gz)} images, {gz.group.nunique()} patients")
print(fold_summary(gz).round(3))
