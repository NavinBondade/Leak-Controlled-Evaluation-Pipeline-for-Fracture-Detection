"""Five-fold training entry point. See --help for options."""

import argparse
import os

os.environ["HF_HUB_OFFLINE"] = "1"  # offline: weights come from weights/*.pth

import numpy as np
import pandas as pd

from fracai.train import load_cache, run_cv

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--smoke", action="store_true", help="tiny run to prove the code works")
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--arch", default="convnext_tiny")
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--tag", default="")
ap.add_argument("--fold-col", default="fold", help="fold column in the meta csv; fold_naive = leaky random split")
ap.add_argument(
    "--box-frac",
    type=float,
    default=0.0,
    help="share of fractured training images whose box supervises the localisation head (0 = off)",
)
ap.add_argument(
    "--aux",
    action="store_true",
    help="aux head on even with --box-frac 0: only negatives supervise it (all-empty map) - the control for box information",
)
ap.add_argument(
    "--aux-kind",
    default="coarse",
    choices=["coarse", "fine"],
    help="fine = 32x32 pyramid localisation head (16 px cells)",
)
ap.add_argument(
    "--small-rep",
    type=int,
    default=1,
    help="oversample smallest-quartile fractured training images this many times per epoch",
)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--box-lam", type=float, default=1.0)
ap.add_argument("--box-seed", type=int, default=0)
a = ap.parse_args()

df = pd.read_csv(f"results/{a.name}_meta.csv")
df["fold"] = df[a.fold_col]
arr = load_cache(a.name, 512)
aux = None
if a.box_frac > 0 or a.aux:
    masks, has = np.load("results/fracatlas_boxmask64.npy", mmap_mode="r"), np.load("results/fracatlas_hasbox.npy")
    area = pd.read_csv("results/fracatlas_boxes.csv").set_index("case_id").rel_area.reindex(df.case_id).values
    cand = np.where((df.y.values == 1) & has)[0]
    use = np.zeros(len(df), bool)
    use[np.random.default_rng(a.box_seed).choice(cand, int(round(a.box_frac * len(cand))), replace=False)] = True
    sw = np.ones(len(df), np.float32)  # small fractures count more
    ok = use & ~np.isnan(area)
    sw[ok] = np.clip(np.sqrt(np.median(area[ok]) / area[ok]), 0.5, 3.0)
    aux = dict(masks=masks, known=(df.y.values == 0) | use, sw=sw, lam=a.box_lam, kind=a.aux_kind)
    print(
        f"box supervision on {int(use.sum())} of {len(cand)} fractured images (frac {a.box_frac}), lam {a.box_lam}",
        flush=True,
    )
small = None
if a.small_rep > 1:
    area = pd.read_csv("results/fracatlas_boxes.csv").set_index("case_id").rel_area.reindex(df.case_id).values
    q1 = np.nanquantile(area[df.y.values == 1], 0.25)
    small = (df.y.values == 1) & (area <= q1)
    print(f"oversampling {int(small.sum())} smallest-quartile fractured images x{a.small_rep}", flush=True)
if a.smoke:
    pos = sorted(df.groupby("fold").sample(80, random_state=0).index)
    arr, df = np.asarray(arr[pos]), df.loc[pos].reset_index(drop=True)
    if aux:
        aux = dict(
            masks=np.asarray(aux["masks"][pos]),
            known=aux["known"][pos],
            sw=aux["sw"][pos],
            lam=aux["lam"],
            kind=aux["kind"],
        )
    run_cv(
        a.name + "_smoke" + ("_box" if aux else ""),
        df,
        arr,
        folds=[0],
        arch=a.arch,
        epochs=1,
        bs=8,
        out_size=224,
        aux=aux,
    )
else:
    run_cv(
        a.name + a.tag,
        df,
        arr,
        arch=a.arch,
        epochs=a.epochs,
        bs=a.bs,
        out_size=a.size,
        aux=aux,
        seed=a.seed,
        small=small,
        small_rep=a.small_rep,
    )
