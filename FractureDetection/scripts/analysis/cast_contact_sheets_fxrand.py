"""Supplement to cast_contact_sheets.py: 60 random fractured FracAtlas images (seed 3), to estimate how common casts/splints
are among fractured training images in general (the low-scoring set is not representative). CPU only.
Outputs: results/audit2/sheet_fx_rand60.png, results/audit2/blind_fx/page_XX.png, results/audit2/blind_key_fx.csv
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fracai.metrics import thr_at_spec
from fracai.plotstyle import use
from fracai.train import load_cache

use()
OUT = Path("results/audit2")
(OUT / "blind_fx").mkdir(parents=True, exist_ok=True)
oof, val = pd.read_csv("results/oof_fracatlas.csv"), pd.read_csv("results/val_fracatlas.csv")
thr = {k: thr_at_spec(v.y, v.score, 0.90) for k, v in val.groupby("test_fold")}
oof["score"] -= oof.fold.map(thr)
meta = pd.read_csv("results/fracatlas_meta.csv")
row_of = {c: i for i, c in enumerate(meta.case_id)}
arr = load_cache("fracatlas", 512)
g = oof[oof.y == 1].sample(60, random_state=3).sort_values("score", ascending=False)


def tile(ax, cid):
    ax.imshow(arr[row_of[cid]], cmap="gray", vmin=0, vmax=255)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


fig, axes = plt.subplots(6, 10, figsize=(20, 13.6), gridspec_kw={"wspace": 0.04, "hspace": 0.22})
for ax, (i, (_, r)) in zip(axes.flat, enumerate(g.iterrows())):
    tile(ax, r.case_id)
    ax.set_title(
        f"{i + 1:02d} {r.case_id[3:]}  {r.score:+.1f}\n{r.bodypart}{' HW' if r.hardware else ''}",
        fontsize=8,
        fontweight="normal",
        loc="center",
    )
fig.suptitle(
    "FracAtlas: random 60 fractured images (seed 3). Label: rank, case id, shifted out-of-fold score; HW = metadata hardware flag",
    x=0.01,
    ha="left",
    fontsize=12,
    fontweight="bold",
)
fig.savefig(OUT / "sheet_fx_rand60.png", bbox_inches="tight", dpi=110)
plt.close(fig)

key = g.sample(frac=1, random_state=12).reset_index(drop=True)
key["code"] = [f"F{i + 1:03d}" for i in range(len(key))]
key["set"] = "fx_rand60"
key[["code", "case_id", "set", "y", "score", "bodypart", "view", "hardware"]].to_csv(
    OUT / "blind_key_fx.csv", index=False
)
for p in range(0, len(key), 12):
    fig, axes = plt.subplots(3, 4, figsize=(12, 9.4), gridspec_kw={"wspace": 0.03, "hspace": 0.1})
    for ax, (_, r) in zip(axes.flat, key.iloc[p : p + 12].iterrows()):
        tile(ax, r.case_id)
        ax.set_title(r.code, fontsize=10, fontweight="normal", loc="center")
    fig.savefig(OUT / "blind_fx" / f"page_{p // 12 + 1:02d}.png", bbox_inches="tight", dpi=100)
    plt.close(fig)
print(len(g), int(g.hardware.sum()), round((g.score >= 0).mean(), 2))
