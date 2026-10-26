"""Contact sheets for a visual cast / hardware audit of FracAtlas out-of-fold scores. CPU only, no model.

Sets (scores shifted so 0 = fold threshold at 90% validation specificity, as in scripts/audit/slide_grids.py):
  fa_top60   : 60 highest-scoring non-fractured images
  tn_rand60  : 60 random correctly rejected non-fractured images (score < 0, seed 3)
  fx_low60   : 60 lowest-scoring fractured images
Outputs (results/audit2/):
  sheet_<set>.png        labelled 6 x 10 contact sheets (deliverable)
  blind/page_XX.png      the same 180 images shuffled, 12 per page, labelled only by a blind code (for reading)
  blind_key.csv          blind code -> case_id, set, score, metadata
Run: .venv\\Scripts\\python.exe scripts\\analysis\\cast_contact_sheets.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPU is owned by another job

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
(OUT / "blind").mkdir(parents=True, exist_ok=True)

oof, val = pd.read_csv("results/oof_fracatlas.csv"), pd.read_csv("results/val_fracatlas.csv")
thr = {k: thr_at_spec(v.y, v.score, 0.90) for k, v in val.groupby("test_fold")}
oof["score"] -= oof.fold.map(thr)
meta = pd.read_csv("results/fracatlas_meta.csv")
row_of = {c: i for i, c in enumerate(meta.case_id)}
arr = load_cache("fracatlas", 512)

neg, pos = oof[oof.y == 0], oof[oof.y == 1]
sets = {
    "fa_top60": (
        "Highest-scoring 60 non-fractured images (false alarms)",
        neg.sort_values("score", ascending=False).head(60),
    ),
    "tn_rand60": (
        "Random 60 correctly rejected non-fractured images (seed 3)",
        neg[neg.score < 0].sample(60, random_state=3),
    ),
    "fx_low60": ("Lowest-scoring 60 fractured images (misses)", pos.sort_values("score").head(60)),
}


def tile(ax, cid):
    ax.imshow(arr[row_of[cid]], cmap="gray", vmin=0, vmax=255)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


for key, (title, g) in sets.items():
    fig, axes = plt.subplots(6, 10, figsize=(20, 13.6), gridspec_kw={"wspace": 0.04, "hspace": 0.22})
    for ax, (i, (_, r)) in zip(axes.flat, enumerate(g.iterrows())):
        tile(ax, r.case_id)
        hw = " HW" if r.hardware else ""
        ax.set_title(
            f"{i + 1:02d} {r.case_id[3:]}  {r.score:+.1f}\n{r.bodypart}{hw}",
            fontsize=8,
            fontweight="normal",
            loc="center",
        )
    fig.suptitle(
        f"FracAtlas: {title}. Label: rank, case id, shifted out-of-fold score; HW = metadata hardware flag",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(OUT / f"sheet_{key}.png", bbox_inches="tight", dpi=110)
    plt.close(fig)

# blind reading pages: shuffle all 180, show code only
key = pd.concat([g.assign(set=k) for k, (_, g) in sets.items()])
key = key.sample(frac=1, random_state=11).reset_index(drop=True)
key["code"] = [f"B{i + 1:03d}" for i in range(len(key))]
key[["code", "case_id", "set", "y", "score", "bodypart", "view", "hardware"]].to_csv(OUT / "blind_key.csv", index=False)
for p in range(0, len(key), 12):
    fig, axes = plt.subplots(3, 4, figsize=(12, 9.4), gridspec_kw={"wspace": 0.03, "hspace": 0.1})
    for ax, (_, r) in zip(axes.flat, key.iloc[p : p + 12].iterrows()):
        tile(ax, r.case_id)
        ax.set_title(r.code, fontsize=10, fontweight="normal", loc="center")
    fig.savefig(OUT / "blind" / f"page_{p // 12 + 1:02d}.png", bbox_inches="tight", dpi=100)
    plt.close(fig)
print(
    {k: (len(g), round(g.score.min(), 2), round(g.score.max(), 2), int(g.hardware.sum())) for k, (_, g) in sets.items()}
)
