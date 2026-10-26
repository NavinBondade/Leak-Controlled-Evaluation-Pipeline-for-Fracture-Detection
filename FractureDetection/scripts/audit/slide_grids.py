"""Image grids with Grad-CAM overlays for slides: detected, missed, small fractures, false alarms. Output: results/figures/slides/"""

from pathlib import Path

import matplotlib
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from fracai.data import load_boxes
from fracai.gradcam import gradcam
from fracai.metrics import thr_at_spec
from fracai.plotstyle import use
from fracai.train import build_model, dev, load_cache, to_batch

use()
OUT = Path("results/figures/slides")
OUT.mkdir(parents=True, exist_ok=True)
N = 6
rng_seed = 7

oof, val = pd.read_csv("results/oof_fracatlas.csv"), pd.read_csv("results/val_fracatlas.csv")
thr = {k: thr_at_spec(v.y, v.score, 0.90) for k, v in val.groupby("test_fold")}
oof["score"] -= oof.fold.map(thr)
meta = pd.read_csv("results/fracatlas_meta.csv")
row_of = {c: i for i, c in enumerate(meta.case_id)}
arr = load_cache("fracatlas", 512)
size = pd.read_csv("results/fracatlas_boxes.csv").set_index("case_id")
oof["rel_area"] = oof.case_id.map(size.rel_area)
small_cut = oof[oof.y == 1].rel_area.quantile(0.25)

boxes = load_boxes()

pos = oof[(oof.y == 1) & oof.case_id.isin(boxes)]
small = pos[pos.rel_area <= small_cut]
groups = {
    "detected": ("Correctly detected fractures (random sample)", pos[pos.score >= 0].sample(N, random_state=rng_seed)),
    "missed": ("Missed fractures (random sample)", pos[pos.score < 0].sample(N, random_state=rng_seed)),
    "small": (
        "Smallest-quartile fractures: top three detected, bottom three missed (random sample)",
        pd.concat(
            [
                small[small.score >= 0].sample(3, random_state=rng_seed),
                small[small.score < 0].sample(3, random_state=rng_seed),
            ]
        ),
    ),
    "false_alarms": (
        "False alarms: the six highest-scoring non-fractured images",
        oof[(oof.y == 0)].sort_values("score", ascending=False).head(N),
    ),
}

need = pd.concat([g for _, g in groups.values()]).drop_duplicates("case_id")
cams = {}
for k, grp in need.groupby("fold"):
    model = build_model("convnext_tiny").to(dev)
    model.load_state_dict(torch.load(f"results/model_fracatlas_fold{int(k)}.pt", map_location=dev))
    for cid in grp.case_id:
        cams[cid], _ = gradcam(model, to_batch(arr, [row_of[cid]], False, 512))
    del model
    torch.cuda.empty_cache()

for key, (title, g) in groups.items():
    fig, axes = plt.subplots(2, N, figsize=(10, 3.9), gridspec_kw={"wspace": 0.04, "hspace": 0.06})
    for j, (_, r) in enumerate(g.iterrows()):
        top, bottom = axes[0, j], axes[1, j]
        for ax in (top, bottom):
            ax.imshow(arr[row_of[r.case_id]], cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        bottom.imshow(cams[r.case_id], cmap="jet", alpha=0.4, vmin=0, vmax=1)
        for bx, by, bw, bh in boxes.get(r.case_id, []):
            top.add_patch(Rectangle((bx, by), bw, bh, fill=False, ec="lime", lw=1.5))
        top.set_title(f"{r.bodypart}, {r.score:+.1f}", fontsize=9, fontweight="normal", loc="center")
    axes[0, 0].set_ylabel("Image", fontsize=10)
    axes[1, 0].set_ylabel("Grad-CAM", fontsize=10)
    fig.savefig(OUT / f"grid_{key}.png", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("saved", OUT / f"grid_{key}.png")
