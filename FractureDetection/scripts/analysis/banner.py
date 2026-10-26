"""Header image for the README: title, headline numbers and four example cases. Output: results/figures/slides/banner.png"""

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
from fracai.plotstyle import BLUE, GREY, ORANGE, RED, use
from fracai.train import build_model, dev, load_cache, to_batch

use()
OUT = Path("results/figures/slides")
oof, val = pd.read_csv("results/oof_fracatlas.csv"), pd.read_csv("results/val_fracatlas.csv")
thr = {k: thr_at_spec(v.y, v.score, 0.90) for k, v in val.groupby("test_fold")}
oof["score"] -= oof.fold.map(thr)
meta = pd.read_csv("results/fracatlas_meta.csv")
row_of = {c: i for i, c in enumerate(meta.case_id)}
arr = load_cache("fracatlas", 512)
boxes = load_boxes()
area = pd.read_csv("results/fracatlas_boxes.csv").set_index("case_id").rel_area
oof["rel_area"] = oof.case_id.map(area)
pos = oof[(oof.y == 1) & oof.case_id.isin(boxes)]
small = pos[pos.rel_area <= pos.rel_area.quantile(0.25)]
cases = pd.concat(
    [
        pos[pos.score >= 3].sample(2, random_state=7),
        small[small.score >= 0].sample(1, random_state=7),
        small[small.score < 0].sample(1, random_state=7),
    ]
)

cams = {}
for k, grp in cases.groupby("fold"):
    model = build_model("convnext_tiny").to(dev)
    model.load_state_dict(torch.load(f"results/model_fracatlas_fold{int(k)}.pt", map_location=dev))
    for cid in grp.case_id:
        cams[cid], _ = gradcam(model, to_batch(arr, [row_of[cid]], False, 512))
    del model

fig = plt.figure(figsize=(12, 3.6))
fig.text(0.03, 0.80, "Leak-Controlled Evaluation Pipeline\nfor Fracture Detection", fontsize=22, fontweight="bold", va="top", linespacing=1.15)
fig.text(
    0.03, 0.50, "Internal accuracy, external transfer and shortcut audits\non public radiographs", fontsize=11, color=GREY, va="top"
)
for x, num, label, c in [
    (0.03, "0.92", "AUC, FracAtlas\n(internal)", BLUE),
    (0.17, "0.83", "AUC, GRAZPEDWRI-DX\n(external)", ORANGE),
    (0.31, "0.70", "AUC, initial exams\n(external)", RED),
]:
    fig.text(x, 0.27, num, fontsize=26, fontweight="bold", color=c, va="center")
    fig.text(x, 0.12, label, fontsize=8.5, color=GREY, va="center")

left, width, gap = 0.50, 0.115, 0.01
for j, (_, r) in enumerate(cases.iterrows()):
    for row, with_cam in enumerate([False, True]):
        height = width * 12 / 3.6
        ax = fig.add_axes([left + j * (width + gap), 0.48 - row * (height + 0.04) + 0.04, width, height])
        ax.imshow(arr[row_of[r.case_id]], cmap="gray", vmin=0, vmax=255)
        if with_cam:
            ax.imshow(cams[r.case_id], cmap="jet", alpha=0.4, vmin=0, vmax=1)
        else:
            for bx, by, bw, bh in boxes[r.case_id]:
                ax.add_patch(Rectangle((bx, by), bw, bh, fill=False, ec="lime", lw=1.5))
            ax.set_title(f"{r.bodypart}, {r.score:+.1f}", fontsize=9, fontweight="normal", loc="center")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
fig.savefig(OUT / "banner.png", dpi=200)
print("saved", OUT / "banner.png")
