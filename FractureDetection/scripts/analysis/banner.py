"""Header image for the README: radiograph and Grad-CAM for three detected and three missed fractures.
Output: results/figures/slides/banner.png"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from fracai.data import load_boxes
from fracai.gradcam import gradcam
from fracai.metrics import thr_at_spec
from fracai.plotstyle import BLUE, GREY, RED, use
from fracai.train import build_model, dev, load_cache, to_batch

use()
OUT = Path("results/figures/slides")
SEED = 3
PARTS = ["hand", "leg", "mixed"]

# out-of-fold scores shifted so that 0 is each fold's operating point
oof, val = pd.read_csv("results/oof_fracatlas.csv"), pd.read_csv("results/val_fracatlas.csv")
thr = {k: thr_at_spec(v.y, v.score, 0.90) for k, v in val.groupby("test_fold")}
oof["score"] -= oof.fold.map(thr)
meta = pd.read_csv("results/fracatlas_meta.csv")
row_of = {c: i for i, c in enumerate(meta.case_id)}
arr = load_cache("fracatlas", 512)
boxes = load_boxes()
long_side = pd.read_csv("results/fracatlas_boxes.csv").set_index("case_id").long_side_512

# one box, big enough to see at banner size
pos = oof[(oof.y == 1) & oof.case_id.isin(boxes)]
pos = pos[(pos.case_id.map(long_side) >= 30) & (pos.case_id.map(boxes).str.len() == 1)]


def pick(hit):
    pool = pos[pos.score >= 0] if hit else pos[pos.score < 0]
    return [pool[pool.bodypart == p].sample(1, random_state=SEED).iloc[0] for p in PARTS]


cases = pick(True) + pick(False)

cams = {}
for r in cases:
    model = build_model("convnext_tiny").to(dev)
    model.load_state_dict(torch.load(f"results/model_fracatlas_fold{int(r.fold)}.pt", map_location=dev))
    cams[r.case_id], _ = gradcam(model, to_batch(arr, [row_of[r.case_id]], False, 512))
    del model


def crop_window(case_id):
    """Square window around the annotated box, kept inside the 512 px image."""
    qx, qy, qw, qh = boxes[case_id][0]
    size = int(np.clip(4 * max(qw, qh), 160, 320))
    x0 = int(np.clip(qx + qw / 2 - size / 2, 0, 512 - size))
    y0 = int(np.clip(qy + qh / 2 - size / 2, 0, 512 - size))
    return x0, y0, size


W, H = 12, 4.75
left0, gap, tile, split = 0.95, 0.1, 1.65, 0.25  # split: extra space between the two groups
top_cap, row1, row2 = 0.72, 0.88, 0.88 + 1.65 + 0.1
fig = plt.figure(figsize=(W, H))


def tile_x(j):
    return left0 + j * (tile + gap) + (split if j >= 3 else 0)


for j, r in enumerate(cases):
    x0, y0, s = crop_window(r.case_id)
    qx, qy, qw, qh = boxes[r.case_id][0]
    for row, top in enumerate([row1, row2]):
        ax = fig.add_axes([tile_x(j) / W, (H - top - tile) / H, tile / W, tile / H])
        ax.imshow(arr[row_of[r.case_id]][y0 : y0 + s, x0 : x0 + s], cmap="gray", vmin=0, vmax=255)
        if row == 1:
            ax.imshow(cams[r.case_id][y0 : y0 + s, x0 : x0 + s], cmap="jet", alpha=0.3, vmin=0, vmax=1)
        ax.add_patch(Rectangle((qx - x0, qy - y0), qw, qh, fill=False, ec="white", lw=1, ls="--"))
        ax.set_xlim(0, s)
        ax.set_ylim(s, 0)
        ax.axis("off")
    fig.text(
        (tile_x(j) + tile / 2) / W,
        (H - top_cap) / H,
        f"{r.bodypart.capitalize()}, score {r.score:+.1f}",
        fontsize=9,
        ha="center",
        va="center",
    )

# group headers with a rule, then row labels on the left
for first, label, colour in [
    (0, "Detected: score above the fixed threshold", BLUE),
    (3, "Missed: score below it", RED),
]:
    x = tile_x(first)
    y = (H - 0.22) / H
    fig.add_artist(plt.Line2D([x / W, (x + 3 * tile + 2 * gap) / W], [y - 0.045, y - 0.045], color=colour, lw=1.5))
    fig.text(x / W, y, label, fontsize=12, fontweight="bold", color=colour, va="center")
for top, label in [(row1, "Radiograph"), (row2, "Grad-CAM")]:
    fig.text(
        0.38 / W, (H - top - tile / 2) / H, label, fontsize=12, fontweight="bold", rotation=90, ha="center", va="center"
    )
fig.text(
    left0 / W,
    0.14 / H,
    "Dashed box: annotated fracture. Grad-CAM: where the model looked. Threshold: 90% specificity, fixed on validation data.",
    fontsize=9,
    color=GREY,
    va="center",
)

fig.savefig(OUT / "banner.png", dpi=200)
print("saved", OUT / "banner.png")
print(pd.DataFrame(cases)[["case_id", "bodypart", "score"]])
