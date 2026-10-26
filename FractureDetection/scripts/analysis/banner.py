"""Header image for the README: title, pipeline spine, internal vs external AUC and four Grad-CAM cases.
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
from fracai.plotstyle import BLUE, GREY, ORANGE, RED, use
from fracai.train import build_model, dev, load_cache, to_batch

use()
OUT = Path("results/figures/slides")
BIG, MID, SMALL = 24, 12, 9  # the only three text sizes on the banner
INK = "#222222"
W, H = 12, 4.0
SEED = 3

# AUC with 95% group-bootstrap CI, from results/REPORT.md
AUCS = [
    ("FracAtlas, internal (out of fold)", 0.923, 0.910, 0.935, BLUE),
    ("GRAZPEDWRI-DX, all images", 0.825, 0.816, 0.833, ORANGE),
    ("GRAZPEDWRI-DX, no cast or metal", 0.744, 0.735, 0.753, ORANGE),
    ("GRAZPEDWRI-DX, initial exams only", 0.697, 0.687, 0.707, RED),
]
STEPS = [
    ("FracAtlas, 4,083 radiographs", BLUE),
    ("Leak-controlled split, 5 group folds", BLUE),
    ("ConvNeXt-Tiny, out-of-fold scores", BLUE),
    ("Threshold fixed at 90% specificity", BLUE),
    ("GRAZPEDWRI-DX external test, 20,327 images", ORANGE),
]

# out-of-fold scores shifted so that 0 is each fold's operating point
oof, val = pd.read_csv("results/oof_fracatlas.csv"), pd.read_csv("results/val_fracatlas.csv")
thr = {k: thr_at_spec(v.y, v.score, 0.90) for k, v in val.groupby("test_fold")}
oof["score"] -= oof.fold.map(thr)
meta = pd.read_csv("results/fracatlas_meta.csv")
row_of = {c: i for i, c in enumerate(meta.case_id)}
arr = load_cache("fracatlas", 512)
boxes = load_boxes()
long_side = pd.read_csv("results/fracatlas_boxes.csv").set_index("case_id").long_side_512
pos = oof[(oof.y == 1) & oof.case_id.isin(boxes) & (oof.bodypart == "hand")]
pos = pos[(pos.case_id.map(long_side) >= 30) & (pos.case_id.map(boxes).str.len() == 1)]  # one box, big enough to see
cases = pd.concat([pos[pos.score >= 0].sample(2, random_state=SEED), pos[pos.score < 0].sample(2, random_state=SEED)])

cams = {}
for k, grp in cases.groupby("fold"):
    model = build_model("convnext_tiny").to(dev)
    model.load_state_dict(torch.load(f"results/model_fracatlas_fold{int(k)}.pt", map_location=dev))
    for cid in grp.case_id:
        cams[cid], _ = gradcam(model, to_batch(arr, [row_of[cid]], False, 512))
    del model


def crop_window(case_id):
    """Square window around the annotated boxes, kept inside the 512 px image."""
    b = np.array(boxes[case_id])
    x1, y1, x2, y2 = b[:, 0].min(), b[:, 1].min(), (b[:, 0] + b[:, 2]).max(), (b[:, 1] + b[:, 3]).max()
    size = int(np.clip(4 * max(x2 - x1, y2 - y1), 160, 320))
    x0 = int(np.clip((x1 + x2 - size) / 2, 0, 512 - size))
    y0 = int(np.clip((y1 + y2 - size) / 2, 0, 512 - size))
    return x0, y0, size


def axes_in(x, top, w, h):
    """Axes placed in inches, measured from the top-left corner."""
    return fig.add_axes([x / W, (H - top - h) / H, w / W, h / H])


def text_in(x, top, s, **kw):
    kw = {"fontsize": SMALL, "color": INK, "va": "top", **kw}
    fig.text(x / W, (H - top) / H, s, **kw)


fig = plt.figure(figsize=(W, H))
margin, head, body, foot = 0.4, 1.22, 1.6, 3.8  # grid lines in inches from the top
cols = [0.4, 4.35, 9.54]  # left edge of the three panels

text_in(margin, 0.3, "Leak-Controlled Evaluation Pipeline for Fracture Detection", fontsize=BIG, fontweight="bold")
text_in(margin, 0.82, "Internal accuracy, external transfer and shortcut audits on public radiographs", fontsize=MID, color=GREY)
for x, title in zip(cols, ["Pipeline", "AUC with 95% CI", "Grad-CAM, FracAtlas hand"]):
    text_in(x, head, title, fontsize=MID, fontweight="bold")

# pipeline spine: one dot per step, the external test in orange
spine = axes_in(cols[0], body, 3.5, 1.85)
spine.set_xlim(0, 1)
spine.set_ylim(len(STEPS) - 0.6, -0.4)
spine.axis("off")
spine.plot([0.03, 0.03], [0, len(STEPS) - 1], color=GREY, lw=1, zorder=1)
for i, (step, colour) in enumerate(STEPS):
    spine.scatter(0.03, i, s=36, color=colour, zorder=2)
    spine.text(0.09, i, step, fontsize=SMALL, color=INK, va="center")

# AUC panel: label above each bar, CI as a thin whisker
ax = axes_in(cols[1], body, 4.3, 1.85)
for i, (label, auc, lo, hi, colour) in enumerate(AUCS):
    ax.barh(i, auc - 0.5, left=0.5, height=0.42, color=colour)
    ax.plot([lo, hi], [i, i], color=INK, lw=0.8)
    ax.text(0.505, i - 0.27, label, fontsize=SMALL, color=INK, va="bottom")
    ax.text(hi + 0.008, i, f"{auc:.3f}", fontsize=SMALL, color=INK, va="center")
ax.set_xlim(0.5, 1.0)
ax.set_ylim(len(AUCS) - 0.6, -0.75)
ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.tick_params(axis="x", labelsize=SMALL, colors=GREY, length=3)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_color(GREY)

# Grad-CAM cases: detected in the left column, missed in the right
size, gap = 0.98, 0.1
for col, (label, colour) in enumerate([("Detected", BLUE), ("Missed", RED)]):
    text_in(cols[2] + col * (size + gap), body - 0.02, label, fontweight="bold", color=colour, va="bottom")
for j, (_, r) in enumerate(cases.iterrows()):
    ax = axes_in(cols[2] + (j // 2) * (size + gap), body + (j % 2) * (size + gap), size, size)
    x0, y0, s = crop_window(r.case_id)
    ax.imshow(arr[row_of[r.case_id]][y0 : y0 + s, x0 : x0 + s], cmap="gray", vmin=0, vmax=255)
    ax.imshow(cams[r.case_id][y0 : y0 + s, x0 : x0 + s], cmap="jet", alpha=0.3, vmin=0, vmax=1)
    for qx, qy, qw, qh in boxes[r.case_id]:
        ax.add_patch(Rectangle((qx - x0, qy - y0), qw, qh, fill=False, ec="white", lw=1, ls="--"))
    ax.set_xlim(0, s)
    ax.set_ylim(s, 0)
    ax.axis("off")

text_in(margin, foot, "Research evaluation on public data. Not a clinical device.", color=GREY)
text_in(cols[1], foot, "Models and thresholds fixed on FracAtlas, not refitted", color=GREY)
text_in(cols[2], foot, "Dashed box: annotated fracture", color=GREY)

fig.savefig(OUT / "banner.png", dpi=200)
print("saved", OUT / "banner.png")
print(cases[["case_id", "fold", "score"]])
