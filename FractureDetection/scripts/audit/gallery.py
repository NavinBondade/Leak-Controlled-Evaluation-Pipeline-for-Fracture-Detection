"""Gallery of the worst-missed fractures and worst false alarms for clinician review."""

import json
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from fracai.data import FRAC
from fracai.metrics import thr_at_spec
from fracai.train import load_cache

name = sys.argv[1] if len(sys.argv) > 1 else "fracatlas"
oof, val = pd.read_csv(f"results/oof_{name}.csv"), pd.read_csv(f"results/val_{name}.csv")
thr = {k: thr_at_spec(v.y, v.score, 0.90) for k, v in val.groupby("test_fold")}
oof["score"] = oof.score - oof.fold.map(thr)  # 0 = the fixed operating point
meta = pd.read_csv(f"results/{name}_meta.csv")
row_of = {c: i for i, c in enumerate(meta.case_id)}
arr = load_cache(name, 512)

coco = json.load(open(FRAC / "Annotations" / "COCO JSON" / "COCO_fracture_masks.json"))
img = {i["id"]: i for i in coco["images"]}
boxes = {}
for a in coco["annotations"]:
    im = img[a["image_id"]]
    s = 512 / max(im["width"], im["height"])
    ox, oy = (512 - im["width"] * s) / 2, (512 - im["height"] * s) / 2  # letterbox offsets
    x, y, w, h = a["bbox"]
    boxes.setdefault(im["file_name"].replace(".jpg", ""), []).append((x * s + ox, y * s + oy, w * s, h * s))

missed = oof[(oof.y == 1) & (oof.score < 0)].sort_values("score").head(8)
alarms = oof[(oof.y == 0) & (oof.score >= 0)].sort_values("score", ascending=False).head(4)
fig, axes = plt.subplots(3, 4, figsize=(16, 12.5))
for ax in axes.flat:
    ax.axis("off")
for ax, (_, r) in zip(axes.flat, pd.concat([missed, alarms]).iterrows()):
    ax.imshow(arr[row_of[r.case_id]], cmap="gray", vmin=0, vmax=255)
    for x, y, w, h in boxes.get(r.case_id, []):
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="lime", lw=2))
    kind = "MISSED fracture" if r.y == 1 else "FALSE ALARM"
    ax.set_title(
        f"{kind}\n{r.case_id} | {r.bodypart} | score {r.score:+.1f}" + (" | hardware" if r.hardware else ""),
        fontsize=9,
        color="crimson" if r.y == 1 else "darkorange",
    )
fig.suptitle(
    "Error gallery - for clinician review, no clinical interpretation implied. Green box = annotated fracture. "
    "Score is relative to the fixed 90%-specificity threshold (0).",
    fontsize=10,
)
plt.tight_layout()
import os

os.makedirs("results/figures", exist_ok=True)
fig.savefig("results/figures/error_gallery.png", dpi=80)
print(
    f"saved results/figures/error_gallery.png | {int((oof.y == 1).sum())} positives, {int(((oof.y == 1) & (oof.score < 0)).sum())} missed at the fixed threshold"
)
