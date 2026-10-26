"""Compare Grad-CAM and LayerCAM overlays on correctly detected fractures."""

import json

import matplotlib
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from fracai.data import FRAC
from fracai.gradcam import gradcam, layercam, pointing_hit
from fracai.metrics import group_bootstrap, thr_at_spec
from fracai.train import build_model, dev, load_cache, to_batch

name = "fracatlas"
oof, val = pd.read_csv(f"results/oof_{name}.csv"), pd.read_csv(f"results/val_{name}.csv")
thr = {k: thr_at_spec(v.y, v.score, 0.90) for k, v in val.groupby("test_fold")}
oof["score"] = oof.score - oof.fold.map(thr)
meta = pd.read_csv(f"results/{name}_meta.csv")
row_of = {c: i for i, c in enumerate(meta.case_id)}
arr = load_cache(name, 512)
coco = json.load(open(FRAC / "Annotations" / "COCO JSON" / "COCO_fracture_masks.json"))
img = {i["id"]: i for i in coco["images"]}
boxes = {}
for a in coco["annotations"]:
    im = img[a["image_id"]]
    s = 512 / max(im["width"], im["height"])
    ox, oy = (512 - im["width"] * s) / 2, (512 - im["height"] * s) / 2
    x, y, w, h = a["bbox"]
    boxes.setdefault(im["file_name"].replace(".jpg", ""), []).append((x * s + ox, y * s + oy, w * s, h * s))

tp = oof[(oof.y == 1) & (oof.score >= 0)]
show = tp.sample(6, random_state=7)
store, rows = {}, []
for k, grp in oof[oof.y == 1].groupby("fold"):
    model = build_model("convnext_tiny").to(dev)
    model.load_state_dict(torch.load(f"results/model_{name}_fold{int(k)}.pt", map_location=dev))
    for _, r in grp.iterrows():
        x = to_batch(arr, [row_of[r.case_id]], False, 512)
        c1, _ = gradcam(model, x)
        c2, _ = layercam(model, x, stage=2)
        if r.case_id in set(show.case_id):
            store[r.case_id] = (c1, c2)
        rows.append(
            {
                "case_id": r.case_id,
                "group": meta.group[row_of[r.case_id]],
                "detected": int(r.score >= 0),
                "hit_coarse": int(pointing_hit(c1, boxes[r.case_id])),
                "hit_fine": int(pointing_hit(c2, boxes[r.case_id])),
            }
        )
    del model
    torch.cuda.empty_cache()
pg = pd.DataFrame(rows)
pg.to_csv("results/gradcam_pointing_fine.csv", index=False)
print("Peak attention inside the annotated box (box grown by 16 px):")
for lab, sub in [("detected (TP)", pg[pg.detected == 1]), ("missed (FN)", pg[pg.detected == 0])]:
    for col in ["hit_coarse", "hit_fine"]:
        p, lo, hi = group_bootstrap(sub, lambda d: d[col].mean(), "group", 1000)
        print(f"  {lab:14s} {col:11s} n={len(sub):4d}  {p:.3f} ({lo:.3f}-{hi:.3f})")

fig, axes = plt.subplots(3, 4, figsize=(16, 12.5))
for ax in axes.flat:
    ax.axis("off")
for i, (_, r) in enumerate(show.iterrows()):
    for j, (cam, lab) in enumerate(zip(store[r.case_id], ["Grad-CAM, 32 px cells", "LayerCAM, 16 px cells"])):
        ax = axes.flat[i * 2 + j]
        ax.imshow(arr[row_of[r.case_id]], cmap="gray", vmin=0, vmax=255)
        ax.imshow(cam, cmap="jet", alpha=0.35, vmin=0, vmax=1)
        for x0, y0, w, h in boxes[r.case_id]:
            ax.add_patch(Rectangle((x0, y0), w, h, fill=False, ec="lime", lw=2))
        ax.set_title(f"DETECTED {r.case_id} | {r.bodypart} | {lab}", fontsize=9)
fig.suptitle(
    "Correctly detected fractures (6 random). Green box = annotated fracture. Finer maps on the right.", fontsize=11
)
plt.tight_layout()
fig.savefig("results/figures/detected_gallery_cam.png", dpi=80)
print("saved results/figures/detected_gallery_cam.png")
