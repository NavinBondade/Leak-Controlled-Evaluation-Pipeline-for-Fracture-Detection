"""Grad-CAM gallery and pointing-game check on the annotated positives."""

import json
import os

import matplotlib
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from fracai.data import FRAC
from fracai.gradcam import gradcam, pointing_hit
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
boxes, content = {}, {}
for a in coco["annotations"]:
    im = img[a["image_id"]]
    s = 512 / max(im["width"], im["height"])
    ox, oy = (512 - im["width"] * s) / 2, (512 - im["height"] * s) / 2
    x, y, w, h = a["bbox"]
    cid = im["file_name"].replace(".jpg", "")
    boxes.setdefault(cid, []).append((x * s + ox, y * s + oy, w * s, h * s))
    content[cid] = (im["width"] * s) * (im["height"] * s)

missed = oof[(oof.y == 1) & (oof.score < 0)].sort_values("score").head(8)
alarms = oof[(oof.y == 0) & (oof.score >= 0)].sort_values("score", ascending=False).head(4)
gallery = pd.concat([missed, alarms])
need = pd.concat([oof[oof.y == 1], gallery]).drop_duplicates("case_id")

cams, rows = {}, []
for k, grp in need.groupby("fold"):
    model = build_model("convnext_tiny").to(dev)
    model.load_state_dict(torch.load(f"results/model_{name}_fold{int(k)}.pt", map_location=dev))
    for _, r in grp.iterrows():
        cam, _ = gradcam(model, to_batch(arr, [row_of[r.case_id]], False, 512))
        if r.case_id in set(gallery.case_id):
            cams[r.case_id] = cam
        if r.y == 1:
            bx = boxes[r.case_id]
            chance = min(1.0, sum((w + 32) * (h + 32) for _, _, w, h in bx) / content[r.case_id])
            rows.append(
                {
                    "case_id": r.case_id,
                    "group": meta.group[row_of[r.case_id]],
                    "detected": int(r.score >= 0),
                    "hit": int(pointing_hit(cam, bx)),
                    "chance": chance,
                    "bodypart": r.bodypart,
                }
            )
    del model
    torch.cuda.empty_cache()
    print(f"fold {int(k)} done", flush=True)

pg = pd.DataFrame(rows)
pg.to_csv("results/gradcam_pointing.csv", index=False)
print("\nAttention peak inside the annotated fracture box (box grown by 16 px):")
for lab, sub in [("all positives", pg), ("detected (TP)", pg[pg.detected == 1]), ("missed (FN)", pg[pg.detected == 0])]:
    p, lo, hi = group_bootstrap(sub, lambda d: d.hit.mean(), "group", 1000)
    print(
        f"  {lab:14s} n={len(sub):4d}  hit rate {p:.3f} ({lo:.3f}-{hi:.3f})   chance if attention were uniform: {sub.chance.mean():.3f}"
    )

fig, axes = plt.subplots(3, 4, figsize=(16, 12.5))
for ax in axes.flat:
    ax.axis("off")
for ax, (_, r) in zip(axes.flat, gallery.iterrows()):
    ax.imshow(arr[row_of[r.case_id]], cmap="gray", vmin=0, vmax=255)
    ax.imshow(cams[r.case_id], cmap="jet", alpha=0.35, vmin=0, vmax=1)
    for x, y, w, h in boxes.get(r.case_id, []):
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="lime", lw=2))
    kind = "MISSED fracture" if r.y == 1 else "FALSE ALARM"
    ax.set_title(
        f"{kind}\n{r.case_id} | {r.bodypart} | score {r.score:+.1f}",
        fontsize=9,
        color="crimson" if r.y == 1 else "darkorange",
    )
fig.suptitle(
    "Grad-CAM (red/yellow = regions pushing the 'fracture' score up). Green box = annotated fracture. "
    "For clinician review; saliency maps are not proof of what the model relies on.",
    fontsize=10,
)
plt.tight_layout()
os.makedirs("results/figures", exist_ok=True)
fig.savefig("results/figures/error_gallery_cam.png", dpi=80)
print("saved results/figures/error_gallery_cam.png")
