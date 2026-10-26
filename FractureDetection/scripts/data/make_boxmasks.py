"""Turn the annotated boxes into 64x64 masks aligned with fracatlas_meta.csv."""
import json

import numpy as np
import pandas as pd

from fracai.data import FRAC

M, MARGIN = 64, 16
meta = pd.read_csv("results/fracatlas_meta.csv")
row_of = {c: i for i, c in enumerate(meta.case_id)}
coco = json.load(open(FRAC / "Annotations" / "COCO JSON" / "COCO_fracture_masks.json"))
img = {i["id"]: i for i in coco["images"]}
masks = np.zeros((len(meta), M, M), np.uint8)
has = np.zeros(len(meta), bool)
k = 512 / M
for a in coco["annotations"]:
    im = img[a["image_id"]]
    cid = im["file_name"].replace(".jpg", "")
    if cid not in row_of:
        continue
    s = 512 / max(im["width"], im["height"])
    ox, oy = (512 - im["width"] * s) / 2, (512 - im["height"] * s) / 2
    x, y, w, h = a["bbox"]
    x0, y0, x1, y1 = x * s + ox - MARGIN, y * s + oy - MARGIN, (x + w) * s + ox + MARGIN, (y + h) * s + oy + MARGIN
    r = row_of[cid]
    masks[r, max(0, int(np.floor(y0 / k))):min(M, int(np.ceil(y1 / k))), max(0, int(np.floor(x0 / k))):min(M, int(np.ceil(x1 / k)))] = 255
    has[r] = True
np.save("results/fracatlas_boxmask64.npy", masks)
np.save("results/fracatlas_hasbox.npy", has)
pos = meta.y.values == 1
print(f"positives {pos.sum()}, with a box {int((has & pos).sum())}, boxes on negatives {int((has & ~pos).sum())}, "
      f"mean mask fraction on positives {(masks[has & pos] > 0).mean():.3f}")
