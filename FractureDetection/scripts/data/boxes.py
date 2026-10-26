"""Per-image fracture size from the COCO boxes, written to results/fracatlas_boxes.csv."""
import json

import pandas as pd

from fracai.data import FRAC

d = json.load(open(FRAC / "Annotations" / "COCO JSON" / "COCO_fracture_masks.json"))
img = {i["id"]: i for i in d["images"]}
rows = []
for a in d["annotations"]:
    im = img[a["image_id"]]
    x, y, w, h = a["bbox"]
    rows.append({"case_id": im["file_name"].replace(".jpg", ""), "area": w * h, "rel_area": w * h / (im["width"] * im["height"]),
                 "long_side_512": max(w, h) * 512 / max(im["width"], im["height"]), "W": im["width"], "H": im["height"]})
b = pd.DataFrame(rows)
out = b.groupby("case_id").agg(n_boxes=("area", "size"), rel_area=("rel_area", "max"), long_side_512=("long_side_512", "max"),
                               W=("W", "first"), H=("H", "first")).reset_index()
out.to_csv("results/fracatlas_boxes.csv", index=False)
print(len(out), "images with boxes;", len(b), "boxes")
print(out[["rel_area", "long_side_512"]].describe(percentiles=[.1, .25, .5, .75, .9]).round(4))
print("image sizes (W x H) most common:", out.groupby(["W", "H"]).size().sort_values(ascending=False).head(5).to_dict())
