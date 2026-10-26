"""Occlusion test: cover the annotated fracture and measure the score drop against random patches."""

import json
import os

os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np
import pandas as pd
import torch

from fracai.data import FRAC
from fracai.metrics import group_bootstrap, thr_at_spec
from fracai.train import build_model, dev, load_cache

K, MARGIN, SEED = 5, 16, 0
name = "fracatlas"
rng = np.random.default_rng(SEED)
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
    nw, nh = im["width"] * s, im["height"] * s
    ox, oy = (512 - nw) / 2, (512 - nh) / 2
    x, y, w, h = a["bbox"]
    cid = im["file_name"].replace(".jpg", "")
    boxes.setdefault(cid, []).append((x * s + ox, y * s + oy, w * s, h * s))
    content[cid] = (ox, oy, nw, nh)


def cover(base, rects, fill):
    out = base.copy()
    for x, y, w, h in rects:
        out[max(0, int(y)) : min(512, int(np.ceil(y + h))), max(0, int(x)) : min(512, int(np.ceil(x + w)))] = fill
    return out


def random_rects(bx, c, base, tissue_thr):
    """Same-size patches at random body locations: inside the image content, away from the fracture, on tissue/bone."""
    ox, oy, nw, nh = c
    rects = []
    for _, _, w, h in bx:
        for _ in range(60):
            x, y = rng.uniform(ox, ox + max(nw - w, 1)), rng.uniform(oy, oy + max(nh - h, 1))
            clear = all(
                x + w < bx0 - MARGIN or x > bx0 + bw0 + MARGIN or y + h < by0 - MARGIN or y > by0 + bh0 + MARGIN
                for bx0, by0, bw0, bh0 in bx
            )
            if clear and base[int(y) : int(y + h) + 1, int(x) : int(x + w) + 1].mean() > tissue_thr:
                rects.append((x, y, w, h))
                break
        else:
            return None
    return rects


@torch.no_grad()
def score(model, imgs):
    x = torch.from_numpy(np.stack(imgs)).to(dev).float().div_(255).unsqueeze(1)
    with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
        return model((x - 0.45) / 0.25).float().squeeze(1).cpu().numpy()


tp = oof[(oof.y == 1) & (oof.score >= 0)]
rows = []
for k, grp in tp.groupby("fold"):
    model = build_model("convnext_tiny").to(dev)
    model.load_state_dict(torch.load(f"results/model_{name}_fold{int(k)}.pt", map_location=dev))
    model.eval()
    for _, r in grp.iterrows():
        base = np.asarray(arr[row_of[r.case_id]])
        ox, oy, nw, nh = content[r.case_id]
        body = base[int(oy) : int(oy + nh), int(ox) : int(ox + nw)]
        fill, tissue_thr = int(np.median(body)), np.percentile(body, 30)
        bx, c = boxes[r.case_id], content[r.case_id]
        ctrls = [random_rects(bx, c, base, tissue_thr) for _ in range(K)]
        ctrls = [q for q in ctrls if q is not None]
        if not ctrls:
            continue
        s = score(model, [base, cover(base, bx, fill)] + [cover(base, q, fill) for q in ctrls])
        t = thr[int(k)]
        s0, s_box, s_ctl = s[0] - t, s[1] - t, s[2:] - t
        rows.append(
            {
                "case_id": r.case_id,
                "group": meta.group[row_of[r.case_id]],
                "hardware": meta.hardware[row_of[r.case_id]],
                "bodypart": r.bodypart,
                "n_ctrl": len(ctrls),
                "s0": s0,
                "drop_box": s0 - s_box,
                "drop_ctrl": s0 - s_ctl.mean(),
                "flip_box": int(s_box < 0),
                "flip_ctrl": float((s_ctl < 0).mean()),
            }
        )
    del model
    torch.cuda.empty_cache()
    print(f"fold {int(k)} done", flush=True)

d = pd.DataFrame(rows)
d["diff"] = d.drop_box - d.drop_ctrl
d.to_csv("results/occlusion.csv", index=False)
print(f"\nimages analysed: {len(d)} of {len(tp)} detected positives (rest: no valid control placement)")
print("\nMean score drop (logit units; larger = the model relied more on the covered region):")
for lab, s in [("ALL", d), ("no hardware", d[d.hardware == 0]), ("with hardware", d[d.hardware == 1])]:
    b = group_bootstrap(s, lambda x: x.drop_box.mean(), "group", 1000)
    c = group_bootstrap(s, lambda x: x.drop_ctrl.mean(), "group", 1000)
    df_ = group_bootstrap(s, lambda x: x["diff"].mean(), "group", 1000)
    w = group_bootstrap(s, lambda x: (x["diff"] > 0).mean(), "group", 1000)
    fb = group_bootstrap(s, lambda x: x.flip_box.mean(), "group", 1000)
    fc = group_bootstrap(s, lambda x: x.flip_ctrl.mean(), "group", 1000)
    print(
        f"  {lab:14s} n={len(s):3d} | cover fracture {b[0]:.2f} ({b[1]:.2f}-{b[2]:.2f}) | cover control {c[0]:.2f} ({c[1]:.2f}-{c[2]:.2f}) "
        f"| difference {df_[0]:.2f} ({df_[1]:.2f}-{df_[2]:.2f}) | fracture-cover hurts more in {w[0]:.0%} ({w[1]:.0%}-{w[2]:.0%})"
    )
    print(
        f"  {'':14s}        flagged->not flagged: fracture-cover {fb[0]:.1%} ({fb[1]:.1%}-{fb[2]:.1%}) vs control {fc[0]:.1%} ({fc[1]:.1%}-{fc[2]:.1%})"
    )
