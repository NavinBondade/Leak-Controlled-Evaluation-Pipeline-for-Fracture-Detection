"""Blur and rescale the out-of-fold images and re-score them at the fixed operating point."""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from fracai.metrics import group_bootstrap, safe_auc, sens_spec, thr_at_spec
from fracai.train import build_model, dev, load_cache

name = "fracatlas"
oof, val = pd.read_csv(f"results/oof_{name}.csv"), pd.read_csv(f"results/val_{name}.csv")
thr = {k: thr_at_spec(v.y, v.score, 0.90) for k, v in val.groupby("test_fold")}
meta = pd.read_csv(f"results/{name}_meta.csv")
row_of = {c: i for i, c in enumerate(meta.case_id)}
boxes = pd.read_csv("results/fracatlas_boxes.csv").set_index("case_id")
arr = load_cache(name, 512)
CONDS = [
    ("native 512", 512, 512),
    ("reduce to 384, back to 512", 384, 512),
    ("reduce to 256, back to 512", 256, 512),
    ("reduce to 192, back to 512", 192, 512),
    ("reduce to 128, back to 512", 128, 512),
    ("input 384", 512, 384),
    ("input 640 (upscaled)", 512, 640),
    ("input 768 (upscaled)", 512, 768),
]


@torch.no_grad()
def run(model, idx, mid, out):
    res = []
    for i in range(0, len(idx), 8):
        x = torch.from_numpy(np.stack([arr[j] for j in idx[i : i + 8]])).to(dev).float().div_(255).unsqueeze(1)
        if mid != 512:
            x = F.interpolate(x, size=mid, mode="bilinear", antialias=True, align_corners=False)
        if out != x.shape[-1]:
            x = F.interpolate(x, size=out, mode="bilinear", align_corners=False)
        with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
            res.append(model((x - 0.45) / 0.25).float().squeeze(1).cpu())
    return torch.cat(res).numpy()


scores = {c[0]: np.zeros(len(oof)) for c in CONDS}
for k in sorted(oof.fold.unique()):
    sel = np.where(oof.fold.values == k)[0]
    idx = [row_of[c] for c in oof.case_id.values[sel]]
    model = build_model("convnext_tiny").to(dev)
    model.load_state_dict(torch.load(f"results/model_{name}_fold{int(k)}.pt", map_location=dev))
    model.eval()
    for lab, mid, out in CONDS:
        scores[lab][sel] = run(model, idx, mid, out) - thr[int(k)]
    del model
    torch.cuda.empty_cache()
    print(f"fold {int(k)} done", flush=True)

oof = oof.merge(meta[["case_id", "group"]].drop_duplicates("case_id"), on="case_id", how="left", suffixes=("", "_m"))
oof["rel_area"] = oof.case_id.map(boxes.rel_area)
q1 = oof[oof.y == 1].rel_area.quantile(0.25)
rows = []
for lab, _, _ in CONDS:
    d = oof.assign(score=scores[lab])
    small = d[(d.y == 1) & (d.rel_area <= q1)]
    a = group_bootstrap(d, lambda x: safe_auc(x.y, x.score), "group", 300)
    se = group_bootstrap(d, lambda x: sens_spec(x.y, x.score, 0.0)[0], "group", 300)
    sp = group_bootstrap(d, lambda x: sens_spec(x.y, x.score, 0.0)[1], "group", 300)
    ss = group_bootstrap(small, lambda x: (x.score >= 0).mean(), "group", 300)
    rows.append(
        {
            "condition": lab,
            "auc": f"{a[0]:.3f} ({a[1]:.3f}-{a[2]:.3f})",
            "sens": f"{se[0]:.3f}",
            "spec": f"{sp[0]:.3f}",
            "sens_smallest_quartile": f"{ss[0]:.3f} ({ss[1]:.3f}-{ss[2]:.3f})",
        }
    )
t = pd.DataFrame(rows)
t.to_csv("results/resolution.csv", index=False)
print(t.to_string(index=False))
