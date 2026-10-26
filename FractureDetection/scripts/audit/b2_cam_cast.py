"""Job B2: where does Grad-CAM (fracai.gradcam, last ConvNeXt stage) sit on casted vs non-casted GRAZ wrists?

Sample (seed 0): casted 150 negatives + 150 positives (all casted images are follow-ups); non-casted 150 negatives +
150 positives, each half follow-up / half initial exam. CAM = mean of the five fold models' normalised Grad-CAMs.

Segmentation-free region proxy on the 512 px letterboxed image (letterbox padding excluded):
  border  = outer 10% band of the image content on each side (collimation edge, labels, markers, frame);
  interior pixels split by a per-image 3-class Otsu on the smoothed intensity into
  dark (air/background), mid (soft tissue, also thin cast shell) and bright (bone; also dense cast, metal).
X-ray attenuation orders air < soft tissue < bone <= metal and the cache's per-image percentile normalisation keeps
that order, so intensity class is a usable proxy. It cannot separate plaster/fibreglass from bone.
Per image: region of the CAM peak, and share of the top-10% CAM pixels in each region (and its ratio to area share).
"""

import time

import matplotlib
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from b_common import OUT, fmt, graz_arr, graz_table, load_fold, patient_boot, stratified_sample

from fracai import plotstyle
from fracai.gradcam import gradcam
from fracai.train import to_batch

t0 = time.time()
d, arr = graz_table(), graz_arr()
c, ie, y, fu = d.cast == 1, d.initial_exam == 1, d.y == 1, d.initial_exam == 0
S = stratified_sample(
    d,
    [
        (c & ~y, 150),
        (c & y, 150),
        (~c & fu & ~y, 75),
        (~c & ie & ~y, 75),
        (~c & fu & y, 75),
        (~c & ie & y, 75),
    ],
).reset_index(drop=True)
print(S.groupby(["cast", "initial_exam", "y"]).size())
REG = ["border", "dark", "mid", "bright"]


def otsu3(v):
    h, e = np.histogram(v, bins=128)
    p = h / h.sum()
    mid = (e[:-1] + e[1:]) / 2
    w, mu = np.cumsum(p), np.cumsum(p * mid)
    best, bt = -1, (0, 0)
    for i in range(1, 127):
        for j in range(i + 1, 128):
            w0, w1, w2 = w[i - 1], w[j - 1] - w[i - 1], 1 - w[j - 1]
            if min(w0, w1, w2) <= 0:
                continue
            m0, m1, m2 = mu[i - 1] / w0, (mu[j - 1] - mu[i - 1]) / w1, (mu[-1] - mu[j - 1]) / w2
            s = w0 * m0**2 + w1 * m1**2 + w2 * m2**2
            if s > best:
                best, bt = s, (e[i], e[j])
    return bt


def regions(img):
    """Label map: -1 padding, 0 border, 1 dark, 2 mid, 3 bright."""
    nz_r, nz_c = np.where(img.any(1))[0], np.where(img.any(0))[0]
    r0, r1, c0, c1 = nz_r[0], nz_r[-1] + 1, nz_c[0], nz_c[-1] + 1
    lab = -np.ones(img.shape, np.int8)
    lab[r0:r1, c0:c1] = 0
    br, bc = int(round(0.1 * (r1 - r0))), int(round(0.1 * (c1 - c0)))
    inner = np.zeros(img.shape, bool)
    inner[r0 + br : r1 - br, c0 + bc : c1 - bc] = True
    sm = gaussian_filter(img.astype(np.float32), 2)
    t1, t2 = otsu3(sm[inner])
    lab[inner] = np.where(sm[inner] < t1, 1, np.where(sm[inner] < t2, 2, 3))
    return lab


models = [load_fold(k) for k in range(5)]  # ~0.5 GB in total, fits easily
rows, keep = [], {}
for i, r in S.iterrows():
    x = to_batch(arr, [r.row], False, 512).requires_grad_(True)  # input grad only so Grad-CAM can backprop; no training
    cam = np.mean([gradcam(m, x)[0] for m in models], 0)
    img = np.asarray(arr[r.row])
    lab = regions(img)
    valid = lab >= 0
    cv = cam[valid]
    top = cam >= np.quantile(cv, 0.9)
    top &= valid
    pk = np.unravel_index(np.argmax(np.where(valid, cam, -1)), cam.shape)
    rec = {
        "case_id": r.case_id,
        "patient_id": r.patient_id,
        "cast": r.cast,
        "y": r.y,
        "initial_exam": r.initial_exam,
        "score": r.score,
        "peak_region": REG[lab[pk]],
    }
    for j, g in enumerate(REG):
        area = (lab == j).sum() / valid.sum()
        share = (top & (lab == j)).sum() / top.sum()
        rec[f"area_{g}"], rec[f"top10_{g}"], rec[f"enrich_{g}"] = area, share, share / max(area, 1e-6)
        rec[f"peak_{g}"] = int(lab[pk] == j)
    rows.append(rec)
    keep[r.case_id] = (cam[::2, ::2].astype(np.float16), pk)
    if i % 100 == 0:
        print(f"{i}/{len(S)} ({time.time() - t0:.0f}s)", flush=True)
del models
torch.cuda.empty_cache()
R = pd.DataFrame(rows)
R.to_csv(OUT / "b2_cam_regions.csv", index=False)

# summary: casted vs non-casted, within label
out = []
for yv in (0, 1, None):
    sub = R if yv is None else R[R.y == yv]
    for g in REG:
        for metric in ("peak", "top10", "area"):
            col = f"{metric}_{g}"
            a = patient_boot(sub[sub.cast == 1], lambda x: x[col].mean())
            b = patient_boot(sub[sub.cast == 0], lambda x: x[col].mean())
            dlt = patient_boot(sub, lambda x: x[x.cast == 1][col].mean() - x[x.cast == 0][col].mean())
            out.append(
                {
                    "label": "all" if yv is None else f"y={yv}",
                    "region": g,
                    "metric": metric,
                    "casted": fmt(a),
                    "non_casted": fmt(b),
                    "cast_minus_noncast": fmt(dlt),
                }
            )
out = pd.DataFrame(out)
out.to_csv(OUT / "b2_cam_summary.csv", index=False)
print(out.to_string(index=False))

# contact sheet: random (seeded, not cherry-picked) casted negatives, casted positives, non-casted reference
plotstyle.use()
rng = np.random.default_rng(1)
sets = [
    ("Casted, no fracture", R[(R.cast == 1) & (R.y == 0)]),
    ("Casted, fracture", R[(R.cast == 1) & (R.y == 1)]),
    ("No cast, no fracture (reference)", R[(R.cast == 0) & (R.y == 0)]),
]
ncol = 6
fig, axes = plt.subplots(3, ncol, figsize=(13, 7.4))
for ri, (title, sub) in enumerate(sets):
    pick = sub.iloc[rng.choice(len(sub), ncol, replace=False)]
    for ci, (_, r) in enumerate(pick.iterrows()):
        ax = axes[ri, ci]
        img = np.asarray(arr[d.row[d.case_id == r.case_id].iloc[0]])
        cam, pk = keep[r.case_id]
        nz_r, nz_c = np.where(img.any(1))[0], np.where(img.any(0))[0]
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.imshow(np.kron(cam.astype(np.float32), np.ones((2, 2))), cmap="inferno", alpha=0.4, vmin=0, vmax=1)
        ax.plot(pk[1], pk[0], marker="+", color="#00e5ff", ms=12, mew=2)
        ax.set_xlim(nz_c[0], nz_c[-1])
        ax.set_ylim(nz_r[-1], nz_r[0])
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(
            f"score {r.score:+.1f} ({'flag' if r.score >= 0 else 'no flag'})\npeak: {r.peak_region}",
            fontsize=8,
            loc="center",
        )
    axes[ri, 0].text(
        -0.08,
        0.5,
        title,
        transform=axes[ri, 0].transAxes,
        rotation=90,
        va="center",
        ha="right",
        fontsize=10,
        weight="bold",
    )
fig.suptitle(
    "Grad-CAM (mean of five fold models) on GRAZPEDWRI-DX wrists; + marks the CAM peak. "
    "Randomly chosen examples (seed 1).",
    fontsize=10,
    x=0.01,
    ha="left",
)
fig.tight_layout(rect=(0.02, 0, 1, 0.96))
fig.savefig(OUT / "b2_cam_contact_sheet.png")
print(f"saved figure; total {time.time() - t0:.0f}s")
