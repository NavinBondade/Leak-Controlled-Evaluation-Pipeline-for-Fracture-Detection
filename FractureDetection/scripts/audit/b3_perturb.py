"""Job B3: ensemble-score sensitivity to simple image perturbations, casted vs non-casted GRAZ images.

Sample (seed 0, 1000 images): casted all 237 negatives + 263 positives; non-casted 125 negatives and 125 positives
from each of follow-up and initial exams. Ensemble score = mean of five validation-shifted fold logits (flag if >= 0),
exactly as in scripts/audit/score_external.py. Perturbations act on the 512 px uint8 cache image:
  blur_s2 / blur_s4     Gaussian blur, sigma 2 / 4 px
  border_mask10         outer 10% band of the image content set to 0 (same value as the letterbox padding)
  border_crop10         outer 10% of the content cropped away and the rest rescaled back to the letterbox (zoom)
  bright+0.1 / -0.1     intensity shift of +/-10% of full range
  contrast0.8 / 1.25    contrast scaled about mid-grey
  hi_mask               pixels at or above the 95th intensity percentile of the content (smoothed, dilated 2 px)
                        filled with the content median: removes the brightest material (cast, metal, labels, dense bone)
  hi_mask_ctrl          the same mask shape shifted to a random place (control for 'any masking of that area')
Means of score change (logit units) with 1000 patient-bootstrap CIs.
"""

import time

import numpy as np
import pandas as pd
import torch
from b_common import OUT, fmt, graz_arr, graz_table, load_fold, patient_boot, stratified_sample, thresholds
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter

from fracai.train import dev

t0 = time.time()
d, arr, thr = graz_table(), graz_arr(), thresholds()
c, ie, y = d.cast == 1, d.initial_exam == 1, d.y == 1
S = stratified_sample(
    d,
    [(c & ~y, 237), (c & y, 263), (~c & ~ie & ~y, 125), (~c & ie & ~y, 125), (~c & ~ie & y, 125), (~c & ie & y, 125)],
).reset_index(drop=True)
print(S.groupby(["cast", "initial_exam", "y"]).size())
base = np.stack([np.asarray(arr[r]) for r in S.row])
rng = np.random.default_rng(0)


def box(img):
    r, c_ = np.where(img.any(1))[0], np.where(img.any(0))[0]
    return r[0], r[-1] + 1, c_[0], c_[-1] + 1


def border_mask(img):
    r0, r1, c0, c1 = box(img)
    br, bc = round(0.1 * (r1 - r0)), round(0.1 * (c1 - c0))
    out = np.zeros_like(img)
    out[r0 + br : r1 - br, c0 + bc : c1 - bc] = img[r0 + br : r1 - br, c0 + bc : c1 - bc]
    return out


def border_crop(img):
    r0, r1, c0, c1 = box(img)
    br, bc = round(0.1 * (r1 - r0)), round(0.1 * (c1 - c0))
    a = img[r0 + br : r1 - br, c0 + bc : c1 - bc]
    h, w = a.shape
    s = 512 / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    small = np.asarray(Image.fromarray(a).resize((nw, nh), Image.BILINEAR))
    out = np.zeros_like(img)
    y0, x0 = (512 - nh) // 2, (512 - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = small
    return out


def lin(img, a, b):
    """Apply to content only (padding stays 0): v' = (v-0.5)*a + 0.5 + b on [0,1] scale."""
    v = img.astype(np.float32) / 255
    o = np.clip((v - 0.5) * a + 0.5 + b, 0, 1)
    o = np.where(img > 0, o, 0)
    return (o * 255).round().astype(np.uint8)


def hi_masks(img):
    r0, r1, c0, c1 = box(img)
    content = np.zeros(img.shape, bool)
    content[r0:r1, c0:c1] = True
    sm = gaussian_filter(img.astype(np.float32), 1)
    m = binary_dilation((sm >= np.percentile(sm[content], 95)) & content, iterations=2)
    med = np.median(img[content])
    a = img.copy()
    a[m] = med
    dy, dx = rng.integers(64, 448, 2)
    mc = np.roll(m, (dy, dx), (0, 1)) & content & ~m
    b = img.copy()
    b[mc] = med
    return a, b, m[content].mean(), mc[content].mean()


conds = {
    "none": lambda im: im,
    "blur_s2": lambda im: gaussian_filter(im.astype(np.float32), 2).round().astype(np.uint8),
    "blur_s4": lambda im: gaussian_filter(im.astype(np.float32), 4).round().astype(np.uint8),
    "border_mask10": border_mask,
    "border_crop10": border_crop,
    "bright+0.1": lambda im: lin(im, 1, 0.1),
    "bright-0.1": lambda im: lin(im, 1, -0.1),
    "contrast0.8": lambda im: lin(im, 0.8, 0),
    "contrast1.25": lambda im: lin(im, 1.25, 0),
}
hm = [hi_masks(im) for im in base]
S["hi_mask_area"] = [h[2] for h in hm]
S["hi_ctrl_area"] = [h[3] for h in hm]

models = [load_fold(k) for k in range(5)]


@torch.no_grad()
def ens(imgs, nets=models, bs=32):
    out = []
    for i in range(0, len(imgs), bs):
        x = torch.from_numpy(imgs[i : i + bs]).to(dev).float().div_(255).unsqueeze(1)
        x = (x - 0.45) / 0.25
        with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
            out.append(np.mean([m(x).float().squeeze(1).cpu().numpy() - thr[k] for k, m in enumerate(nets)], 0))
    return np.concatenate(out)


scores = {}
for name, fn in conds.items():
    scores[name] = ens(np.stack([fn(im) for im in base]))
    print(f"{name} done ({time.time() - t0:.0f}s)", flush=True)
scores["hi_mask"] = ens(np.stack([h[0] for h in hm]))
scores["hi_mask_ctrl"] = ens(np.stack([h[1] for h in hm]))
del models, hm
torch.cuda.empty_cache()
print("max |recomputed baseline - saved|:", float(np.abs(scores["none"] - S.score.values).max()))
for k, v in scores.items():
    S[f"s_{k}"] = v
S.to_csv(OUT / "b3_perturb_scores.csv", index=False)

rows = []
for name in scores:
    if name == "none":
        continue
    for lab, sub in [
        ("casted, all", S[S.cast == 1]),
        ("casted, y=0", S[(S.cast == 1) & (S.y == 0)]),
        ("casted, y=1", S[(S.cast == 1) & (S.y == 1)]),
        ("non-casted, all", S[S.cast == 0]),
        ("non-casted, y=0", S[(S.cast == 0) & (S.y == 0)]),
        ("non-casted, y=1", S[(S.cast == 0) & (S.y == 1)]),
    ]:
        dl = patient_boot(sub, lambda x: (x[f"s_{name}"] - x.s_none).mean(), n=1000)
        f0 = patient_boot(sub, lambda x: (x.s_none >= 0).mean(), n=1000)
        f1 = patient_boot(sub, lambda x: (x[f"s_{name}"] >= 0).mean(), n=1000)
        df_ = patient_boot(sub, lambda x: (x[f"s_{name}"] >= 0).mean() - (x.s_none >= 0).mean(), n=1000)
        rows.append(
            {
                "perturbation": name,
                "subset": lab,
                "n": len(sub),
                "mean_score_change": fmt(dl, 2),
                "flag_rate_before": f"{f0[0]:.3f}",
                "flag_rate_after": fmt(f1),
                "flag_rate_change": fmt(df_),
            }
        )
    # cast vs non-cast difference in score change
    dd = patient_boot(
        S,
        lambda x: (
            (x[x.cast == 1][f"s_{name}"] - x[x.cast == 1].s_none).mean()
            - (x[x.cast == 0][f"s_{name}"] - x[x.cast == 0].s_none).mean()
        ),
        n=1000,
    )
    rows.append(
        {"perturbation": name, "subset": "casted minus non-casted", "n": len(S), "mean_score_change": fmt(dd, 2)}
    )
R = pd.DataFrame(rows)
R.to_csv(OUT / "b3_perturb_summary.csv", index=False)
pd.set_option("display.width", 250)
print(R.to_string(index=False))
print(
    "hi-mask area share: casted",
    S[S.cast == 1].hi_mask_area.mean().round(3),
    "non-casted",
    S[S.cast == 0].hi_mask_area.mean().round(3),
)
print(f"total {time.time() - t0:.0f}s")
