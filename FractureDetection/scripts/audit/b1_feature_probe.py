"""Job B1: linear probes on the fold models' pooled features (768-d, after head norm, before the final linear layer).

Sample (~4000 GRAZ images, seed 0, stratified by cast x initial_exam x label; all Agfa images added).
Probe = StandardScaler + L2 logistic regression, fitted with 5-fold GroupKFold by patient (out-of-fold probabilities),
one probe per backbone; the five backbones' out-of-fold probabilities are averaged. AUC CIs: 200 patient bootstrap draws.
Baselines: the model's own ensemble score, and a probe on raw pixels (32x32 thumbnail + 32-bin histogram).
"""

import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from b_common import OUT, fmt, graz_arr, graz_table, load_fold, patient_boot, stratified_sample, thresholds
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fracai.train import dev, to_batch

t0 = time.time()
d, arr, thr = graz_table(), graz_arr(), thresholds()
c, ie, y = d.cast == 1, d.initial_exam == 1, d.y == 1
S = stratified_sample(
    d,
    [
        (c & ~y, 10_000),  # all 237 casted negatives
        (c & y, 900),
        (~c & ~ie & ~y, 10_000),  # all 244 non-cast follow-up negatives
        (~c & ~ie & y, 700),
        (ie & ~y, 1000),
        (ie & y, 800),
        (d.device == "Agfa", 10_000),  # all Agfa
    ],
)
S = S.sort_values("row").reset_index(drop=True)
S["agfa"] = (S.device == "Agfa").astype(int)
print("sample:", len(S), "images,", S.patient_id.nunique(), "patients")
comp = S.groupby(["cast", "initial_exam", "y"]).size().rename("n").reset_index()
print(comp.to_string(index=False))
rows = S.row.values

# ---- feature extraction (inference only)
feats, logits = [], []
fcache = OUT / "b1_features.npz"
if fcache.exists() and np.array_equal(np.load(fcache)["rows"], rows):
    z = np.load(fcache)
    feats, logits = list(z["feats"]), list(z["logits"])
else:
    for k in range(5):
        m = load_fold(k)
        F_, L_ = [], []
        with torch.no_grad():
            for i in range(0, len(rows), 32):
                x = to_batch(arr, rows[i : i + 32], False, 512)
                with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                    f = m.forward_head(m.forward_features(x), pre_logits=True).float()
                F_.append(f.cpu().numpy())
                L_.append(m.head.fc(f).squeeze(1).cpu().numpy())
        feats.append(np.concatenate(F_))
        logits.append(np.concatenate(L_))
        del m
        torch.cuda.empty_cache()
        print(f"model {k} features done ({time.time() - t0:.0f}s)", flush=True)
    np.savez_compressed(fcache, rows=rows, feats=np.stack(feats), logits=np.stack(logits))
# sanity: recomputed ensemble score vs saved one
ens = np.mean([logits[k] - thr[k] for k in range(5)], axis=0)
print("max |recomputed - saved score|:", float(np.abs(ens - S.score.values).max()))

# raw-pixel baseline features
thumb = np.stack([np.asarray(arr[r], np.float32) for r in rows])
pix = F.adaptive_avg_pool2d(torch.from_numpy(thumb).unsqueeze(1), 32).flatten(1).numpy() / 255
hist = np.stack([np.histogram(t[t > 0], bins=32, range=(0, 255), density=True)[0] for t in thumb])
pix = np.concatenate([pix, hist], 1)
del thumb


def oof_probe(X, yv, groups, C=0.05):
    p = np.zeros(len(yv))
    for tr, te in GroupKFold(5).split(X, yv, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
        clf.fit(X[tr], yv[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    return p


def auc_fn(col):
    return lambda x: roc_auc_score(x.t, x[col]) if 0 < x.t.sum() < len(x) else np.nan


# (target, subset mask, description)
tasks = [
    ("y", np.ones(len(S), bool), "fracture | whole sample"),
    ("y", ((S.cast == 0) & (S.initial_exam == 1)).values, "fracture | initial exams, no cast"),
    ("y", (S.cast == 1).values, "fracture | casted only"),
    ("cast", np.ones(len(S), bool), "cast | whole sample"),
    ("cast", ((S.y == 1) & (S.initial_exam == 0)).values, "cast | fractured follow-ups only"),
    ("cast", ((S.y == 0) & (S.initial_exam == 0)).values, "cast | non-fractured follow-ups only"),
    ("followup", np.ones(len(S), bool), "follow-up | whole sample"),
    ("followup", (S.cast == 0).values, "follow-up | no cast"),
    ("followup", ((S.cast == 0) & (S.y == 1)).values, "follow-up | no cast, fractured"),
    ("agfa", np.ones(len(S), bool), "device Agfa | whole sample"),
]
res = []
for tgt, msk, lab in tasks:
    sub = S[msk].copy()
    sub["t"] = sub[tgt].values
    g = sub.patient_id.values
    pk = [oof_probe(feats[k][msk], sub.t.values, g) for k in range(5)]
    sub["probe"] = np.mean(pk, 0)
    sub["pix"] = oof_probe(pix[msk], sub.t.values, g)
    sub["own"] = sub.score
    per = [roc_auc_score(sub.t, p) for p in pk]
    r = {
        "task": lab,
        "n_img": len(sub),
        "n_pos": int(sub.t.sum()),
        "n_patients": sub.patient_id.nunique(),
        "probe_auc": fmt(patient_boot(sub, auc_fn("probe"))),
        "probe_auc_per_backbone": f"{min(per):.3f}-{max(per):.3f}",
        "pixel_probe_auc": fmt(patient_boot(sub, auc_fn("pix"))),
        "own_score_auc": fmt(patient_boot(sub, auc_fn("own"))),
    }
    res.append(r)
    print(r, f"({time.time() - t0:.0f}s)", flush=True)
res = pd.DataFrame(res)
res.to_csv(OUT / "b1_probe_auc.csv", index=False)

# own score vs cast / follow-up on the FULL GRAZ set (not the sample), within label
own = []
for tgt, lab, msk in [
    ("y", "fracture | all", np.ones(len(d), bool)),
    ("cast", "cast | all", np.ones(len(d), bool)),
    ("cast", "cast | fractured follow-ups", ((d.y == 1) & (d.initial_exam == 0)).values),
    ("cast", "cast | non-fractured follow-ups", ((d.y == 0) & (d.initial_exam == 0)).values),
    ("followup", "follow-up | all", np.ones(len(d), bool)),
    ("followup", "follow-up | no cast, fractured", ((d.cast == 0) & (d.y == 1)).values),
    ("followup", "follow-up | no cast, non-fractured", ((d.cast == 0) & (d.y == 0)).values),
]:
    sub = d[msk].copy()
    sub["t"] = sub[tgt].values
    own.append(
        {
            "task": lab,
            "n_img": len(sub),
            "n_pos": int(sub.t.sum()),
            "own_score_auc": fmt(patient_boot(sub, auc_fn("score"))),
            "flag_rate_target1": f"{sub[sub.t == 1].flag.mean():.3f}",
            "flag_rate_target0": f"{sub[sub.t == 0].flag.mean():.3f}",
        }
    )
own = pd.DataFrame(own)
own.to_csv(OUT / "b1_own_score_full.csv", index=False)
comp.to_csv(OUT / "b1_sample_composition.csv", index=False)
print(res.to_string(index=False))
print(own.to_string(index=False))
print(f"total {time.time() - t0:.0f}s")
