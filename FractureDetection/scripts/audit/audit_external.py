"""External validation of the FracAtlas models on GRAZPEDWRI-DX, grouped by patient."""

import pandas as pd

from fracai.audit import subgroup_table
from fracai.metrics import group_bootstrap, safe_auc, sens_spec, thr_at_spec

d = pd.read_csv("results/ext_graz_scores.csv")
from fracai.data import GRAZ

ie = pd.read_csv(GRAZ / "dataset.csv", usecols=["filestem", "initial_exam"]).rename(columns={"filestem": "case_id"})
d = d.merge(ie, on="case_id", how="left")
d["initial_exam"] = d.initial_exam.fillna(0).astype(int)  # 1 = first presentation; 0/blank = follow-up (often casted)
d["age_band"] = pd.cut(d.age, [0, 5, 10, 14, 20], right=False).astype(str)
cols = ["view", "hardware", "cast", "uncertain", "sex", "age_band"]
fmt = lambda t: t.assign(**{c: t[c].round(3) for c in t.select_dtypes("float").columns})
show = [
    "subgroup",
    "level",
    "n",
    "pos",
    "prevalence",
    "auc",
    "auc_lo",
    "auc_hi",
    "sens",
    "sens_lo",
    "sens_hi",
    "spec",
    "unreliable",
]

sub = subgroup_table(d, cols, 0.0)
sub.to_csv("results/audit_graz_external_subgroups.csv", index=False)
print("== GRAZPEDWRI-DX, FracAtlas models, pre-specified threshold ==")
print(fmt(sub[show]).to_string(index=False))

# subsets closer to real emergency use: no treatment artefacts, and/or first presentation only
print("\n== Clean subsets (pre-specified threshold) ==")
for lab, m in [
    ("no cast and no metal", (d.cast == 0) & (d.hardware == 0)),
    ("initial exam only", d.initial_exam == 1),
    ("initial exam, no cast, no metal", (d.initial_exam == 1) & (d.cast == 0) & (d.hardware == 0)),
]:
    sub_ = d[m]
    r = subgroup_table(sub_, ["view"], 0.0).rename(columns={"subgroup": "subset"})
    r.insert(0, "which", lab)
    print(
        fmt(
            r[["which", "subset", "level", "n", "pos", "prevalence", "auc", "auc_lo", "auc_hi", "sens", "spec"]]
        ).to_string(index=False)
    )

# post-hoc, clearly labelled: threshold re-fitted on GRAZ itself at 90% specificity
t = thr_at_spec(d.y, d.score, 0.90)
s = group_bootstrap(d, lambda x: sens_spec(x.y, x.score, t)[0], "group", 1000)
print(f"\nPOST-HOC (GRAZ-fitted threshold {t:.2f}, 90% spec): sensitivity {s[0]:.3f} ({s[1]:.3f}-{s[2]:.3f})")

# per-model consistency: do the five fold-models agree on GRAZ?
per = {f"model{k}": round(safe_auc(d.y, d[f"shifted_{k}"]), 3) for k in range(5)}
print("per-model AUC on GRAZ:", per, "| ensemble:", round(safe_auc(d.y, d.score), 3))

# like-with-like reference: FracAtlas hand images, controlled split
fa = pd.read_csv("results/oof_fracatlas.csv")
val = pd.read_csv("results/val_fracatlas.csv")
thr = {k: thr_at_spec(v.y, v.score, 0.90) for k, v in val.groupby("test_fold")}
fa["score"] = fa.score - fa.fold.map(thr)
h = fa[fa.bodypart == "hand"]
ref = [
    group_bootstrap(h, lambda x: safe_auc(x.y, x.score), "group", 1000),
    group_bootstrap(h, lambda x: sens_spec(x.y, x.score, 0.0)[0], "group", 1000),
    group_bootstrap(h, lambda x: sens_spec(x.y, x.score, 0.0)[1], "group", 1000),
]
print(
    "\nFracAtlas hand (internal, controlled)  AUC %.3f (%.3f-%.3f)  sens %.3f (%.3f-%.3f)  spec %.3f (%.3f-%.3f)  prevalence %.2f"
    % (*ref[0], *ref[1], *ref[2], h.y.mean())
)
