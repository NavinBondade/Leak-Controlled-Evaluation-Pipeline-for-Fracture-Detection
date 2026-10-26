"""Extra GRAZ analyses on the saved external scores. Writes results/extra_*.csv."""

import pandas as pd

from fracai.data import GRAZ
from fracai.metrics import group_bootstrap, safe_auc, sens_spec

NB = 300
raw = pd.read_csv(
    GRAZ / "dataset.csv", usecols=["filestem", "study_number", "initial_exam", "ao_classification", "pixel_spacing"]
)
d = pd.read_csv("results/ext_graz_scores.csv").merge(
    raw.rename(columns={"filestem": "case_id"}), on="case_id", how="left"
)
d["initial_exam"] = d.initial_exam.fillna(0).astype(int)
d["flag"] = (d.score >= 0).astype(int)
sens = lambda x: sens_spec(x.y, x.score, 0.0)[0]
spec = lambda x: sens_spec(x.y, x.score, 0.0)[1]
fmt = lambda p: f"{p[0]:.3f} ({p[1]:.3f}-{p[2]:.3f})"

# 1. which fracture types are missed (initial exams, fractured images only; AO class collapsed to bone/type prefix)
pos = d[(d.y == 1) & (d.initial_exam == 1) & d.ao_classification.notna()].copy()
pos["ao"] = pos.ao_classification.str.split("/").str[0]
rows = []
for k, g in pos.groupby("ao"):
    if g.group.nunique() >= 30:
        rows.append({"ao_class": k, "n": len(g), "sens": fmt(group_bootstrap(g, sens, "group", NB))})
t1 = pd.DataFrame(rows).sort_values("n", ascending=False)
t1.to_csv("results/extra_ao_sensitivity.csv", index=False)
print("1. Sensitivity by AO class (initial exams, >=30 patients):\n", t1.to_string(index=False))

# (device analysis dropped: every GRAZ image is from one Siemens machine, so a device shortcut cannot be tested)

# 3. cast effect within the same label: negatives and positives separately
print("\n3. Mean score by cast status, within label (initial and follow-up pooled):")
for y in (0, 1):
    a, b = d[(d.y == y) & (d.cast == 0)].score, d[(d.y == y) & (d.cast == 1)].score
    print(
        f"   y={y}: no cast {a.mean():+.2f} (n={len(a)}) | cast {b.mean():+.2f} (n={len(b)}) | shift {b.mean() - a.mean():+.2f}"
    )

# 4. combine frontal+lateral of the same study: does it fix the weak frontal view?
d["study"] = d.patient_id.astype(str) + "_" + d.study_number.astype(str)
st = (
    d[d.initial_exam == 1]
    .groupby("study")
    .agg(
        y=("y", "max"),
        group=("group", "first"),
        n_img=("score", "size"),
        max_score=("score", "max"),
        mean_score=("score", "mean"),
    )
    .reset_index()
)
fr = d[(d.initial_exam == 1) & (d.view == "frontal")].groupby("study").score.max().rename("frontal")
la = d[(d.initial_exam == 1) & (d.view == "lateral")].groupby("study").score.max().rename("lateral")
st = st.merge(fr, on="study", how="left").merge(la, on="study", how="left")
both = st.dropna(subset=["frontal", "lateral"])
rows = []
for lab, col in [
    ("frontal only", "frontal"),
    ("lateral only", "lateral"),
    ("max of the two views", "max_score"),
    ("mean of the two views", "mean_score"),
]:
    t = both.assign(score=both[col])
    rows.append(
        {
            "read": lab,
            "studies": len(t),
            "auc": fmt(group_bootstrap(t, lambda x: safe_auc(x.y, x.score), "group", NB)),
            "sens": fmt(group_bootstrap(t, sens, "group", NB)),
            "spec": fmt(group_bootstrap(t, spec, "group", NB)),
        }
    )
t4 = pd.DataFrame(rows)
t4.to_csv("results/extra_study_level.csv", index=False)
print(f"\n4. Study-level (first visits with both views, {len(both)} studies):\n", t4.to_string(index=False))

# 5. same-patient follow-ups: does the model score a fractured wrist lower or higher once casted/treated?
print("\n5. Fractured images, initial vs follow-up: mean score and flag rate")
for lab, g in [
    ("initial", d[(d.y == 1) & (d.initial_exam == 1)]),
    ("follow-up", d[(d.y == 1) & (d.initial_exam == 0)]),
]:
    print(f"   {lab:10s} n={len(g):5d} mean score {g.score.mean():+.2f} flagged {g.flag.mean():.1%}")
