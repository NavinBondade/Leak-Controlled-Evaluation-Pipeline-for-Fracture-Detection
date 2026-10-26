"""A3. Device manufacturer (Agfa vs Siemens) on GRAZPEDWRI-DX external scores, threshold score >= 0.

Note: an earlier comment in scripts/audit/extra_analysis.py said every GRAZ image is Siemens; dataset.csv lists 112 Agfa.
Reports AUC/sens/spec (patient bootstrap, 1000 draws), mean score by label, composition, a case-mix-restricted
comparison, and a within-patient comparison for patients imaged on both devices.
Writes results/audit2/A3_device.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from fracai.audit import subgroup_table
from fracai.data import GRAZ
from fracai.metrics import group_bootstrap

OUT = Path("results/audit2")
OUT.mkdir(parents=True, exist_ok=True)
NB = 1000

raw = pd.read_csv(GRAZ / "dataset.csv", usecols=["filestem", "initial_exam", "study_number"])
d = pd.read_csv("results/ext_graz_scores.csv").merge(
    raw.rename(columns={"filestem": "case_id"}), on="case_id", how="left"
)
d["initial_exam"] = d.initial_exam.fillna(0).astype(int)
pd.set_option("display.width", 250)

a = d[d.device == "Agfa"]
print(
    f"Agfa: {len(a)} images, {a.patient_id.nunique()} patients, studies {a.groupby(['patient_id', 'study_number']).ngroups}"
)
print("Agfa composition (initial_exam x cast x y):\n", pd.crosstab([a.initial_exam, a.cast], a.y))
print("Agfa age:", a.age.describe().round(1).to_dict())

out = []
for lab, sub in [
    ("all images", d),
    ("no-cast initial exams", d[(d.initial_exam == 1) & (d.cast == 0)]),
    ("follow-up exams", d[d.initial_exam == 0]),
]:
    t = subgroup_table(sub, ["device"], 0.0, n_boot=NB)
    t.insert(0, "set", lab)
    for yy in (0, 1):
        t[f"mean_score_y{yy}"] = [sub[sub.y == yy].score.mean()] + [
            sub[(sub.device == lv) & (sub.y == yy)].score.mean() for lv in t.level[1:]
        ]
    out.append(t)
res = pd.concat(out)
res.to_csv(OUT / "A3_device.csv", index=False)
show = res.copy()
for c in show.select_dtypes("float").columns:
    show[c] = show[c].round(3)
print(show.to_string(index=False))

# mean score by label x device within exam-type/cast strata (case-mix check)
print("\nMean score (n) by initial_exam x cast x y x device:")
print(d.groupby(["initial_exam", "cast", "y", "device"]).score.agg(["size", "mean"]).round(2).unstack("device"))

# within-patient: patients with images on both devices, same label and view
both = d[d.patient_id.isin(a.patient_id)]
cells = both.groupby(["patient_id", "y", "view", "device"]).score.mean().unstack("device").dropna()
cells = cells.assign(diff=cells.Agfa - cells.Siemens).reset_index()
per = cells.groupby("patient_id", as_index=False)["diff"].mean().assign(group=lambda x: x.patient_id)
est = group_bootstrap(per, lambda x: x["diff"].mean(), "group", NB) if len(per) > 1 else (np.nan,) * 3
print(
    f"\nWithin-patient Agfa minus Siemens (same label & view): {len(per)} patients, {len(cells)} cells, "
    f"mean diff {est[0]:+.2f} ({est[1]:+.2f} to {est[2]:+.2f})"
)
print(cells.groupby("y")["diff"].agg(["size", "mean"]).round(2))
