"""A1. Within-patient cast test on GRAZPEDWRI-DX external scores (no training, table work only).

The pooled 'cast shifts the score' comparison is confounded: every casted negative is a follow-up exam,
and un-casted follow-ups already score high. Here the cast contrast is taken inside the same patient,
same fracture label (fracture_visible) and same projection, and (primary) inside follow-up exams only.

Exam ordering: study_number (timehash is monotone with study_number within every patient, checked below).
Unit of resampling: patient (== 'group' column). 1000 bootstrap draws.
Writes results/audit2/A1_*.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from fracai.data import GRAZ
from fracai.metrics import group_bootstrap

NB = 1000
OUT = Path("results/audit2")
OUT.mkdir(parents=True, exist_ok=True)

raw = pd.read_csv(GRAZ / "dataset.csv", usecols=["filestem", "study_number", "timehash", "initial_exam"])
d = pd.read_csv("results/ext_graz_scores.csv").merge(
    raw.rename(columns={"filestem": "case_id"}), on="case_id", how="left"
)
d["initial_exam"] = d.initial_exam.fillna(0).astype(int)
assert (d.group == d.patient_id).all()
mono = (
    d.sort_values(["patient_id", "study_number", "timehash"])
    .groupby("patient_id")
    .timehash.apply(lambda s: s.is_monotonic_increasing)
    .mean()
)
print(f"timehash monotone with study_number within patient: {mono:.1%} of patients")
fmt = lambda p: f"{p[0]:+.2f} ({p[1]:+.2f} to {p[2]:+.2f})"

print("\nCell counts (cast x initial_exam x label):")
print(pd.crosstab([d.cast, d.initial_exam], d.y))
print("\nMean score by initial_exam x cast x label:")
print(d.groupby(["initial_exam", "cast", "y"]).score.agg(["size", "mean"]).round(2))


def paired(sub, label):
    """One difference per patient x label x view cell having both cast=1 and cast=0 images;
    then one value per patient (mean over its cells). Returns per-patient frame."""
    cells = sub.groupby(["patient_id", "y", "view", "cast"]).score.mean().unstack("cast")
    cells = cells.dropna(subset=[0, 1]) if {0, 1} <= set(cells.columns) else cells.iloc[0:0]
    cells = cells.assign(diff=cells[1] - cells[0]).reset_index()
    cells["setting"] = label
    return cells


rows, allcells = [], []
settings = {
    "follow-up only (cast vs no-cast follow-ups)": d[d.initial_exam == 0],
    "any exam (no-cast side may be an initial exam)": d,
}
for lab, sub in settings.items():
    c = paired(sub, lab)
    allcells.append(c)
    strata = [("all labels/views", c)]
    strata += [(f"y={y}", g) for y, g in c.groupby("y")]
    strata += [(f"y={y}, {v}", g) for (y, v), g in c.groupby(["y", "view"])]
    for sname, g in strata:
        per_pat = g.groupby("patient_id", as_index=False)["diff"].mean().assign(group=lambda x: x.patient_id)
        est = group_bootstrap(per_pat, lambda x: x["diff"].mean(), "group", NB) if len(per_pat) >= 2 else (np.nan,) * 3
        rows.append(
            {
                "setting": lab,
                "stratum": sname,
                "patients": len(per_pat),
                "cells": len(g),
                "mean_diff_cast_minus_nocast": round(est[0], 3),
                "lo": round(est[1], 3),
                "hi": round(est[2], 3),
                "median_cell_diff": round(g["diff"].median(), 3),
                "frac_cells_cast_higher": round((g["diff"] > 0).mean(), 3),
            }
        )
t1 = pd.DataFrame(rows)
t1.to_csv(OUT / "A1_paired_within_patient.csv", index=False)
pd.concat(allcells).to_csv(OUT / "A1_paired_cells.csv", index=False)
print("\n== Paired within-patient cast effect (patient x label x view cells; patient bootstrap) ==")
print(t1.to_string(index=False))

# temporal direction among follow-up cells: is the casted image earlier (cast later removed) or later?
fu = d[d.initial_exam == 0]
order = fu.groupby(["patient_id", "y", "view", "cast"]).study_number.mean().unstack("cast").dropna()
print(
    f"\nFollow-up cells with both: casted exam earlier than un-casted in {(order[1] < order[0]).mean():.1%}, "
    f"later in {(order[1] > order[0]).mean():.1%}, same study in {(order[1] == order[0]).mean():.1%} (n={len(order)})"
)

# within-patient fixed-effects regression: score ~ cast + y + view + initial_exam, patient-demeaned
X_cols = ["cast", "y", "lateral", "oblique", "initial_exam"]
d["lateral"] = (d.view == "lateral").astype(int)
d["oblique"] = (d.view == "oblique").astype(int)


def fe_cast(x):
    dm = x[X_cols + ["score"]] - x.groupby("group")[X_cols + ["score"]].transform("mean")
    beta, *_ = np.linalg.lstsq(dm[X_cols].values, dm.score.values, rcond=None)
    return beta[0]


fe_all = group_bootstrap(d, fe_cast, "group", NB)
fe_fu = group_bootstrap(d[d.initial_exam == 0], fe_cast, "group", NB)
print(f"\nPatient fixed-effects cast coefficient (adj. label, view, initial_exam), all images: {fmt(fe_all)}")
print(f"Patient fixed-effects cast coefficient, follow-up images only: {fmt(fe_fu)}")

# stratified (unpaired) estimate inside follow-up exams, by label and by projection
srows = []
for (y, v), g in [((y, "all"), g) for y, g in fu.groupby("y")] + list(fu.groupby(["y", "view"])):
    n0, n1 = (g.cast == 0).sum(), (g.cast == 1).sum()
    if n0 < 2 or n1 < 2:
        est = (np.nan,) * 3
    else:
        est = group_bootstrap(g, lambda x: x[x.cast == 1].score.mean() - x[x.cast == 0].score.mean(), "group", NB)
    srows.append(
        {
            "y": y,
            "view": v,
            "n_nocast": n0,
            "n_cast": n1,
            "pat_nocast": g[g.cast == 0].patient_id.nunique(),
            "pat_cast": g[g.cast == 1].patient_id.nunique(),
            "mean_nocast": round(g[g.cast == 0].score.mean(), 3),
            "mean_cast": round(g[g.cast == 1].score.mean(), 3),
            "diff": round(est[0], 3),
            "lo": round(est[1], 3),
            "hi": round(est[2], 3),
            "flag_nocast": round((g[g.cast == 0].score >= 0).mean(), 3),
            "flag_cast": round((g[g.cast == 1].score >= 0).mean(), 3),
        }
    )
t2 = pd.DataFrame(srows)
t2.to_csv(OUT / "A1_stratified_followup.csv", index=False)
print("\n== Stratified, follow-up exams only (unpaired, patient bootstrap) ==")
print(t2.to_string(index=False))
