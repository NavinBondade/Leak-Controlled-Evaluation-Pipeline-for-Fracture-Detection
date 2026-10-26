"""A2. Age and sex subgroups on GRAZPEDWRI-DX external scores, fixed threshold score >= 0.

Age bands (years) follow skeletal/developmental stages relevant to paediatric wrist fractures:
  <6 pre-school, 6-<10 school age, 10-<13 early adolescence / pre-pubertal growth spurt,
  13-<16 peak-incidence adolescence (open but narrowing physes), >=16 near skeletal maturity.
Each run twice: all images, and no-cast initial exams only (removes the follow-up / cast confound).
Patient-level bootstrap (1000 draws) via fracai.audit.subgroup_table. Subgroups with <30 pos or neg flagged.
Writes results/audit2/A2_age_sex.csv.
"""

from pathlib import Path

import pandas as pd

from fracai.audit import subgroup_table
from fracai.data import GRAZ

OUT = Path("results/audit2")
OUT.mkdir(parents=True, exist_ok=True)

raw = pd.read_csv(GRAZ / "dataset.csv", usecols=["filestem", "initial_exam"])
d = pd.read_csv("results/ext_graz_scores.csv").merge(
    raw.rename(columns={"filestem": "case_id"}), on="case_id", how="left"
)
d["initial_exam"] = d.initial_exam.fillna(0).astype(int)
d["age_band"] = pd.cut(d.age, [0, 6, 10, 13, 16, 100], right=False, labels=["<6", "6-<10", "10-<13", "13-<16", ">=16"])
print("missing age:", d.age.isna().sum(), "| sex counts:", d.sex.value_counts(dropna=False).to_dict())

out = []
for lab, sub in [("all images", d), ("no-cast initial exams", d[(d.initial_exam == 1) & (d.cast == 0)])]:
    t = subgroup_table(sub, ["age_band", "sex"], 0.0, n_boot=1000)
    t.insert(0, "set", lab)
    t["flag_rate"] = [sub.score.ge(0).mean()] + [
        sub[sub[c] == lv].score.ge(0).mean() for c, lv in zip(t.subgroup[1:], t.level[1:])
    ]
    t["pct_followup"] = [1 - sub.initial_exam.mean()] + [
        1 - sub[sub[c] == lv].initial_exam.mean() for c, lv in zip(t.subgroup[1:], t.level[1:])
    ]
    t["pct_cast"] = [sub.cast.mean()] + [sub[sub[c] == lv].cast.mean() for c, lv in zip(t.subgroup[1:], t.level[1:])]
    out.append(t)
res = pd.concat(out)
res.to_csv(OUT / "A2_age_sex.csv", index=False)
show = res.copy()
for c in show.select_dtypes("float").columns:
    show[c] = show[c].round(3)
pd.set_option("display.width", 250)
print(show.drop(columns=["sens_lo", "sens_hi", "spec_lo", "spec_hi"]).to_string(index=False))
print("\nSens/spec CIs:")
print(
    show[["set", "subgroup", "level", "sens", "sens_lo", "sens_hi", "spec", "spec_lo", "spec_hi"]].to_string(
        index=False
    )
)
