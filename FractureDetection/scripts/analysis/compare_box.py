"""Compare box-supervised and control runs against the baselines with paired group bootstraps."""

import os
import sys

import pandas as pd

from fracai.metrics import group_bootstrap, safe_auc, sens_spec, thr_at_spec

base = "fracatlas"
settings = [
    ("baseline (no boxes)", base),
    ("boxes 100%", base + "_box100"),
    ("boxes 25%", base + "_box25"),
    ("boxes 10%", base + "_box10"),
    ("CONTROL aux head, no boxes", base + "_box0"),
    ("CONTROL baseline, seed 43", base + "_s43"),
    ("fine head, boxes 100%", base + "_fine100"),
    ("CONTROL fine head, no boxes", base + "_fine0"),
    ("oversample small x3", base + "_small3"),
]
boxes = pd.read_csv("results/fracatlas_boxes.csv").set_index("case_id")


def load(name):
    oof, val = pd.read_csv(f"results/oof_{name}.csv"), pd.read_csv(f"results/val_{name}.csv")
    thr = {k: thr_at_spec(v.y, v.score, 0.90) for k, v in val.groupby("test_fold")}
    oof["s"] = oof.score - oof.fold.map(thr)
    return oof.set_index("case_id")


b = load(base)
q1 = boxes.rel_area.reindex(b.index[b.y == 1]).quantile(0.25)
rows = []
for lab, name in settings:
    if not os.path.exists(f"results/oof_{name}.csv"):
        continue
    o = load(name)
    if len(o) != len(b) or o.fold.nunique() < 5:
        print(f"{lab}: not finished ({o.fold.nunique()} folds)", file=sys.stderr)
        continue
    d = b[["y", "group", "hardware"]].copy()
    d["s0"], d["s1"] = b.s, o.s.reindex(d.index)
    d["small"] = (d.y == 1) & (boxes.rel_area.reindex(d.index) <= q1)
    f = lambda col: lambda x: safe_auc(x.y, x[col])
    auc = lambda col: group_bootstrap(d, f(col), "group", 1000)
    sm = lambda col: lambda x: (x[col][x.small] >= 0).mean()
    sp = lambda col: lambda x: sens_spec(x.y, x[col], 0.0)[1]
    se = lambda col: lambda x: sens_spec(x.y, x[col], 0.0)[0]
    nh = lambda x: x[x.hardware == 0]
    r = {
        "setting": lab,
        "auc": "%.3f (%.3f-%.3f)" % auc("s1"),
        "sens": "%.3f" % se("s1")(d),
        "spec": "%.3f" % sp("s1")(d),
        "auc_no_hardware": "%.3f" % safe_auc(nh(d).y, nh(d).s1),
        "sens_smallest_quartile": "%.3f (%.3f-%.3f)" % group_bootstrap(d, sm("s1"), "group", 1000),
    }
    if name != base:
        dsm = group_bootstrap(d, lambda x: sm("s1")(x) - sm("s0")(x), "group", 1000)
        dau = group_bootstrap(d, lambda x: f("s1")(x) - f("s0")(x), "group", 1000)
        dsp = group_bootstrap(d, lambda x: sp("s1")(x) - sp("s0")(x), "group", 1000)
        r["d_small_sens"] = "%+.3f (%+.3f to %+.3f)" % dsm
        r["d_auc"] = "%+.3f (%+.3f to %+.3f)" % dau
        r["d_spec"] = "%+.3f (%+.3f to %+.3f)" % dsp
        r["vs_seed42_only"] = (
            "CI above 0 vs seed 42 only (superseded: needs seed 43 too)"
            if (dsm[1] > 0 and dau[1] > -0.01 and dsp[1] > -0.03)
            else "CI includes 0"
        )
    rows.append(r)
t = pd.DataFrame(rows)
t.to_csv("results/box_comparison.csv", index=False)
print(t.to_string(index=False))


# paired test of each setting against its matching control
PAIRS = [
    ("boxes 100% vs aux-head control", base + "_box100", base + "_box0"),
    ("boxes 10% vs aux-head control", base + "_box10", base + "_box0"),
    ("fine boxes 100% vs fine control", base + "_fine100", base + "_fine0"),
    ("boxes 100% vs baseline seed 43", base + "_box100", base + "_s43"),
    ("boxes 10% vs baseline seed 43", base + "_box10", base + "_s43"),
    ("aux control vs baseline seed 43", base + "_box0", base + "_s43"),
    ("fine boxes 100% vs baseline seed 43", base + "_fine100", base + "_s43"),
    ("fine control vs baseline seed 43", base + "_fine0", base + "_s43"),
    ("oversample small x3 vs baseline seed 43", base + "_small3", base + "_s43"),
    ("boxes 25% vs aux-head control", base + "_box25", base + "_box0"),
    ("boxes 25% vs baseline seed 43", base + "_box25", base + "_s43"),
]
prow = []
for lab, a_, c_ in PAIRS:
    if not (os.path.exists(f"results/oof_{a_}.csv") and os.path.exists(f"results/oof_{c_}.csv")):
        continue
    A, C = load(a_), load(c_)
    if A.fold.nunique() < 5 or C.fold.nunique() < 5:
        continue
    d = b[["y", "group"]].copy()
    d["s0"], d["s1"] = C.s.reindex(d.index), A.s.reindex(d.index)
    d["small"] = (d.y == 1) & (boxes.rel_area.reindex(d.index) <= q1)
    smf = lambda col: lambda x: (x[col][x.small] >= 0).mean()
    dsm = group_bootstrap(d, lambda x: smf("s1")(x) - smf("s0")(x), "group", 1000)
    dau = group_bootstrap(d, lambda x: safe_auc(x.y, x.s1) - safe_auc(x.y, x.s0), "group", 1000)
    prow.append(
        {
            "comparison": lab,
            "d_small_sens": "%+.3f (%+.3f to %+.3f)" % dsm,
            "d_auc": "%+.3f (%+.3f to %+.3f)" % dau,
            "ci_above_zero": dsm[1] > 0,
        }
    )
if prow:
    pt = pd.DataFrame(prow)
    pt.to_csv("results/box_vs_control.csv", index=False)
    print("\nBoxes vs matching no-box control:\n" + pt.to_string(index=False))


# exploratory: each run against the mean of the two baseline seeds
s43 = load(base + "_s43")
mrows = []
for lab, name in settings:
    if name in (base, base + "_s43") or not os.path.exists(f"results/oof_{name}.csv"):
        continue
    o = load(name)
    if o.fold.nunique() < 5:
        continue
    d = b[["y", "group"]].copy()
    d["a"], d["b0"], d["b1"] = o.s.reindex(d.index), b.s, s43.s.reindex(d.index)
    d["small"] = (d.y == 1) & (boxes.rel_area.reindex(d.index) <= q1)
    ss = lambda col: lambda x: (x[col][x.small] >= 0).mean()
    dm = group_bootstrap(d, lambda x: ss("a")(x) - (ss("b0")(x) + ss("b1")(x)) / 2, "group", 1000)
    mrows.append({"setting": lab, "d_small_sens_vs_mean_of_baselines": "%+.3f (%+.3f to %+.3f)" % dm})
if mrows:
    mt = pd.DataFrame(mrows)
    mt.to_csv("results/vs_mean_baseline.csv", index=False)
    print("\nExploratory: vs mean of the two baseline seeds:\n" + mt.to_string(index=False))
