"""Subgroup audit of an out-of-fold run: run_audit.py fracatlas [spec]."""

import sys

import numpy as np
import pandas as pd

from fracai.audit import hardware_shortcut, positives_only_sensitivity, size_shortcut, subgroup_table
from fracai.metrics import ece_brier, safe_auc, thr_at_spec

name = sys.argv[1]
spec = float(sys.argv[2]) if len(sys.argv) > 2 else 0.90
oof = pd.read_csv(f"results/oof_{name}.csv")
val = pd.read_csv(f"results/val_{name}.csv")

ece, brier = ece_brier(oof.y, oof.score)  # raw logits, before the threshold shift
thr = {k: thr_at_spec(v.y, v.score, spec) for k, v in val.groupby("test_fold")}
# shift each fold's scores so a common threshold of 0 equals that fold's validation threshold
oof["score"] = oof.score - oof.fold.map(thr)
oof["group"] = oof["group"] if "group" in oof else np.arange(len(oof))
if name.startswith("fracatlas"):
    oof = oof.merge(pd.read_csv("results/fracatlas_sizes.csv")[["case_id", "size_class"]], on="case_id", how="left")
    bx = pd.read_csv("results/fracatlas_boxes.csv")[["case_id", "long_side_512"]]
    oof = oof.merge(bx, on="case_id", how="left")
    oof["fracture_size"] = pd.qcut(oof.long_side_512, 4, labels=["Q1 smallest", "Q2", "Q3", "Q4 largest"])
cols = [c for c in ["bodypart", "view", "hardware", "multiscan", "cast", "uncertain", "sex", "size_class"] if c in oof]
if "fracture_count" in oof:
    oof["fracture_n"] = oof.fracture_count.clip(upper=3).map({0: "0", 1: "1", 2: "2", 3: "3+"})
if "age" in oof:
    oof["age_band"] = pd.cut(oof.age, [0, 5, 10, 14, 20], right=False).astype(str)
    cols.append("age_band")

pd.set_option("display.width", 220, "display.max_columns", 30)
fmt = lambda t: t.assign(**{c: t[c].round(3) for c in t.select_dtypes("float").columns})
sub = subgroup_table(oof, cols, 0.0)
hw = hardware_shortcut(oof, 0.0)
sub.to_csv(f"results/audit_{name}_subgroups.csv", index=False)
hw.to_csv(f"results/audit_{name}_hardware.csv", index=False)
print(
    f"threshold: fold-wise, {spec:.0%} specificity on validation scores; fold thresholds:",
    {k: round(v, 2) for k, v in thr.items()},
)
print(f"calibration (raw logits): ECE {ece:.3f}, Brier {brier:.3f}")
print(
    "overall AUC",
    round(safe_auc(oof.y, oof.score), 4),
    "| per-fold AUC",
    oof.groupby("fold").apply(lambda d: round(safe_auc(d.y, d.score), 3)).tolist(),
)
print(
    "\n== hardware shortcut ==\n",
    fmt(hw[["set", "n", "pos", "auc", "auc_lo", "auc_hi", "sens", "spec", "unreliable"]]).to_string(index=False),
)
print(
    "\n== subgroups ==\n",
    fmt(
        sub[["subgroup", "level", "n", "pos", "prevalence", "auc", "auc_lo", "auc_hi", "sens", "spec", "unreliable"]]
    ).to_string(index=False),
)

pos_tabs = [positives_only_sensitivity(oof, c, 0.0) for c in ["fracture_n", "fracture_size"] if c in oof]
if pos_tabs:
    pt = pd.concat(pos_tabs)
    pt.to_csv(f"results/audit_{name}_positives_only.csv", index=False)
    print("\n== sensitivity among positives only (no AUC possible) ==\n", fmt(pt).to_string(index=False))
if "size_class" in oof:
    sz = size_shortcut(oof, 0.0)
    sz.to_csv(f"results/audit_{name}_size_shortcut.csv", index=False)
    print(
        "\n== image-size shortcut ==\n",
        fmt(sz[["set", "n", "pos", "auc", "auc_lo", "auc_hi", "sens", "spec", "unreliable"]]).to_string(index=False),
    )
