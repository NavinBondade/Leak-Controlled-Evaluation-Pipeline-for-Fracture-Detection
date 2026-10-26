"""Figures for the shortcut audit, drawn from the saved audit tables in results/audit2/.
Output: results/figures/audit/{cast_effect,probe_auc,age_sex}.png"""

import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fracai.plotstyle import BLUE, GREY, ORANGE, RED, use

use()
IN = Path("results/audit2")
OUT = Path("results/figures/audit")
OUT.mkdir(parents=True, exist_ok=True)


def split_ci(s):
    """'0.998 (0.994 to 1.000)' -> (0.998, 0.994, 1.000)"""
    m = re.match(r"\s*(-?[\d.]+)\s*\(\s*(-?[\d.]+) to (-?[\d.]+)\)", s)
    return tuple(float(x) for x in m.groups())


# 1. the cast effect: score by group, and the within-patient difference
strat = pd.read_csv(IN / "A1_stratified_followup.csv")
paired = pd.read_csv(IN / "A1_paired_within_patient.csv")
paired = paired[paired.setting.str.startswith("follow-up only")]

fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
# (a) means: initial exam without cast, follow-up without cast, follow-up with cast
dev = pd.read_csv(IN / "A3_device.csv")
row = dev[(dev.set == "no-cast initial exams") & (dev.subgroup == "ALL")].iloc[0]
ini = {0: row.mean_score_y0, 1: row.mean_score_y1}
x = np.arange(2)
w = 0.26
neg = strat[(strat.y == 0) & (strat.view == "all")].iloc[0]
pos = strat[(strat.y == 1) & (strat.view == "all")].iloc[0]
groups = [
    ("Initial exam, no cast", [ini[0], ini[1]], GREY),
    ("Follow-up, no cast", [neg.mean_nocast, pos.mean_nocast], BLUE),
    ("Follow-up, cast", [neg.mean_cast, pos.mean_cast], ORANGE),
]
for i, (lab, vals, col) in enumerate(groups):
    bars = a.bar(x + (i - 1) * w, vals, w, color=col, label=lab)
    for r, v in zip(bars, vals):
        a.text(
            r.get_x() + r.get_width() / 2,
            v + (0.15 if v >= 0 else -0.15),
            f"{v:+.1f}",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=8.5,
        )
a.axhline(0, color="k", lw=0.8)
a.set_xticks(x, ["Non-fractured", "Fractured"])
a.set_ylabel("Mean score (0 = decision threshold)")
a.set_title("(a) Score by exam type and cast")
a.legend(frameon=False, fontsize=8.5, loc="upper left")
a.set_ylim(-2.2, 9)

# (b) within-patient change from a cast, fractured wrists
rows = paired[paired.stratum.isin(["y=1", "y=1, frontal", "y=1, lateral"])]
names = {"y=1": "All views", "y=1, frontal": "Frontal", "y=1, lateral": "Lateral"}
for j, r in enumerate(rows.itertuples()):
    b.errorbar(
        r.mean_diff_cast_minus_nocast,
        j,
        xerr=[[r.mean_diff_cast_minus_nocast - r.lo], [r.hi - r.mean_diff_cast_minus_nocast]],
        fmt="o",
        color=BLUE,
        capsize=3,
    )
    b.text(r.hi + 0.15, j, f"{r.mean_diff_cast_minus_nocast:+.1f}  ({r.patients} patients)", va="center", fontsize=8.5)
b.axvline(0, color="k", lw=0.8)
b.set_yticks(range(len(rows)), [names[s] for s in rows.stratum])
b.set_xlim(0, 8.8)
b.set_xlabel("Score change when a cast is present (same patient)")
b.set_title("(b) Within-patient cast effect, fractured wrists")
b.set_ylim(len(rows) - 0.5, -0.5)
fig.tight_layout()
fig.savefig(OUT / "cast_effect.png")
plt.close(fig)

# 2. what a probe can read from the model's features
probe = pd.read_csv(IN / "b1_probe_auc.csv")
keep = [
    ("cast | whole sample", "Cast"),
    ("cast | non-fractured follow-ups only", "Cast, non-fractured follow-ups"),
    ("follow-up | whole sample", "Follow-up exam"),
    ("follow-up | no cast", "Follow-up exam, no cast"),
    ("fracture | whole sample", "Fracture"),
    ("fracture | initial exams, no cast", "Fracture, initial exams, no cast"),
    ("fracture | casted only", "Fracture, casted wrists"),
]
fig, ax = plt.subplots(figsize=(10, 4.2))
for j, (task, lab) in enumerate(keep):
    r = probe[probe.task == task].iloc[0]
    p, lo, hi = split_ci(r.probe_auc)
    q, qlo, qhi = split_ci(r.pixel_probe_auc)
    ax.errorbar(
        p,
        j - 0.12,
        xerr=[[p - lo], [hi - p]],
        fmt="o",
        color=BLUE,
        capsize=3,
        label="Model features" if j == 0 else None,
    )
    ax.errorbar(
        q, j + 0.12, xerr=[[q - qlo], [qhi - q]], fmt="s", color=GREY, capsize=3, label="Raw pixels" if j == 0 else None
    )
ax.set_yticks(range(len(keep)), [k[1] for k in keep])
ax.invert_yaxis()
ax.axvline(0.5, color="k", lw=0.8, ls=":")
ax.set_xlim(0.45, 1.02)
ax.set_xlabel("Probe AUC, patients held out")
ax.set_title("Which properties can be read from the model's features")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "probe_auc.png")
plt.close(fig)

# 3. age and sex on initial exams without a cast
ag = pd.read_csv(IN / "A2_age_sex.csv")
ag = ag[(ag.set == "no-cast initial exams") & ag.subgroup.isin(["age_band", "sex"]) & (ag.level != "O")]
ag = ag.assign(label=ag.level.map(lambda s: {"F": "Female", "M": "Male"}.get(s, s)))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, (col, lo, hi, title) in zip(
    axes,
    [("auc", "auc_lo", "auc_hi", "(a) AUC"), ("sens", "sens_lo", "sens_hi", "(b) Sensitivity at the fixed threshold")],
):
    for j, r in enumerate(ag.itertuples()):
        v, lo_v, hi_v = getattr(r, col), getattr(r, lo), getattr(r, hi)
        ax.errorbar(v, j, xerr=[[v - lo_v], [hi_v - v]], fmt="o", color=RED if r.subgroup == "sex" else BLUE, capsize=3)
    ax.set_yticks(range(len(ag)), [f"{r.label}  (n={r.n:,})" for r in ag.itertuples()])
    ax.set_title(title)
    ax.set_xlabel(col.upper() if col == "auc" else "Sensitivity")
axes[0].invert_yaxis()
fig.tight_layout()
fig.savefig(OUT / "age_sex.png")
plt.close(fig)
print("saved", *sorted(p.name for p in OUT.iterdir()))
