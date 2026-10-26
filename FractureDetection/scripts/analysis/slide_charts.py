"""Report charts built from the saved results. Output: results/figures/slides/"""

import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fracai.metrics import safe_auc, thr_at_spec
from fracai.plotstyle import BLUE, GREY, ORANGE, RED, use

use()
OUT = Path("results/figures/slides")
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("saved", OUT / name)


oof, val = pd.read_csv("results/oof_fracatlas.csv"), pd.read_csv("results/val_fracatlas.csv")
thr = {k: thr_at_spec(v.y, v.score, 0.90) for k, v in val.groupby("test_fold")}
oof["s"] = oof.score - oof.fold.map(thr)
graz = pd.read_csv("results/ext_graz_scores.csv")
clean = graz[(graz.cast == 0) & (graz.hardware == 0)]

# ROC curves, internal against external
fig, ax = plt.subplots(figsize=(6.2, 5.4))
for lab, y, s, c in [
    ("FracAtlas, internal", oof.y, oof.s, BLUE),
    ("GRAZ, external", graz.y, graz.score, ORANGE),
    ("GRAZ, no cast or metal", clean.y, clean.score, RED),
]:
    fpr, tpr, _ = roc_curve(y, s)
    ax.plot(fpr, tpr, color=c, lw=2, label=f"{lab} (AUC {safe_auc(y, s):.3f})")
    ax.plot((s[y == 0] >= 0).mean(), (s[y == 1] >= 0).mean(), "o", color=c, ms=7, mec="white")
ax.plot([0, 1], [0, 1], ":", color=GREY, lw=1)
ax.set_xlabel("1 - specificity")
ax.set_ylabel("Sensitivity")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.set_aspect("equal")
ax.legend(loc="lower right")
save(fig, "01_roc_internal_vs_external.png")

# the cast shortcut
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
bins = np.linspace(-12, 14, 40)
for ax, (title, d) in zip(axes, [("(a) No cast", graz[graz.cast == 0]), ("(b) Cast present", graz[graz.cast == 1])]):
    ax.hist(d.score[d.y == 0], bins, alpha=0.75, color=BLUE, label="not fractured")
    ax.hist(d.score[d.y == 1], bins, alpha=0.75, color=ORANGE, label="fractured")
    ax.axvline(0, color="k", lw=1)
    ax.set_title(f"{title} ({len(d):,} images)")
    ax.set_xlabel("Score (0 = operating point)")
axes[0].set_ylabel("Images")
axes[0].legend(loc="upper right")
fig.tight_layout()
save(fig, "02_cast_shortcut.png")

# small-fracture sensitivity by run
tab = pd.read_csv("results/box_comparison.csv")
pat = re.compile(r"([\d.]+) \(([\d.]+)-([\d.]+)\)")
runs = []
for _, r in tab.iterrows():
    m = pat.match(r.sens_smallest_quartile)
    runs.append((r.setting.replace("CONTROL ", "control: "), *map(float, m.groups())))
fig, ax = plt.subplots(figsize=(10, 4.4))
ax.axvspan(0.761, 0.794, color=ORANGE, alpha=0.2, label="two baseline seeds")
for i, (lab, p, lo, hi) in enumerate(runs[::-1]):
    c = GREY if "baseline" in lab or "control" in lab else BLUE
    ax.plot([lo, hi], [i, i], color=c, lw=2)
    ax.plot(p, i, "o", color=c, ms=7)
ax.set_yticks(range(len(runs)), [r[0] for r in runs[::-1]])
ax.set_xlabel("Sensitivity on the smallest-fracture quartile (95% interval)")
ax.legend(loc="lower right")
fig.tight_layout()
save(fig, "03_small_fracture_runs.png")

# ED simulation
sim = pd.read_csv("results/hospital_sim.csv")
sim = sim[sim.usual_miss_rate == 0.05]
order = [
    "no AI",
    "AI internal point: alert",
    "AI internal point: triage",
    "AI external first-visit point: alert",
    "AI external first-visit point: triage",
]
names = ["No AI", "Alert\n(int.)", "Triage\n(int.)", "Alert\n(ext.)", "Triage\n(ext.)"]
colours = [GREY, BLUE, BLUE, ORANGE, ORANGE]
fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
d = sim[(sim.reporter_utilisation == 0.75) & sim.scenario.isin(order)].set_index("scenario").loc[order]
for ax, col, title in [
    (axes[0], "caught_at_ed", "(a) Misses caught at the ED\n(fractures a year)"),
    (axes[1], "false_alarms_per_day", "(b) False-alarm re-reads\n(a day)"),
]:
    bars = ax.bar(range(5), d[col], color=colours)
    ax.bar_label(bars, fmt="%.0f" if col == "caught_at_ed" else "%.1f", fontsize=8, padding=2)
    ax.set_xticks(range(5), names)
    ax.set_title(title)
w = 0.38
for j, (u, c) in enumerate([(0.75, BLUE), (0.95, RED)]):
    d = sim[(sim.reporter_utilisation == u) & sim.scenario.isin(order)].set_index("scenario").loc[order]
    axes[2].bar(np.arange(5) + (j - 0.5) * w, d.undetected_hours_per_year, w, color=c, label=f"reporter {int(u * 100)}% busy")
axes[2].set_xticks(range(5), names)
axes[2].set_title("(c) Undetected fracture-hours\n(a year)")
axes[2].legend(loc="upper right")
fig.tight_layout()
save(fig, "04_ed_simulation.png")
