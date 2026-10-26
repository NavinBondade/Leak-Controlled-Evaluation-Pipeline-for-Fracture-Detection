"""Slide-sized charts built from the saved results. Output: results/figures/slides/"""

import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fracai.metrics import safe_auc, thr_at_spec

OUT = Path("results/figures/slides")
OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GREY, RED = "#2a6f97", "#e08a1e", "#7f8c8d", "#c0392b"
plt.rcParams.update(
    {"figure.figsize": (11, 6.2), "font.size": 13, "axes.spines.top": False, "axes.spines.right": False}
)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("saved", OUT / name)


oof, val = pd.read_csv("results/oof_fracatlas.csv"), pd.read_csv("results/val_fracatlas.csv")
thr = {k: thr_at_spec(v.y, v.score, 0.90) for k, v in val.groupby("test_fold")}
oof["s"] = oof.score - oof.fold.map(thr)
graz = pd.read_csv("results/ext_graz_scores.csv")
clean = graz[(graz.cast == 0) & (graz.hardware == 0)]

# ROC curves, internal against external
fig, ax = plt.subplots(figsize=(7.5, 7))
for lab, d, y, s, c in [
    ("FracAtlas, internal", oof, oof.y, oof.s, BLUE),
    ("GRAZ, external", graz, graz.y, graz.score, ORANGE),
    ("GRAZ, no cast or metal", clean, clean.y, clean.score, RED),
]:
    fpr, tpr, _ = roc_curve(y, s)
    ax.plot(fpr, tpr, color=c, lw=2.5, label=f"{lab} (AUC {safe_auc(y, s):.2f})")
    ax.plot((s[y == 0] >= 0).mean(), (s[y == 1] >= 0).mean(), "o", color=c, ms=9)
ax.plot([0, 1], [0, 1], ":", color=GREY)
ax.set_xlabel("1 - specificity")
ax.set_ylabel("Sensitivity")
ax.set_title(
    "Same model, same threshold rule: the external curve is lower\n(dots mark the fixed operating point)",
    loc="left",
    fontsize=13,
)
ax.legend(loc="lower right", frameon=False)
save(fig, "01_roc_internal_vs_external.png")

# the cast shortcut
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
bins = np.linspace(-12, 14, 40)
for ax, (title, d) in zip(axes, [("No cast", graz[graz.cast == 0]), ("Cast present", graz[graz.cast == 1])]):
    ax.hist(d.score[d.y == 0], bins, alpha=0.7, color=BLUE, label="not fractured")
    ax.hist(d.score[d.y == 1], bins, alpha=0.7, color=ORANGE, label="fractured")
    ax.axvline(0, color="k", lw=1.5)
    ax.set_title(f"{title} ({len(d):,} images)", loc="left")
    ax.set_xlabel("Score (0 = flag as fracture)")
axes[0].set_ylabel("Images")
axes[0].legend(frameon=False)
fig.suptitle(
    "A cast pushes the score above the threshold whatever the label", x=0.01, ha="left", fontsize=15, weight="bold"
)
save(fig, "02_cast_shortcut.png")

# small-fracture sensitivity by run
tab = pd.read_csv("results/box_comparison.csv")
pat = re.compile(r"([\d.]+) \(([\d.]+)-([\d.]+)\)")
runs = []
for _, r in tab.iterrows():
    m = pat.match(r.sens_smallest_quartile)
    runs.append((r.setting.replace("CONTROL ", "control: "), *map(float, m.groups())))
fig, ax = plt.subplots(figsize=(11, 6.2))
for i, (lab, p, lo, hi) in enumerate(runs[::-1]):
    c = GREY if "baseline" in lab or "control" in lab else BLUE
    ax.plot([lo, hi], [i, i], color=c, lw=3)
    ax.plot(p, i, "o", color=c, ms=10)
ax.set_yticks(range(len(runs)), [r[0] for r in runs[::-1]])
ax.axvspan(0.761, 0.794, color=ORANGE, alpha=0.2, label="two baseline seeds")
ax.set_xlabel("Sensitivity on the smallest-fracture quartile (95% interval)")
ax.set_title(
    "No variant clearly beats the baseline; seed-to-seed change is as large as the gains", loc="left", fontsize=14
)
ax.legend(frameon=False, loc="upper right")
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
names = ["no AI", "alert\n(internal)", "triage\n(internal)", "alert\n(external)", "triage\n(external)"]
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
for ax, col, title in [
    (axes[0], "caught_at_ed", "Missed fractures caught\nbefore the patient leaves (a year)"),
    (axes[1], "false_alarms_per_day", "False-alarm re-reads a day"),
]:
    d = sim[(sim.reporter_utilisation == 0.75) & sim.scenario.isin(order)].set_index("scenario").loc[order]
    ax.bar(range(5), d[col], color=[GREY, BLUE, BLUE, ORANGE, ORANGE])
    ax.set_xticks(range(5), names, fontsize=10)
    ax.set_title(title, loc="left", fontsize=12)
w = 0.38
for j, (u, c) in enumerate([(0.75, BLUE), (0.95, RED)]):
    d = sim[(sim.reporter_utilisation == u) & sim.scenario.isin(order)].set_index("scenario").loc[order]
    axes[2].bar(
        np.arange(5) + (j - 0.5) * w, d.undetected_hours_per_year, w, color=c, label=f"reporter {int(u * 100)}% busy"
    )
axes[2].set_xticks(range(5), names, fontsize=10)
axes[2].set_title("Undetected fracture-hours a year", loc="left", fontsize=12)
axes[2].legend(frameon=False, fontsize=10)
fig.suptitle(
    "Illustrative ED simulation, 5% usual miss rate: assumptions in docs, not a prediction",
    x=0.01,
    ha="left",
    fontsize=14,
    weight="bold",
)
save(fig, "04_ed_simulation.png")
