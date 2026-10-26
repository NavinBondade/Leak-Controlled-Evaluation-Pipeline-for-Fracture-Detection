"""Draw the one-page summary figure."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

fig = plt.figure(figsize=(11.7, 8.3))
fig.suptitle(
    "Open fracture-AI evaluation: it looks good internally, and transfers poorly",
    fontsize=15,
    weight="bold",
    x=0.02,
    ha="left",
)
fig.text(
    0.02,
    0.925,
    "ConvNeXt-Tiny trained on FracAtlas (leak-controlled 5-fold), tested externally on GRAZPEDWRI-DX. "
    "Public data only; 95% group-bootstrap CIs; not a clinical validation.",
    fontsize=9,
)

ax = fig.add_axes([0.28, 0.55, 0.68, 0.33])
rows = [
    ("FracAtlas internal, all", 0.923, 0.910, 0.935),
    ("FracAtlas internal, hand only", 0.899, 0.876, 0.920),
    ("GRAZ external, all", 0.825, 0.816, 0.833),
    ("GRAZ external, no cast / no metal", 0.744, 0.735, 0.753),
    ("GRAZ external, initial exams", 0.697, 0.687, 0.707),
    ("GRAZ external, cast present", 0.560, None, None),
]
for i, (lab, a, lo, hi) in enumerate(rows[::-1]):
    ax.barh(i, a, color="#2a6f97" if "internal" in lab else "#c0392b", height=0.6)
    if lo:
        ax.plot([lo, hi], [i, i], color="k", lw=1.5)
    ax.text((hi or a) + 0.01, i, f"{a:.3f}", va="center", fontsize=9)
ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]], fontsize=9)
ax.set_xlim(0.4, 1)
ax.axvline(0.5, color="grey", ls=":")
ax.set_xlabel("AUC (0.5 = chance)")
ax.set_title("1. Performance drops outside the training data (blue internal, red external)", loc="left", fontsize=10)

ax = fig.add_axes([0.06, 0.12, 0.27, 0.30])
ax.bar(["sensitivity", "specificity"], [0.995, 0.017], color=["#2a6f97", "#c0392b"])
for x, v in enumerate([0.995, 0.017]):
    ax.text(x, v + 0.02, f"{v:.1%}", ha="center", fontsize=9)
ax.set_ylim(0, 1.1)
ax.set_title("2. Casted images (96% fractured):\nmodel flags nearly all of them", loc="left", fontsize=10)

ax = fig.add_axes([0.40, 0.12, 0.22, 0.30])
ax.bar(["frontal", "lateral", "max of both"], [0.580, 0.785, 0.775], color="#7f8c8d")
ax.set_ylim(0.4, 0.9)
ax.set_ylabel("AUC")
for x, v in enumerate([0.580, 0.785, 0.775]):
    ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
ax.set_title("3. GRAZ first visits (4,951 studies):\nfrontal weak; combining views no help", loc="left", fontsize=10)

ax = fig.add_axes([0.72, 0.12, 0.24, 0.30])
b = pd.read_csv("results/breakeven.csv")
b = b[(b.usual_miss_rate == 0.10)]
for lab, g in b.groupby("ai"):
    ax.plot(
        g.cost_per_miss,
        g.breakeven_price,
        marker="o",
        label="internal point" if "FracAtlas" in lab else "external point",
    )
ax.axhline(0, color="k", lw=0.8)
ax.set_xscale("log")
ax.set_xlabel("cost per missed fracture")
ax.set_ylabel("max cost-neutral price / scan")
ax.legend(fontsize=8)
ax.set_title("4. Break-even (ILLUSTRATIVE,\n10% usual miss rate; nothing sourced)", loc="left", fontsize=10)

fig.text(
    0.02,
    0.003,
    "Caveats: single training run; no patient ids in FracAtlas (groups from near-duplicate images); GRAZ is paediatric wrist only;\n"
    "licences unverified; break-even costs are placeholders. Leakage inflation NOT demonstrated.\n"
    "Small-fracture fixes (boxes, fine head): inconclusive, gains +0.03 to +0.05, not separable from seed variation with 2 seeds.",
    fontsize=8,
    color="#444",
)
fig.savefig("results/figures/one_page_summary.png", dpi=110)
