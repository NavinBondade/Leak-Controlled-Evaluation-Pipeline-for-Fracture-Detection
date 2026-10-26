"""Illustrative break-even grid for AI as a second reader."""

import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fracai.breakeven import breakeven_grid

PREV, USUAL_SPEC, C_FP = 0.20, 0.90, 50.0
MISS = np.array([0.02, 0.05, 0.10, 0.15, 0.20])
C_MISS = np.array([200, 500, 1000, 2000, 5000])
AI = {
    "FracAtlas internal (sens 0.81, spec 0.90)": (0.81, 0.90),
    "GRAZ initial exams, external (sens 0.54, spec 0.79)": (0.54, 0.79),
}


def assisted(m, ai):
    """usual miss rate m -> assisted sens/spec under the safety-net rule."""
    return 1 - m * (1 - ai[0]), USUAL_SPEC * ai[1]


rows, grids = [], {}
for lab, ai in AI.items():
    g = np.zeros((len(MISS), len(C_MISS)))
    for i, m in enumerate(MISS):
        g[i] = breakeven_grid(PREV, USUAL_SPEC, assisted(m, ai), [m], C_MISS, C_FP)[0]
    grids[lab] = g
    for i, m in enumerate(MISS):
        for j, c in enumerate(C_MISS):
            rows.append({"ai": lab, "usual_miss_rate": m, "cost_per_miss": c, "breakeven_price": g[i, j]})
os.makedirs("results/figures", exist_ok=True)
pd.DataFrame(rows).to_csv("results/breakeven.csv", index=False)

vmax = max(abs(g).max() for g in grids.values())
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for ax, (lab, g) in zip(axes, grids.items()):
    im = ax.imshow(g, cmap="RdBu", vmin=-vmax, vmax=vmax, origin="lower", aspect="auto")
    ax.set_xticks(range(len(C_MISS)), [f"{c:,}" for c in C_MISS])
    ax.set_yticks(range(len(MISS)), [f"{m:.0%}" for m in MISS])
    for i in range(len(MISS)):
        for j in range(len(C_MISS)):
            ax.text(j, i, f"{g[i, j]:.1f}", ha="center", va="center", fontsize=9)
    ax.set_title(lab, fontsize=10)
    ax.set_xlabel("cost of one missed fracture (arbitrary currency units)")
axes[0].set_ylabel("usual-care miss rate")
fig.colorbar(im, ax=axes, label="max cost-neutral price per scan")
fig.suptitle(
    f"ILLUSTRATIVE break-even (prevalence {PREV:.0%}, false-alarm cost {C_FP:.0f}). Blue: AI can pay for itself; red: cannot.\n"
    "AI operating points are measured; costs and miss rates are not sourced.",
    fontsize=10,
)
fig.savefig("results/figures/breakeven.png", dpi=90, bbox_inches="tight")
for lab, g in grids.items():
    print(lab)
    print(pd.DataFrame(g.round(2), index=[f"miss {m:.0%}" for m in MISS], columns=C_MISS).to_string())
