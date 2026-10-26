"""Illustrative break-even grid for AI as a second reader."""

import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fracai.breakeven import breakeven_grid
from fracai.plotstyle import use

use()
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
titles = ["(a) FracAtlas internal\n(sens 0.81, spec 0.90)", "(b) GRAZ initial exams, external\n(sens 0.54, spec 0.79)"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, g, title in zip(axes, grids.values(), titles):
    im = ax.imshow(g, cmap="RdBu", vmin=-vmax, vmax=vmax, origin="lower", aspect="auto")
    ax.set_xticks(range(len(C_MISS)), [f"{c:,}" for c in C_MISS])
    ax.set_yticks(range(len(MISS)), [f"{m:.0%}" for m in MISS])
    for i in range(len(MISS)):
        for j in range(len(C_MISS)):
            dark = abs(g[i, j]) > 0.6 * vmax
            ax.text(j, i, f"{g[i, j]:.1f}", ha="center", va="center", fontsize=8.5, color="white" if dark else "black")
    ax.set_title(title)
    ax.set_xlabel("Cost of one missed fracture (arbitrary units)")
    for sp in ax.spines.values():
        sp.set_visible(False)
axes[0].set_ylabel("Usual-care miss rate")
fig.colorbar(im, ax=axes, label="Max cost-neutral price per scan", shrink=0.9)
fig.savefig("results/figures/breakeven.png", bbox_inches="tight", pad_inches=0.08)
for lab, g in grids.items():
    print(lab)
    print(pd.DataFrame(g.round(2), index=[f"miss {m:.0%}" for m in MISS], columns=C_MISS).to_string())
