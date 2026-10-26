"""Pipeline overview figure for the README. Run from the FractureDetection folder."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

BLUE, ORANGE, GREY = "#2a6f97", "#e08a1e", "#7f8c8d"
TEXT = "#2b2b2b"
W = 2.8  # box width, in figure inches

fig = plt.figure(figsize=(16, 8))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 8)
ax.axis("off")


def box(x, y, h, title, lines, color, dashed=False):
    """Rounded box centred on (x, y); returns its edges for the arrows."""
    patch = FancyBboxPatch(
        (x - W / 2, y - h / 2), W, h, boxstyle="round,pad=0,rounding_size=0.15",
        facecolor=to_rgba(color, 0.07), edgecolor=color, lw=1.6, ls=(0, (5, 3)) if dashed else "-",
    )
    ax.add_patch(patch)
    # centre the title and body block vertically
    top = y + (0.42 + 0.235 * len(lines)) / 2
    ax.text(x, top, title, ha="center", va="top", fontsize=13.5, weight="bold", color=color)
    ax.text(x, top - 0.42, "\n".join(lines), ha="center", va="top", fontsize=12.5, color=TEXT, linespacing=1.35)
    return dict(l=(x - W / 2, y), r=(x + W / 2, y), t=(x, y + h / 2), b=(x, y - h / 2))


def arrow(a, b, label=None, color=TEXT, offset=(0, 0.18), ha="center"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, lw=1.1, color=color, shrinkA=2, shrinkB=2))
    if label:
        mx, my = (a[0] + b[0]) / 2 + offset[0], (a[1] + b[1]) / 2 + offset[1]
        ax.text(mx, my, label, ha=ha, va="center", fontsize=11.5, style="italic", color=TEXT)


cols = [1.55 + 3.225 * k for k in range(5)]
top, mid, low = 6.05, 3.75, 1.3

# main pipeline, internal data
data = box(cols[0], top, 1.75, "FracAtlas (internal)", ["X-rays of mixed body", "parts, fracture boxes,", "no patient ids"], BLUE)
split = box(cols[1], top, 1.75, "Leak-controlled split",
            ["near-duplicates (perceptual", "hash) and multi-scan runs", "grouped, then 5 stratified", "group folds"], BLUE)
train = box(cols[2], top, 1.75, "ConvNeXt-Tiny, 5-fold CV",
            ["512 px; per fold 3 train,", "1 validation, 1 scored", "gives out-of-fold scores"], BLUE)
thresh = box(cols[3], top, 1.75, "Threshold per fold",
             ["set at 90% specificity", "on the validation fold,", "then frozen"], BLUE)
internal = box(cols[4], top, 1.75, "Internal evaluation",
               ["AUC, sensitivity and", "specificity at the frozen", "threshold, 95% group-", "bootstrap CIs"], BLUE)
for a, b in [(data, split), (split, train), (train, thresh), (thresh, internal)]:
    arrow(a["r"], b["l"])

# external test and audits
graz = box(cols[0], mid, 1.45, "GRAZPEDWRI-DX", ["external, paediatric wrist,", "grouped by patient id"], ORANGE)
external = box(cols[3], mid, 1.6, "External test",
               ["5 fold models averaged,", "same frozen thresholds,", "patient-bootstrap CIs"], ORANGE)
audits = box(cols[4], mid, 1.85, "Audits",
             ["cast and hardware shortcut", "subgroups, fracture size", "Grad-CAM and occlusion", "resolution sensitivity"],
             ORANGE)
arrow(graz["r"], external["l"], "held out: never used for training or tuning", offset=(0, 0.22))
arrow(thresh["b"], external["t"], "models and thresholds", offset=(-0.12, 0), ha="right")
arrow(internal["b"], audits["t"])
arrow(external["r"], audits["l"])

# secondary and illustrative parts
small = box(cols[3], low, 1.5, "Small-fracture runs",
            ["box supervision, aux heads,", "oversampling, no-box controls", "result: inconclusive"], GREY, dashed=True)
illus = box(cols[4], low, 1.5, "Illustrative only",
            ["break-even price per scan,", "ED pathway simulation,", "assumed inputs"], GREY, dashed=True)
arrow((audits["l"][0] + 0.25, audits["b"][1]), (small["r"][0] - 0.25, small["t"][1]), color=GREY)
arrow(audits["b"], illus["t"], "operating points", offset=(0.12, 0), ha="left", color=GREY)

ax.text(0.25, 7.65, "Leak-controlled evaluation pipeline", fontsize=17, weight="bold", color=TEXT, va="center")
legend = [
    Patch(facecolor=to_rgba(BLUE, 0.07), edgecolor=BLUE, label="main pipeline (FracAtlas)"),
    Patch(facecolor=to_rgba(ORANGE, 0.07), edgecolor=ORANGE, label="external test and audits"),
    Patch(facecolor=to_rgba(GREY, 0.07), edgecolor=GREY, ls="--", label="secondary or illustrative"),
]
ax.legend(handles=legend, loc="lower left", bbox_to_anchor=(0.015, 0.06), frameon=False, fontsize=12.5,
          handlelength=2.2, handleheight=1.2)

out = Path("fig/pipeline.png")
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=200, facecolor="white")
print("saved", out)
