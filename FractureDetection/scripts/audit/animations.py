"""MP4 animations for slides: a forward pass through the network, Grad-CAM on a detected and a missed small fracture,
and the model pipeline. Output: results/figures/slides/anim_*.mp4"""

import shutil
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import FancyArrowPatch, Rectangle

from fracai.data import load_boxes
from fracai.gradcam import gradcam
from fracai.metrics import thr_at_spec
from fracai.plotstyle import use
from fracai.train import build_model, dev, load_cache, to_batch

use()
matplotlib.rcParams["animation.ffmpeg_path"] = shutil.which("ffmpeg")
OUT = Path("results/figures/slides")
OUT.mkdir(parents=True, exist_ok=True)
FPS = 15
BLUE, ORANGE, GREY = "#2a6f97", "#e08a1e", "#7f8c8d"

oof, val = pd.read_csv("results/oof_fracatlas.csv"), pd.read_csv("results/val_fracatlas.csv")
thr = {k: thr_at_spec(v.y, v.score, 0.90) for k, v in val.groupby("test_fold")}
oof["score"] -= oof.fold.map(thr)
meta = pd.read_csv("results/fracatlas_meta.csv")
row_of = {c: i for i, c in enumerate(meta.case_id)}
arr = load_cache("fracatlas", 512)
boxes = load_boxes()
area = pd.read_csv("results/fracatlas_boxes.csv").set_index("case_id").rel_area
oof["rel_area"] = oof.case_id.map(area)
pos = oof[(oof.y == 1) & oof.case_id.isin(boxes)]
small = pos[pos.rel_area <= pos.rel_area.quantile(0.25)]
detected = pos[pos.score >= 3].sample(6, random_state=7).iloc[0]
small_hit = small[small.score >= 0].sample(3, random_state=7).iloc[0]
small_miss = small[small.score < 0].sample(3, random_state=7).iloc[0]


def load_fold_model(fold):
    model = build_model("convnext_tiny").to(dev)
    model.load_state_dict(torch.load(f"results/model_fracatlas_fold{int(fold)}.pt", map_location=dev))
    return model.eval()


def save(fig, update, frames, name):
    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / FPS)
    writer = animation.FFMpegWriter(fps=FPS, bitrate=4000, extra_args=["-pix_fmt", "yuv420p"])
    anim.save(OUT / name, writer=writer, dpi=100)
    anim.save(OUT / name.replace(".mp4", ".gif"), writer=animation.PillowWriter(fps=FPS), dpi=90)
    update(frames - 1)
    fig.savefig(OUT / name.replace("anim_", "final_").replace(".mp4", ".png"), dpi=150)
    plt.close(fig)
    print("saved", OUT / name)


# 1. one image through the network
model = load_fold_model(detected.fold)
x = to_batch(arr, [row_of[detected.case_id]], False, 512)
acts = {}
hooks = [model.stem.register_forward_hook(lambda m, i, o: acts.__setitem__("Stem", o))]
for k, st in enumerate(model.stages):
    hooks.append(st.register_forward_hook(lambda m, i, o, k=k: acts.__setitem__(f"Stage {k + 1}", o)))
with torch.no_grad():
    model(x)
for h in hooks:
    h.remove()
cam, _ = gradcam(model, x)
maps = {name: a.abs().mean(1)[0].float().cpu().numpy() for name, a in acts.items()}
shapes = {name: tuple(a.shape[1:]) for name, a in acts.items()}
steps = [*maps, "Grad-CAM"]
notes = ["fine detail", "edges", "textures", "parts of bones", "whole-image context", "what raised the score"]

fig = plt.figure(figsize=(10, 5.6))
ax_in = fig.add_axes([0.02, 0.08, 0.36, 0.80])
ax_in.imshow(arr[row_of[detected.case_id]], cmap="gray", vmin=0, vmax=255)
for bx, by, bw, bh in boxes[detected.case_id]:
    ax_in.add_patch(Rectangle((bx, by), bw, bh, fill=False, ec="lime", lw=2))
ax_in.set_title("Input, 512 x 512 (green: annotated fracture)", fontsize=10)
ax_in.axis("off")
panels = []
for j, name in enumerate(steps):
    ax = fig.add_axes([0.42 + (j % 3) * 0.19, 0.58 - (j // 3) * 0.40, 0.18, 0.34])
    ax.axis("off")
    if name == "Grad-CAM":
        ax.imshow(arr[row_of[detected.case_id]], cmap="gray", vmin=0, vmax=255)
        im = ax.imshow(cam, cmap="jet", alpha=0.4, vmin=0, vmax=1)
        title = "Grad-CAM"
    else:
        m = maps[name]
        im = ax.imshow(m / m.max(), cmap="viridis")
        c, h, w = shapes[name]
        title = f"{name}: {h}x{w}, {c} ch"
    ax.set_title(f"{title}\n{notes[j]}", fontsize=9, fontweight="normal")
    im.set_alpha(0.0)
    panels.append(im)
result = fig.text(
    0.42, 0.06, f"Score {detected.score:+.1f}: above 0, flagged as a fracture", fontsize=11, weight="bold", alpha=0
)
LEAD, EACH, TAIL = 10, 12, 35


def update1(f):
    for j, im in enumerate(panels):
        t = np.clip((f - LEAD - j * EACH) / EACH, 0, 1)
        im.set_alpha(t * (0.4 if j == len(panels) - 1 else 1.0))
    result.set_alpha(float(np.clip((f - LEAD - len(panels) * EACH) / 8, 0, 1)))


save(fig, update1, LEAD + EACH * len(panels) + TAIL, "anim_layers.mp4")
del model
torch.cuda.empty_cache()

# 2. Grad-CAM on a detected and a missed small fracture
cases = [(small_hit, "Small fracture, detected"), (small_miss, "Small fracture, missed")]
cams = []
for r, _ in cases:
    model = load_fold_model(r.fold)
    cams.append(gradcam(model, to_batch(arr, [row_of[r.case_id]], False, 512))[0])
    del model
fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
overlays, rects, labels = [], [], []
for ax, (r, title), cm in zip(axes, cases, cams):
    ax.imshow(arr[row_of[r.case_id]], cmap="gray", vmin=0, vmax=255)
    overlays.append(ax.imshow(cm, cmap="jet", alpha=0.0, vmin=0, vmax=1))
    group = []
    for bx, by, bw, bh in boxes[r.case_id]:
        rect = Rectangle((bx, by), bw, bh, fill=False, ec="lime", lw=3, alpha=0)
        ax.add_patch(rect)
        group.append(rect)
    rects.append(group)
    ax.set_title(title, fontsize=12, color=BLUE if r.score >= 0 else "#c0392b")
    labels.append(
        ax.text(
            0.02,
            0.03,
            f"score {r.score:+.1f}",
            transform=ax.transAxes,
            color="white",
            fontsize=13,
            weight="bold",
            alpha=0,
        )
    )
    ax.axis("off")
fig.tight_layout()


def update2(f):
    for ov, group, lab in zip(overlays, rects, labels):
        for rect in group:
            rect.set_alpha(float(f >= 12))
        ov.set_alpha(0.45 * float(np.clip((f - 25) / 20, 0, 1)))
        lab.set_alpha(float(f >= 50))


save(fig, update2, 90, "anim_gradcam_small.mp4")

# 3. the pipeline
stages = [
    ("Input", "512 x 512\n1 channel", 1.00),
    ("Stem", "128 x 128\n96 ch", 0.85),
    ("Stage 1\n3 blocks", "128 x 128\n96 ch", 0.85),
    ("Stage 2\n3 blocks", "64 x 64\n192 ch", 0.65),
    ("Stage 3\n9 blocks", "32 x 32\n384 ch", 0.5),
    ("Stage 4\n3 blocks", "16 x 16\n768 ch", 0.38),
    ("Pool +\nlinear", "1 logit", 0.2),
    ("Minus\nthreshold", "90% specificity\non validation", 0.2),
    ("Flag?", "score >= 0", 0.2),
]
fig, ax = plt.subplots(figsize=(12.8, 7.2))
ax.set_xlim(0, len(stages))
ax.set_ylim(-1.4, 1.9)
ax.axis("off")
ax.set_title("From X-ray to flag: ConvNeXt-Tiny", loc="left", fontsize=20, weight="bold")
blocks, texts = [], []
for i, (name, detail, h) in enumerate(stages):
    fc = ORANGE if i >= 6 else BLUE
    rect = Rectangle((i + 0.1, -h * 0.9), 0.8, h * 1.8, fc=fc, ec="none", alpha=0.0)
    ax.add_patch(rect)
    t1 = ax.text(i + 0.5, 1.15, name, ha="center", va="bottom", fontsize=12, weight="bold", alpha=0)
    t2 = ax.text(i + 0.5, -1.05, detail, ha="center", va="top", fontsize=10, color=GREY, alpha=0)
    blocks.append(rect)
    texts.append((t1, t2))
arrows = []
for i in range(len(stages) - 1):
    ar = FancyArrowPatch((i + 0.92, 0), (i + 1.08, 0), arrowstyle="-|>", mutation_scale=14, color="k", alpha=0)
    ax.add_patch(ar)
    arrows.append(ar)
note = ax.text(
    0,
    -1.38,
    "Each fold model sees an X-ray it was not trained on. The threshold is fixed on validation scores, never on test data.",
    fontsize=10,
)
EACH3 = 9


def update3(f):
    for i, (rect, (t1, t2)) in enumerate(zip(blocks, texts)):
        a = float(np.clip((f - 6 - i * EACH3) / EACH3, 0, 1))
        rect.set_alpha(0.85 * a)
        t1.set_alpha(a)
        t2.set_alpha(a)
        if i < len(arrows):
            arrows[i].set_alpha(a)


save(fig, update3, 6 + EACH3 * len(stages) + 40, "anim_pipeline.mp4")
