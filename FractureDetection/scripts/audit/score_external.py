"""Score GRAZPEDWRI-DX with the five FracAtlas fold models."""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fracai.metrics import shift_and_average
from fracai.train import build_model, dev, load_cache, predict

OUT = Path("results/ext")
OUT.mkdir(parents=True, exist_ok=True)
meta = pd.read_csv("results/graz_meta.csv")
arr = load_cache("graz", 512)
val = pd.read_csv("results/val_fracatlas.csv")
idx = np.arange(len(meta))

logits = []
for k in range(5):
    f = OUT / f"graz_logit_model{k}.npy"
    if not f.exists():
        model = build_model("convnext_tiny").to(dev)
        model.load_state_dict(torch.load(f"results/model_fracatlas_fold{k}.pt", map_location=dev))
        np.save(f, predict(model, arr, idx, 512))
        del model
        torch.cuda.empty_cache()
        print(f"model {k} scored", flush=True)
    logits.append(np.load(f))

meta["score"], shifted = shift_and_average(logits, val, 0.90)
for k in range(5):
    meta[f"shifted_{k}"] = shifted[k]
meta.to_csv("results/ext_graz_scores.csv", index=False)
print("wrote results/ext_graz_scores.csv", len(meta))
