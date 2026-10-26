"""Five-fold cross-validated training and out-of-fold scoring."""

import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .data import DATA

dev = "cuda" if torch.cuda.is_available() else "cpu"


def load_cache(name: str, size: int):
    return np.load(DATA / "cache" / f"{name}_{size}.npy", mmap_mode="r")


def to_batch(arr, idx, train: bool, out_size: int, mask=None):
    """mask: optional (B,1,m,m) float box mask; it gets the same random affine as the image and is returned with it."""
    x = torch.from_numpy(np.stack([arr[i] for i in idx])).to(dev, non_blocking=True).float().div_(255).unsqueeze(1)
    if train:  # light augmentation: small affine and brightness/contrast, no flips
        b = x.shape[0]
        ang = (torch.rand(b, device=dev) - 0.5) * 2 * math.radians(10)
        sc = 1 + (torch.rand(b, device=dev) - 0.5) * 0.2
        tx, ty = [(torch.rand(b, device=dev) - 0.5) * 0.1 for _ in range(2)]
        th = torch.stack(
            [
                torch.stack([sc * torch.cos(ang), -sc * torch.sin(ang), tx], 1),
                torch.stack([sc * torch.sin(ang), sc * torch.cos(ang), ty], 1),
            ],
            1,
        )
        x = F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), padding_mode="zeros", align_corners=False)
        if mask is not None:
            mask = F.grid_sample(
                mask, F.affine_grid(th, mask.shape, align_corners=False), padding_mode="zeros", align_corners=False
            )
        x = (
            (x - 0.5) * (1 + (torch.rand(b, 1, 1, 1, device=dev) - 0.5) * 0.3)
            + 0.5
            + (torch.rand(b, 1, 1, 1, device=dev) - 0.5) * 0.1
        )
        x = x.clamp(0, 1)
    if out_size != x.shape[-1]:
        x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
    x = (x - 0.45) / 0.25
    return x if mask is None else (x, mask)


@torch.no_grad()
def predict(model, arr, idx, out_size, bs=16):
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
            out.append(model(to_batch(arr, idx[i : i + bs], False, out_size)).float().squeeze(1).cpu())
    return torch.cat(out).numpy()


WEIGHTS = Path(__file__).resolve().parents[1] / "weights"
CKPT = Path(__file__).resolve().parents[1] / "results" / "ckpt"


def build_model(arch):
    """Offline: random-init model, backbone filled from weights/<arch>.pth (made once by scripts/data/fetch_weights.py)."""
    model = timm.create_model(arch, pretrained=False, num_classes=1, in_chans=1)
    f = WEIGHTS / f"{arch}.pth"
    if not f.exists():
        raise FileNotFoundError(
            f"{f} missing. Run scripts/data/fetch_weights.py once while online, or pass an existing arch."
        )
    missing, unexpected = model.load_state_dict(torch.load(f, map_location="cpu"), strict=False)
    assert not unexpected and all(m.startswith(("head.fc", "fc.", "classifier", "head.")) for m in missing), (
        missing,
        unexpected,
    )
    return model


class FineHead(torch.nn.Module):
    """Light feature-pyramid head: stride-16 (stage 2) + upsampled stride-32 (stage 3) features -> 32x32 fracture map (16 px cells).
    Training-time only, dropped before saving. The 16x16 map of the plain aux head (32 px cells) is too coarse for small fractures."""

    def __init__(self, model):
        super().__init__()
        c2, c3 = model.stages[2].blocks[-1].conv_dw.out_channels, model.num_features
        self.p2, self.p3 = torch.nn.Conv2d(c2, 128, 1), torch.nn.Conv2d(c3, 128, 1)
        self.out = torch.nn.Conv2d(128, 1, 3, padding=1)

    def forward(self, fs):
        p = self.p2(fs[2]) + F.interpolate(self.p3(fs[3]), size=fs[2].shape[-2:], mode="nearest")
        return self.out(F.relu(p))


def _save_atomic(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)  # atomic write


def train_fold(
    arr,
    df,
    k,
    arch="convnext_tiny",
    epochs=12,
    bs=16,
    lr=1e-4,
    out_size=512,
    seed=42,
    patience=4,
    log=print,
    ckpt_name="run",
    aux=None,
    small=None,
    small_rep=1,
):
    """aux (optional box supervision, training-time only): dict(masks=(N,m,m) uint8 box masks, known=(N,) bool where the
    localisation target is trusted - every negative plus the positives whose box is used, sw=(N,) per-image weight, lam=float).
    An auxiliary 1x1 head predicts a coarse fracture map from the last-stage features; it is dropped before saving/inference."""
    torch.manual_seed(seed + k)
    rng = np.random.default_rng(seed + k)
    tr = df.index[~df.fold.isin([k, (k + 1) % 5])].values
    va = df.index[df.fold == (k + 1) % 5].values
    te = df.index[df.fold == k].values
    extra = tr[small[tr]] if (small is not None and small_rep > 1) else tr[:0]
    pool = np.concatenate([tr] + [extra] * (small_rep - 1)) if len(extra) else tr
    model = build_model(arch)
    if aux is not None:
        model.aux = FineHead(model) if aux.get("kind") == "fine" else torch.nn.Conv2d(model.num_features, 1, 1)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    steps = epochs * math.ceil(len(pool) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    pos_w = torch.tensor((1 - df.y[tr].mean()) / df.y[tr].mean(), device=dev)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    y_all = torch.tensor(df.y.values, dtype=torch.float32)
    best, best_state, bad, start = -1, None, 0, 0
    ck = CKPT / f"{ckpt_name}_fold{k}.pt"
    if ck.exists():  # resume from the last finished epoch
        c = torch.load(ck, map_location="cpu", weights_only=False)  # load on CPU to save VRAM
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        sched.load_state_dict(c["sched"])
        scaler.load_state_dict(c["scaler"])
        rng.bit_generator.state = c["np_rng"]
        torch.set_rng_state(c["torch_rng"].cpu())
        torch.cuda.set_rng_state(c["cuda_rng"].cpu()) if dev == "cuda" else None
        best, best_state, bad, start = c["best"], c["best_state"], c["bad"], c["epoch"]
        del c
        log(f"fold {k}: resumed from checkpoint after epoch {start}")
    for ep in range(start, epochs):
        if bad >= patience:
            break
        model.train()
        t0, tot = time.time(), 0.0
        perm = rng.permutation(pool)
        for i in range(0, len(perm), bs):
            idx = np.sort(perm[i : i + bs])
            y = y_all[idx].to(dev)
            if aux is None:
                x = to_batch(arr, idx, True, out_size)
                with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                    loss = F.binary_cross_entropy_with_logits(model(x).float().squeeze(1), y, pos_weight=pos_w)
            else:
                m = torch.from_numpy(np.stack([aux["masks"][j] for j in idx])).to(dev).float().div_(255).unsqueeze(1)
                x, m = to_batch(arr, idx, True, out_size, m)
                with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                    if aux.get("kind") == "fine":
                        fs, f = [], model.stem(x)
                        for st in model.stages:
                            f = st(f)
                            fs.append(f)
                        amap = model.aux(fs).float()
                    else:
                        f = model.forward_features(x)
                        amap = model.aux(f).float()
                    loss = F.binary_cross_entropy_with_logits(
                        model.forward_head(f).float().squeeze(1), y, pos_weight=pos_w
                    )
                tgt = F.adaptive_max_pool2d(m, amap.shape[-2:])
                w = torch.from_numpy(aux["known"][idx].astype(np.float32) * aux["sw"][idx].astype(np.float32)).to(dev)
                bl = F.binary_cross_entropy_with_logits(
                    amap, tgt, reduction="none", pos_weight=torch.tensor(10.0, device=dev)
                ).mean((1, 2, 3))
                loss = loss + aux["lam"] * (bl * w).sum() / torch.from_numpy(aux["known"][idx].astype(np.float32)).to(
                    dev
                ).sum().clamp_min(1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(idx)
        auc = roc_auc_score(df.y[va], predict(model, arr, va, out_size))
        if dev == "cuda":
            torch.cuda.empty_cache()  # keep peak VRAM low
        log(f"fold {k} ep {ep + 1}/{epochs} loss {tot / len(pool):.4f} val_auc {auc:.4f} ({time.time() - t0:.0f}s)")
        if auc > best:
            best, bad = auc, 0
            best_state = {n: v.detach().clone() for n, v in model.state_dict().items()}
        else:
            bad += 1
        _save_atomic(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "scaler": scaler.state_dict(),
                "np_rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state() if dev == "cuda" else None,
                "best": best,
                "best_state": best_state,
                "bad": bad,
                "epoch": ep + 1,
            },
            ck,
        )
    model.load_state_dict(best_state)
    ck.unlink(missing_ok=True)
    return model, te, va, best


def run_cv(name, df, arr, folds=range(5), **kw):
    parts, vparts = [], []
    done = set()
    if Path(f"results/oof_{name}.csv").exists() and Path(f"results/val_{name}.csv").exists():  # keep finished folds
        o, v0 = pd.read_csv(f"results/oof_{name}.csv"), pd.read_csv(f"results/val_{name}.csv")
        done = set(o.fold.unique())
        parts, vparts = [o], [v0]
        print(f"resuming: folds already finished: {sorted(done)}", flush=True)
    for k in folds:
        if k in done:
            continue
        model, te, va, vauc = train_fold(arr, df, k, ckpt_name=name, log=lambda m: print(m, flush=True), **kw)
        s = df.loc[te].copy()
        s["score"] = predict(model, arr, te, kw.get("out_size", 512))
        s["val_auc"] = vauc
        parts.append(s)
        v = df.loc[va].copy()
        v["score"] = predict(model, arr, va, kw.get("out_size", 512))
        v["test_fold"] = k
        vparts.append(v)
        pd.concat(parts).to_csv(f"results/oof_{name}.csv", index=False)
        pd.concat(vparts).to_csv(f"results/val_{name}.csv", index=False)
        torch.save(
            {n: v for n, v in model.state_dict().items() if not n.startswith("aux.")},
            f"results/model_{name}_fold{k}.pt",
        )
        print(f"fold {k} test auc {roc_auc_score(s.y, s.score):.4f}", flush=True)
        del model
        torch.cuda.empty_cache()
    return pd.concat(parts)
