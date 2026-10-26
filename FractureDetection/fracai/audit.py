"""Subgroup and shortcut audit on a score table."""

import pandas as pd

from .metrics import group_bootstrap, safe_auc, sens_spec

MIN_N = 30  # subgroups below this (in either class) are flagged unreliable


def _stats(df, thr, n_boot, gcol):
    out = {
        "n": len(df),
        "pos": int(df.y.sum()),
        "neg": int((df.y == 0).sum()),
        "prevalence": df.y.mean() if len(df) else float("nan"),
    }
    for name, fn in [
        ("auc", lambda d: safe_auc(d.y, d.score)),
        ("sens", lambda d: sens_spec(d.y, d.score, thr)[0]),
        ("spec", lambda d: sens_spec(d.y, d.score, thr)[1]),
    ]:
        if out["pos"] and out["neg"]:
            p, lo, hi = group_bootstrap(df, fn, gcol, n_boot)
        else:
            p = lo = hi = float("nan")
        out[name], out[f"{name}_lo"], out[f"{name}_hi"] = p, lo, hi
    out["unreliable"] = out["pos"] < MIN_N or out["neg"] < MIN_N
    return out


def subgroup_table(df, cols, thr, gcol="group", n_boot=1000):
    rows = [{"subgroup": "ALL", "level": "", **_stats(df, thr, n_boot, gcol)}]
    for c in cols:
        for lvl, sub in df.groupby(c, dropna=False):
            rows.append({"subgroup": c, "level": lvl, **_stats(sub, thr, n_boot, gcol)})
    return pd.DataFrame(rows)


def hardware_shortcut(df, thr, gcol="group", n_boot=1000):
    """(a) all images, (b) hardware excluded, (c) hardware only, (d) 'model' that only reads the hardware flag."""
    flag = df.assign(score=df.hardware.astype(float))
    parts = {"all": df, "no_hardware": df[df.hardware == 0], "hardware_only": df[df.hardware == 1]}
    rows = [{"set": k, **_stats(v, thr, n_boot, gcol)} for k, v in parts.items()]
    rows.append({"set": "hardware-flag-only baseline (all)", **_stats(flag, 0.5, n_boot, gcol)})
    return pd.DataFrame(rows)


def positives_only_sensitivity(df, col, thr, gcol="group", n_boot=1000):
    """Sensitivity by a property that only positives have (fracture count, fracture size). No AUC: there are no negatives."""
    pos = df[df.y == 1]
    rows = []
    for lvl, sub in pos.groupby(col, observed=True):
        p, lo, hi = group_bootstrap(sub, lambda d: sens_spec(d.y, d.score, thr)[0], gcol, n_boot)
        rows.append(
            {
                "by": col,
                "level": lvl,
                "pos": len(sub),
                "sens": p,
                "sens_lo": lo,
                "sens_hi": hi,
                "unreliable": len(sub) < MIN_N,
            }
        )
    return pd.DataFrame(rows)


def size_shortcut(df, thr, gcol="group", n_boot=1000):
    """Does image size/format carry label information, and does the model's performance depend on it?
    Size-only baseline: score = prevalence of the image's size class (fit on the same data, so optimistic)."""
    prev = df.groupby("size_class").y.transform("mean")
    base = df.assign(score=prev)
    rows = [{"set": "size-class-only baseline (all)", **_stats(base, 0.5, n_boot, gcol)}]
    rows += [{"set": f"size_class={k}", **_stats(v, thr, n_boot, gcol)} for k, v in df.groupby("size_class")]
    return pd.DataFrame(rows)
