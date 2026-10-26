"""Site-parameterised discrete-event simulation of an ED fracture pathway, with calibration and validation."""

import heapq
from dataclasses import dataclass, field, replace

import numpy as np


@dataclass
class Site:
    rate_per_day: float = 49.4
    hourly: np.ndarray = field(default_factory=lambda: np.ones(24) / 24)
    work_start_h: float = 8.0
    work_hours: float = 9.0
    svc_mean_min: float = 9.0
    prev: float = 0.20  # assumed
    miss: float = 0.05  # assumed
    catch: float = 0.8  # assumed


def arrivals(site, days, rng):
    n = rng.poisson(site.rate_per_day * days)
    day = rng.integers(0, days, n)
    hour = rng.choice(24, n, p=site.hourly / site.hourly.sum())
    return np.sort(day * 1440 + hour * 60 + rng.uniform(0, 60, n))


def _next_work(t, s, e):
    d, m = divmod(t, 1440)
    if m < s:
        return d * 1440 + s
    return t if m < e else (d + 1) * 1440 + s


def reports(site, arr, prio, rng):
    """One reporter, priority queue (lower first), non-preemptive; returns completion time of each report."""
    s, e = site.work_start_h * 60, (site.work_start_h + site.work_hours) * 60
    n, out, free, i, wait, t, done = len(arr), np.zeros(len(arr)), 0.0, 0, [], 0.0, 0
    svc = rng.exponential(site.svc_mean_min, n)
    while done < n:
        while i < n and arr[i] <= t:
            heapq.heappush(wait, (prio[i], arr[i], i))
            i += 1
        if not wait:
            t = arr[i]
            continue
        start = _next_work(max(t, free), s, e)
        while i < n and arr[i] <= start:
            heapq.heappush(wait, (prio[i], arr[i], i))
            i += 1
        j = heapq.heappop(wait)[2]
        free = out[j] = start + svc[j]
        t = start
        done += 1
    return out


def delays(site, days, seed):
    """Report delays (minutes) with no AI: what a real site's timestamps show."""
    rng = np.random.default_rng(seed)
    a = arrivals(site, days, rng)
    return a, reports(site, a, np.ones(len(a)), rng) - a


def scenario(site, ai, alert, triage, days, seed):
    """ai = (sens, spec, sens_on_clinician_missed) or None."""
    rng = np.random.default_rng(seed)
    a = arrivals(site, days, rng)
    n = len(a)
    frac = rng.random(n) < site.prev
    edmiss = frac & (rng.random(n) < site.miss)
    u1, u2 = rng.random(n), rng.random(n)
    flag = np.zeros(n, bool)
    if ai:
        sens, spec, sens_m = ai
        flag = np.where(edmiss, u1 < sens_m, np.where(frac, u1 < sens, u1 < 1 - spec))
    caught = edmiss & flag & alert & (u2 < site.catch)
    rep = reports(site, a, np.where(flag & triage, 0, 1), np.random.default_rng(seed + 1))
    late = edmiss & ~caught
    d = (rep - a)[late] / 60
    return {
        "ed_missed": int(edmiss.sum()),
        "caught_at_ed": int(caught.sum()),
        "found_by_report": int(late.sum()),
        "median_delay_h": float(np.median(d)) if late.any() else np.nan,
        "p90_delay_h": float(np.percentile(d, 90)) if late.any() else np.nan,
        "undetected_hours_per_year": float(d.sum() * 365 / days),
        "extra_reviews_per_day": float((flag & alert).sum() / days),
        "false_alarms_per_day": float((flag & ~frac & alert).sum() / days),
        "reviews_of_already_diagnosed_per_day": float((flag & frac & ~edmiss & alert).sum() / days),
    }


def _summ(d_min, n_days):
    return {
        "per_day": len(d_min) / n_days,
        "median_h": float(np.median(d_min) / 60),
        "p90_h": float(np.percentile(d_min, 90) / 60),
    }


def calibrate(xray_min, report_min, seed=0):
    """Fit a Site from observed X-ray and report times (minutes from a common origin). Timestamps cannot identify prevalence or miss rate."""
    x, r = np.asarray(xray_min, float), np.asarray(report_min, float)
    days = int(np.ceil(x.max() / 1440))
    hourly = np.bincount((x % 1440 // 60).astype(int), minlength=24) / len(x)
    comp = np.bincount((r % 1440 // 60).astype(int), minlength=24) / len(r)
    active = np.where(comp >= 0.15 * comp.max())[0]  # relative cut-off ignores jobs that overrun the shift
    base = Site(
        rate_per_day=len(x) / days,
        hourly=hourly,
        work_start_h=float(active.min()),
        work_hours=float(active.max() - active.min() + 1),
    )
    obs = _summ(r - x, days)
    best = (9e9, None)
    for svc in np.arange(3, 25.01, 0.5):
        s = replace(base, svc_mean_min=float(svc))
        sims = [_summ(delays(s, 120, seed + k)[1], 120) for k in range(3)]
        err = sum(abs(np.mean([q[m] for q in sims]) - obs[m]) / obs[m] for m in ("median_h", "p90_h"))
        if err < best[0]:
            best = (err, s)
    return best[1], best[0]


def validate(site, xray_min, report_min, reps=10, seed=500):
    """Compare simulated and observed volume and report-delay summaries on a held-out period."""
    x, r = np.asarray(xray_min, float), np.asarray(report_min, float)
    days = int(np.ceil((x.max() - x.min()) / 1440))
    obs = _summ(r - x, days)
    sims = [_summ(delays(site, days, seed + k)[1], days) for k in range(reps)]
    return {
        m: (obs[m], float(np.mean([q[m] for q in sims])), float(np.mean([q[m] for q in sims]) / obs[m] - 1))
        for m in obs
    }
