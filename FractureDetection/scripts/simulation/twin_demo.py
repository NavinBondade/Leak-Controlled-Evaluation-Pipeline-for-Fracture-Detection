"""Calibrate and validate the ED simulation on synthetic site data."""

import numpy as np
import pandas as pd

from fracai.twin import Site, calibrate, delays, scenario, validate

TOL = {"per_day": 0.10, "median_h": 0.25, "p90_h": 0.25}
AI = (0.808, 0.897, 0.761)  # internal point: sens, spec, sens on small fractures
hourly = np.array([1, 1, 1, 1, 1, 2, 3, 5, 7, 8, 8, 8, 7, 7, 7, 8, 8, 8, 7, 6, 5, 3, 2, 1], float)
truth = Site(rate_per_day=55.0, hourly=hourly / hourly.sum(), work_start_h=8.0, work_hours=10.0, svc_mean_min=9.0)

x, d = delays(truth, 365, seed=7)
r = x + d
cut = 270 * 1440
fit, err = calibrate(x[x < cut], r[x < cut])
print(
    f"calibrated: {fit.rate_per_day:.1f}/day, reporter {fit.work_start_h:.0f}:00 for {fit.work_hours:.0f} h, service {fit.svc_mean_min:.1f} min "
    f"(truth 55.0/day, 8:00 for 10 h, 9.0 min; fit error {err:.3f})"
)
val = validate(fit, x[x >= cut] - cut, r[x >= cut] - cut)
ok = all(abs(v[2]) <= TOL[k] for k, v in val.items())
for k, (o, s, e) in val.items():
    print(f"validation {k}: observed {o:.2f}, simulated {s:.2f}, error {e:+.1%} (tolerance {TOL[k]:.0%})")
print("VALIDATION", "PASSED" if ok else "FAILED")

rows = []
for lab, alert, triage, ai in [
    ("no AI", False, False, None),
    ("alert", True, False, AI),
    ("triage", False, True, AI),
    ("alert + triage", True, True, AI),
]:
    for who, s in (("hidden truth", truth), ("calibrated twin", fit)):
        rs = pd.DataFrame([scenario(s, ai, alert, triage, 365, 900 + k) for k in range(10)])
        rows.append({"scenario": lab, "site": who, **{c: round(rs[c].mean(), 1) for c in rs.columns}})
t = pd.DataFrame(rows)
t["validation"] = "passed" if ok else "FAILED"
t.to_csv("results/twin_demo.csv", index=False)
pd.set_option("display.width", 220)
print(t.drop(columns="validation").to_string(index=False))
