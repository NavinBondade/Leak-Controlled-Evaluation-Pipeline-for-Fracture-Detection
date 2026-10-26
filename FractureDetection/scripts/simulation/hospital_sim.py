"""Illustrative ED fracture-pathway simulation using the NHS volumes in docs/NHS_DATA.md."""

import numpy as np
import pandas as pd

from fracai.twin import Site, scenario

TYPE1_PER_MONTH, N_ED, XRAY_SHARE, WORK_H, REPS, DAYS = 1_503_149, 120, 0.12, 9, 10, 365
RATE = TYPE1_PER_MONTH / N_ED / 30.4 * XRAY_SHARE
HOURLY = np.array(
    [1, 1, 1, 1, 1, 2, 3, 5, 7, 8, 8, 8, 7, 7, 7, 8, 8, 8, 7, 6, 5, 3, 2, 1], float
)  # assumed daytime-peaked shape
AI = {
    "AI internal point": (0.808, 0.897, 0.761),
    "AI external first-visit point": (0.540, 0.790, 0.540 * 0.761 / 0.808),
}

rows = []
for miss in (0.02, 0.05, 0.10):
    for util in (0.75, 0.95):
        site = Site(
            rate_per_day=RATE,
            hourly=HOURLY / HOURLY.sum(),
            work_start_h=8.0,
            work_hours=WORK_H,
            svc_mean_min=util * WORK_H * 60 / RATE,
            miss=miss,
        )
        scen = [("no AI", None, False, False)]
        for lab, ai in AI.items():
            scen += [
                (lab + ": alert", ai, True, False),
                (lab + ": triage", ai, False, True),
                (lab + ": alert + triage", ai, True, True),
            ]
        for name, ai, alert, triage in scen:
            rs = pd.DataFrame([scenario(site, ai, alert, triage, DAYS, 1000 + r) for r in range(REPS)])
            rows.append(
                {
                    "usual_miss_rate": miss,
                    "reporter_utilisation": util,
                    "scenario": name,
                    **{k: round(rs[k].mean(), 1) for k in rs.columns},
                }
            )
t = pd.DataFrame(rows)
t.to_csv("results/hospital_sim.csv", index=False)
pd.set_option("display.width", 260, "display.max_columns", 30)
print(f"X-rays/day per ED: {RATE:.1f}")
print(t[t.usual_miss_rate == 0.05].drop(columns=["usual_miss_rate"]).to_string(index=False))
