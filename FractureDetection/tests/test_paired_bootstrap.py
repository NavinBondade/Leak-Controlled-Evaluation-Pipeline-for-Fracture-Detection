import numpy as np
import pandas as pd

from fracai.metrics import group_bootstrap

# paired comparisons (scripts/analysis/compare_box.py) pass one stat_fn that scores both arms on the same draw


def _two_arms(n_groups=1500, seed=0):
    rng = np.random.default_rng(seed)
    size = rng.integers(1, 5, n_groups)  # 1 to 4 images per patient
    d = pd.DataFrame({"group": np.repeat(np.arange(n_groups), size)})
    d["s0"] = (rng.random(len(d)) < 0.5).astype(float)
    d["s1"] = (rng.random(len(d)) < 0.6).astype(float)  # true difference in flag rate: 0.1
    return d, dict(enumerate(size))


def test_identical_arms_give_zero_width_interval():
    d, _ = _two_arms()
    d["s1"] = d.s0
    # if the arms were scored on different draws this interval would have width
    assert group_bootstrap(d, lambda x: x.s1.mean() - x.s0.mean(), "group", 200) == (0, 0, 0)
    _, lo, hi = group_bootstrap(d, lambda x: x.s0.mean(), "group", 200)
    assert hi - lo > 0.02


def test_interval_for_known_difference_contains_it():
    d, _ = _two_arms()
    p, lo, hi = group_bootstrap(d, lambda x: x.s1.mean() - x.s0.mean(), "group", 1000)
    assert lo < 0.1 < hi and lo < p < hi
    assert hi - lo < 0.06


def test_resamples_whole_groups():
    d, size = _two_arms(300)
    seen = []

    def stat(x):
        n = x.groupby("group").size()
        seen.append((n % n.index.map(size) == 0).all())
        return x.s1.mean() - x.s0.mean()

    group_bootstrap(d, stat, "group", 50)
    assert len(seen) == 51 and all(seen)  # point estimate plus 50 draws
