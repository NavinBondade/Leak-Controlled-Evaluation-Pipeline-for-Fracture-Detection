import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fracai.audit import hardware_shortcut, subgroup_table
from fracai.metrics import group_bootstrap, safe_auc, sens_spec, thr_at_spec


def _scores(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    return pd.DataFrame(
        {
            "y": y,
            "score": rng.normal(y * 1.5, 1),
            "group": np.arange(n) // 2,
            "hardware": (rng.random(n) < 0.04).astype(int),
            "bodypart": rng.choice(["a", "b"], n),
        }
    )


def test_auc_matches_sklearn():
    d = _scores()
    assert safe_auc(d.y, d.score) == roc_auc_score(d.y, d.score)


def test_sens_spec_hand_computed():
    assert sens_spec([1, 1, 0, 0], [0.9, 0.2, 0.1, 0.6], 0.5) == (0.5, 0.5)


def test_threshold_gives_requested_specificity():
    d = _scores(5000)
    t = thr_at_spec(d.y, d.score, 0.9)
    assert abs(sens_spec(d.y, d.score, t)[1] - 0.9) < 0.01


def test_bootstrap_ci_brackets_point_and_is_wider_for_small_samples():
    d = _scores(600)
    p, lo, hi = group_bootstrap(d, lambda x: safe_auc(x.y, x.score), "group", 300)
    assert lo < p < hi
    _, lo2, hi2 = group_bootstrap(d.iloc[:100], lambda x: safe_auc(x.y, x.score), "group", 300)
    assert (hi2 - lo2) > (hi - lo)


def test_audit_tables_run_and_flag_small_groups():
    d = _scores()
    t = subgroup_table(d, ["bodypart"], 0.5, n_boot=50)
    assert {"ALL", "bodypart"} <= set(t.subgroup)
    h = hardware_shortcut(d, 0.5, n_boot=50)
    assert h.loc[h.set == "hardware_only", "unreliable"].item()  # ~60 images -> few positives
