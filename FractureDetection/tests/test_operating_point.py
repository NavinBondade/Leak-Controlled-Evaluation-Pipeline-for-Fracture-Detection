import numpy as np
import pandas as pd

from fracai.metrics import fold_thresholds, sens_spec, shift_and_average, thr_at_spec


def _cv_scores(n=2000, seed=0):
    """Toy out-of-fold and validation scores shaped like results/oof_*.csv and results/val_*.csv."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    fold = np.arange(n) % 5
    oof = pd.DataFrame({"y": y, "fold": fold, "score": rng.normal(y * 1.5, 1)})
    # each fold model sees the next fold as validation and has its own logit offset
    val = pd.DataFrame({"y": y, "test_fold": (fold - 1) % 5})
    val["score"] = rng.normal(y * 1.5, 1) + val.test_fold * 0.3
    return oof, val


def test_threshold_ignores_test_scores_and_other_folds():
    oof, val = _cv_scores()
    thr = fold_thresholds(val)
    assert sorted(thr) == list(range(5))
    oof.score = oof.score * 10 - 3  # test scores play no part
    val2 = val.copy()
    val2.loc[val2.test_fold != 2, "score"] += 5  # nor do other folds' validation rows
    assert fold_thresholds(val2)[2] == thr[2]
    for k, v in val.groupby("test_fold"):
        assert thr[k] == thr_at_spec(v.y, v.score, 0.90)


def test_shifted_validation_scores_put_operating_point_at_zero():
    _, val = _cv_scores()
    thr = fold_thresholds(val)
    for k, v in val.groupby("test_fold"):
        s = v.score - thr[k]
        assert thr_at_spec(v.y, s, 0.90) == 0.0
        assert abs(sens_spec(v.y, s, 0.0)[1] - 0.90) < 0.01


def test_external_shift_and_average():
    _, val = _cv_scores()
    thr = fold_thresholds(val)
    rng = np.random.default_rng(1)
    logits = [rng.normal(0, 1, 300) + k for k in range(5)]
    score, shifted = shift_and_average(logits, val, 0.90)
    for k in range(5):  # model k is shifted by its own validation threshold, not another fold's
        assert np.allclose(shifted[k], logits[k] - thr[k])
        assert ((shifted[k] >= 0) == (logits[k] >= thr[k])).all()
    assert np.allclose(score, np.mean([lg - thr[k] for k, lg in enumerate(logits)], axis=0))
    # the same logits from every model give the plain logit minus the mean threshold
    same, _ = shift_and_average([logits[0]] * 5, val, 0.90)
    assert np.allclose(same, logits[0] - np.mean(list(thr.values())))
