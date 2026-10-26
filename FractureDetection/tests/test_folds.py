import numpy as np
import pandas as pd
import pytest

from fracai.train import fold_split


def _folds(n=500, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"y": rng.integers(0, 2, n), "fold": rng.integers(0, 5, n)})
    return df.set_index(pd.Index(rng.permutation(n) + 1000))  # labels that are not 0..n-1


@pytest.mark.parametrize("k", range(5))
def test_fold_k_tests_on_k_and_validates_on_the_next(k):
    df = _folds()
    tr, va, te = fold_split(df, k)
    assert set(df.loc[te].fold) == {k}
    assert set(df.loc[va].fold) == {(k + 1) % 5}
    assert set(df.loc[tr].fold) == set(range(5)) - {k, (k + 1) % 5}
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
    assert len(tr) + len(va) + len(te) == len(df)


def test_every_row_is_tested_once_and_validated_once():
    df = _folds()
    splits = [fold_split(df, k) for k in range(5)]
    tested = np.concatenate([te for _, _, te in splits])
    validated = np.concatenate([va for _, va, _ in splits])
    assert sorted(tested) == sorted(df.index) == sorted(validated)
