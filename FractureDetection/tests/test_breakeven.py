import numpy as np

from fracai.breakeven import breakeven_grid, breakeven_price, incremental_cost


def test_identical_arms_zero_price_costs_nothing():
    arm = (0.9, 0.85)
    assert incremental_cost(1000, 0.2, arm, arm, c_miss=500, c_fp=50, price=0.0) == 0


def test_price_adds_linearly():
    arm = (0.9, 0.85)
    assert incremental_cost(1000, 0.2, arm, arm, 500, 50, price=2.0) == 2000


def test_better_sensitivity_gives_positive_breakeven_price():
    assert breakeven_price(0.2, (0.85, 0.9), (0.95, 0.9), c_miss=500, c_fp=50) > 0
    assert breakeven_price(0.2, (0.95, 0.9), (0.85, 0.9), c_miss=500, c_fp=50) < 0


def test_breakeven_is_the_price_where_cost_is_zero():
    u, a = (0.85, 0.9), (0.95, 0.88)
    p = breakeven_price(0.2, u, a, 500, 50)
    assert abs(incremental_cost(1000, 0.2, u, a, 500, 50, p)) < 1e-9


def test_monotone_in_cost_of_a_miss_and_baseline_miss_rate():
    g = breakeven_grid(0.2, 0.9, (0.95, 0.9), [0.05, 0.1, 0.2], [100, 500, 1000], c_fp=50)
    assert (np.diff(g, axis=0) >= 0).all() and (np.diff(g, axis=1) >= 0).all()
