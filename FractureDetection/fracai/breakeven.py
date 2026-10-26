"""Break-even price per scan for AI-assisted fracture reading."""

import numpy as np


def outcomes(n, prev, sens, spec):
    return {"fn": n * prev * (1 - sens), "fp": n * (1 - prev) * (1 - spec)}


def incremental_cost(n, prev, usual, assisted, c_miss, c_fp, price):
    """usual/assisted = (sens, spec). Positive = AI-assisted care costs more."""
    u, a = outcomes(n, prev, *usual), outcomes(n, prev, *assisted)
    saved = (u["fn"] - a["fn"]) * c_miss + (u["fp"] - a["fp"]) * c_fp
    return n * price - saved


def breakeven_price(prev, usual, assisted, c_miss, c_fp):
    """Highest price per scan at which incremental cost <= 0 (negative = AI cannot pay for itself)."""
    return -incremental_cost(1, prev, usual, assisted, c_miss, c_fp, 0.0)


def breakeven_grid(prev, usual_spec, assisted, miss_rates, c_miss_values, c_fp, assisted_gain_is_relative=True):
    """Break-even price over (baseline miss rate x cost per missed fracture).

    assisted = (sens_a, spec_a). Usual-care sensitivity is 1 - miss_rate on the grid; usual specificity is fixed.
    """
    g = np.zeros((len(miss_rates), len(c_miss_values)))
    for i, m in enumerate(miss_rates):
        for j, c in enumerate(c_miss_values):
            g[i, j] = breakeven_price(prev, (1 - m, usual_spec), assisted, c, c_fp)
    return g
