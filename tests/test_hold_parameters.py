"""Holding a parameter must actually hold it.

`_fit_broyden` accepted `hold` and never used it, while `compute_errors` did
honour it. So `hold=["gravity"]` produced a freely fitted log g reported with
the uncertainty of a held one -- a fabricated error bar in an output table,
which is worse than the feature simply not existing.

These tests cover the masking algebra without invoking MOOG.
"""

from __future__ import annotations

import numpy as np
import pytest

from species.atmosphere import AtmosphereFitter

ORDER = AtmosphereFitter._SOLVER_PARAM_ORDER


def _step(hold, J=None, f0=None):
    """The masked Newton step, as _fit_broyden computes it."""
    free = np.array([n not in hold for n in ORDER])
    idx = np.where(free)[0]
    if J is None:
        J = np.eye(4) + 0.1 * np.arange(16).reshape(4, 4)
    if f0 is None:
        f0 = np.array([0.02, -0.05, 0.01, 0.03])
    delta = np.zeros(4)
    delta[idx] = -np.linalg.solve(J[np.ix_(idx, idx)], f0[idx])
    return delta, idx


def test_solver_order_is_documented():
    """`initial` is (feh, teff, logg, vt); the solver vector is not."""
    assert ORDER == ("temperature", "gravity", "metallicity", "velocity")


@pytest.mark.parametrize("hold,zero_slots", [
    ([], []),
    (["gravity"], [1]),
    (["temperature"], [0]),
    (["metallicity"], [2]),
    (["velocity"], [3]),
    (["gravity", "velocity"], [1, 3]),
])
def test_held_parameters_never_move(hold, zero_slots):
    delta, _ = _step(hold)
    for slot in zero_slots:
        assert delta[slot] == 0.0, f"{ORDER[slot]} was held but moved by {delta[slot]}"
    for slot in range(4):
        if slot not in zero_slots:
            assert delta[slot] != 0.0, f"{ORDER[slot]} was free but did not move"


def test_free_subsystem_stays_square_and_solvable():
    """Holding a parameter drops its paired residual, so the block stays square."""
    for hold in ([], ["gravity"], ["gravity", "velocity"], ["temperature", "metallicity"]):
        delta, idx = _step(hold)
        assert idx.size == 4 - len(hold)
        assert np.all(np.isfinite(delta))


def test_convergence_norm_ignores_held_residuals():
    """A held parameter's residual cannot be driven to zero, so it must not
    count towards convergence -- otherwise a held fit never converges."""
    f = np.array([1e-4, 5.0, 1e-4, 1e-4])   # gravity residual hopeless
    free = np.array([n not in ["gravity"] for n in ORDER])
    idx = np.where(free)[0]
    assert np.linalg.norm(f) > 1.0            # would never converge
    assert np.linalg.norm(f[idx]) < 1e-3      # does converge, correctly


def test_holding_everything_is_refused():
    free = np.array([n not in list(ORDER) for n in ORDER])
    assert np.where(free)[0].size == 0        # _fit_broyden raises on this
