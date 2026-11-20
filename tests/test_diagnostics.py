"""Solver convergence diagnostics are exposed on AtmosphericParameters.

None means "unknown", never 0. MOOG and the atmosphere grid are faked.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from species.atmosphere import (
    AtmosphereFitter, AtmosphericParameters, _ConvergenceState,
)
from species.config import Tolerance
from species.io.results import AnalysisResult
from species.moog.parser import AbfindResult
from species.moog.wrapper import MOOGError

NEW = ("final_residuals", "n_iters", "hit_iter_cap", "last_moog_status", "n_fe_lines_input")


def _abfind(ep=0.0, dif=0.0, rw=0.0, fe_i=7.5, n_failed=0):
    return AbfindResult(
        fe_i_abundance=fe_i, fe_ii_abundance=fe_i - dif, ep_slope=ep,
        rw_slope=rw, fe_i_fe_ii_diff=dif, n_failed=n_failed, n_fe_i=90, n_fe_ii=8,
    )


class _Session:
    def __init__(self, fn):
        self.fn, self.model_path, self.calls = fn, Path("m.atm"), 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def run_abfind(self, read_mode="linearregression"):
        self.calls += 1
        return self.fn(self.calls)


class _Grid:
    ok = True

    def interpolate_and_write(self, *a, **k):
        return self.ok


def _fitter(fn, minimization="broyden"):
    cfg = SimpleNamespace(
        tolerance=Tolerance(), fe_solar=7.5, read_mode="linearregression",
        minimization=minimization,
    )
    session = _Session(fn)
    moog = SimpleNamespace(open_session=lambda path: session)
    return AtmosphereFitter(cfg, moog, _Grid()), session


def test_defaults_are_none_not_zero():
    p = AtmosphericParameters()
    for name in NEW:
        assert getattr(p, name) is None, name


def test_to_dict_carries_fields():
    p = AtmosphericParameters(
        final_residuals=(0.1, -0.2, 0.3, 0.4), n_iters=7, hit_iter_cap=False,
        last_moog_status="ok", n_fe_lines_input=104,
    )
    d = AnalysisResult(star_name="x", params=p).to_dict()
    assert d["final_residuals"] == [0.1, -0.2, 0.3, 0.4]
    assert d["n_iters"] == 7 and d["hit_iter_cap"] is False
    assert d["last_moog_status"] == "ok" and d["n_fe_lines_input"] == 104
    d0 = AnalysisResult(star_name="x", params=AtmosphericParameters()).to_dict()
    assert all(d0[k] is None for k in NEW)


def test_broyden_converged_on_first_evaluation():
    fitter, session = _fitter(lambda n: _abfind(fe_i=7.5 + 0.0))
    p = fitter.fit(Path("l.txt"), initial_params=(0.0, 5500.0, 4.4, 1.2), method="broyden")
    assert p.converged
    assert p.final_residuals == pytest.approx((0.0, 0.0, 0.0, 0.0))
    assert p.n_iters == 1 == session.calls
    assert p.hit_iter_cap is False
    assert p.last_moog_status == "ok"


def test_broyden_hits_iteration_cap_and_reports_last_step_residuals():
    # Residuals never shrink: constant nonzero EP slope and ionization gap.
    # A residual that wanders with the call count keeps the Jacobian regular
    # but never lets the norm fall under the tolerance.
    def wander(n):
        return _abfind(ep=0.05 + 0.01 * ((n * 7) % 5), dif=0.1 + 0.01 * ((n * 3) % 4),
                       rw=-0.02 + 0.005 * ((n * 5) % 3))

    fitter, session = _fitter(wander)
    p = fitter.fit(Path("l.txt"), initial_params=(0.0, 5500.0, 4.4, 1.2), method="broyden")
    assert not p.converged
    assert p.n_iters == session.calls
    assert p.hit_iter_cap is True
    assert p.final_residuals is not None and len(p.final_residuals) == 4
    assert p.last_moog_status == "ok"


def test_broyden_singular_jacobian_is_not_an_iteration_cap():
    """Constant residuals give a zero Jacobian: stops early, cap is False."""
    fitter, session = _fitter(lambda n: _abfind(ep=0.05, dif=0.1, rw=-0.02))
    p = fitter.fit(Path("l.txt"), initial_params=(0.0, 5500.0, 4.4, 1.2), method="broyden")
    assert not p.converged
    assert p.hit_iter_cap is False
    assert p.final_residuals[0] == pytest.approx(0.05)


def test_broyden_moog_failed_iterations_status():
    fitter, _ = _fitter(lambda n: _abfind(ep=0.05, dif=0.1, n_failed=2))
    p = fitter.fit(Path("l.txt"), initial_params=(0.0, 5500.0, 4.4, 1.2), method="broyden")
    assert p.last_moog_status == "failed_iterations"


def test_broyden_moog_error_status_and_unknown_residuals():
    def boom(n):
        raise MOOGError("no output")
    fitter, _ = _fitter(boom)
    p = fitter.fit(Path("l.txt"), initial_params=(0.0, 5500.0, 4.4, 1.2), method="broyden")
    assert not p.converged
    assert p.last_moog_status == "moog_error"
    assert p.final_residuals is None  # never measured: unknown, not zero


def test_per_parameter_converged_first_pass():
    fitter, session = _fitter(lambda n: _abfind(), "per_parameter")
    p = fitter.fit(Path("l.txt"), initial_params=(0.0, 5500.0, 4.4, 1.2), method="per_parameter")
    assert p.converged and p.method == "per_parameter"
    assert p.final_residuals == pytest.approx((0.0, 0.0, 0.0, 0.0))
    assert p.n_iters == 1
    assert p.hit_iter_cap is False
    assert p.last_moog_status == "ok"


def test_per_parameter_failure_keeps_diagnostics():
    fitter, session = _fitter(lambda n: _abfind(ep=0.2, dif=0.3, rw=0.1), "per_parameter")
    fitter.grid.ok = True
    state = _ConvergenceState((0.0, 5500.0, 4.4, 1.2), [], Tolerance())
    state.n_it = state.n_repeat  # as if the repeat cap tripped
    state.record_moog_output(7.6 - 7.5, 0.2, 0.3, 0.1, 0)
    d = AtmosphereFitter._failed_result(state)
    assert not d.converged
    assert d.final_residuals == pytest.approx((0.2, 0.3, 0.1, 0.1))
    assert d.hit_iter_cap is True


def test_held_parameter_residual_is_reported_unmasked():
    state = _ConvergenceState((0.0, 5500.0, 4.4, 1.2), ["gravity"], Tolerance())
    state.record_moog_output(0.0, 0.01, 0.4, 0.02, 0)
    assert state.moog[2] == 0.0  # solver masks it
    assert state.final_residuals[1] == 0.4  # report does not


def test_simplex_exposes_counts():
    fitter, session = _fitter(lambda n: _abfind(ep=0.0, dif=0.0, rw=0.0), "downhill_simplex")
    fitter.moog.run_abfind = lambda model, ll, read_mode=None: session.fn(1)
    p = fitter.fit(Path("l.txt"), initial_params=(0.0, 5500.0, 4.4, 1.2), method="downhill_simplex")
    assert p.n_iters and p.n_iters > 0
    assert p.last_moog_status == "ok"
    assert p.final_residuals is not None
    assert p.hit_iter_cap is False
