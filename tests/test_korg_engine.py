"""The Korg + MARCS engine.

Julia is optional, and CI will not have it, so these tests cover the contract
rather than the science: availability reporting, the argument mapping between
SPECIES's conventions and Korg's, and the parsing of the driver's JSON. The
driver itself is exercised only when SPECIES_KORG_PROJECT is set.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from species.atmosphere import AtmosphericParameters
from species.korg import KorgFitter, KorgUnavailable
from species.korg.fitter import _DRIVER, _to_result


def test_driver_script_ships_with_the_package():
    assert _DRIVER.exists(), f"missing driver at {_DRIVER}"
    assert _DRIVER.suffix == ".jl"


def test_unavailable_without_a_project(monkeypatch):
    monkeypatch.delenv("SPECIES_KORG_PROJECT", raising=False)
    ok, why = KorgFitter().available()
    assert not ok
    assert "project" in why.lower() or "julia not found" in why.lower()


def test_unavailable_when_julia_is_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("SPECIES_KORG_PROJECT", str(tmp_path))
    monkeypatch.setenv("SPECIES_JULIA_BINARY", "definitely-not-a-real-julia")
    ok, why = KorgFitter().available()
    assert not ok
    assert "julia not found" in why


def test_fit_raises_rather_than_returning_zeros(monkeypatch):
    """An unavailable engine must not look like a failed fit."""
    monkeypatch.delenv("SPECIES_KORG_PROJECT", raising=False)
    with pytest.raises(KorgUnavailable):
        KorgFitter().fit(Path("/nonexistent/linelist.txt"))


def test_unknown_hold_name_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("SPECIES_KORG_PROJECT", str(tmp_path))
    (tmp_path / "Project.toml").write_text("name = \"x\"\n")
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/julia")
    with pytest.raises(ValueError, match="unknown hold parameter"):
        KorgFitter().fit(Path("x.txt"), hold=["pressure"])


# --- parsing the driver's output ------------------------------------------

def _payload(**over):
    base = {
        "status": "ok", "retcode": "ok", "converged": True, "railed": [],
        "teff": 5731.2, "logg": 4.41, "feh": -0.03, "vt": 1.08,
        "n_lines": 169, "n_fe_i": 148, "n_fe_ii": 21,
        "covariance": {
            "order": ["teff", "logg", "vt", "feh"],
            "sigma": [31.0, 0.05, 0.04, 0.02],
            "matrix": [[961.0, 0, 0, 0], [0, 0.0025, 0, 0],
                       [0, 0, 0.0016, 0], [0, 0, 0, 0.0004]],
            "correlation": [[1.0, 0.31, -0.06, 0.87],
                            [0.31, 1.0, 0.02, 0.28],
                            [-0.06, 0.02, 1.0, -0.10],
                            [0.87, 0.28, -0.10, 1.0]],
            "fixed": [], "sigma_A": 0.07, "n_neutral": 148, "n_ionised": 21,
            "korg_sigma": [9.7, 0.04, 0.04, 0.02],
        },
    }
    base.update(over)
    return base


def test_parses_a_successful_solve():
    r = _to_result(_payload())
    assert isinstance(r.params, AtmosphericParameters)
    assert r.params.teff == pytest.approx(5731.2)
    assert r.params.method == "korg"
    assert r.params.converged is True
    assert r.is_measurement
    assert r.correlation("teff", "feh") == pytest.approx(0.87)
    assert r.correlation("feh", "teff") == pytest.approx(0.87)


def test_railed_fit_is_not_reported_as_converged():
    """A parked fit meets Korg's tolerances and means nothing. The rest of the
    pipeline keys off `converged`, so it must not see True here."""
    r = _to_result(_payload(retcode="railed", railed=["logg"], converged=True))
    assert r.retcode == "railed"
    assert r.railed == ["logg"]
    assert not r.is_measurement
    assert r.params.converged is False


def test_not_converged_is_not_a_measurement():
    r = _to_result(_payload(retcode="not_converged", converged=False))
    assert not r.is_measurement
    assert r.params.converged is False


def test_driver_error_yields_zeroed_params_and_a_retcode():
    r = _to_result({"status": "error", "error": "boom"})
    assert r.retcode == "solver_error"
    assert r.params.teff == 0.0
    assert not r.is_measurement


def test_correlation_needs_a_covariance():
    r = _to_result({"status": "error"})
    with pytest.raises(ValueError, match="no covariance"):
        r.correlation("teff", "feh")


# --- the real thing, only where Julia + KLOTHO exist ----------------------

@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("SPECIES_KORG_PROJECT"),
    reason="SPECIES_KORG_PROJECT not set",
)
def test_end_to_end_solar_linelist(tmp_path):
    """Drive the real Julia engine on a solar EW line list.

    Deliberately loose: this asserts the plumbing works and the covariance
    arrives, not that the parameters are right. Accuracy is validated against
    the Gaia FGK Benchmark Stars in the KLOTHO repository, not here.
    """
    fixture = Path(__file__).with_name("data") / "solar_moog_linelist.txt"
    if not fixture.exists():
        pytest.skip(f"no solar line list fixture at {fixture}")

    fitter = KorgFitter()
    ok, why = fitter.available()
    if not ok:
        pytest.skip(why)

    params = fitter.fit(fixture, initial_params=(0.0, 5777.0, 4.44, 1.0))
    result = fitter.last_result

    assert result is not None
    assert result.retcode in {"ok", "not_converged", "railed"}
    if result.is_measurement:
        assert 4500 < params.teff < 7000
        assert 3.0 < params.logg < 5.0
        assert result.covariance is not None
        assert result.covariance["order"] == ["teff", "logg", "vt", "feh"]
        # The whole point of this engine: a real correlation, not four
        # independent error bars.
        assert abs(result.correlation("teff", "feh")) > 0.1
