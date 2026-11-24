"""AnalysisResult writers: every column must be scalar and typed (SPEC-7)."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.table import Table

from species.atmosphere import AtmosphericParameters
from species.io.results import AnalysisMetadata, AnalysisResult

# to_ascii writes right-aligned, space-separated columns: whitespace-split
# reads it back when no string field is empty.
_READ = {"ascii": {"format": "ascii.basic"}, "fits": {"format": "fits"}}


def _result(**diag):
    p = AtmosphericParameters(teff=6027.0, logg=4.53, feh=0.184, vt=0.875, converged=True,
                              method="broyden", **diag)
    return AnalysisResult(star_name="TEST", params=p,
                          metadata=AnalysisMetadata(instrument="MAROON-X", rv=1.4, snr=60.0))


@pytest.mark.parametrize("fmt", ["ascii", "fits"])
def test_writers_with_residual_vector(tmp_path, fmt):
    r = _result(final_residuals=(1e-4, -0.01, 0.002, 3e-5), n_iters=42, hit_iter_cap=False,
                last_moog_status="ok", n_fe_lines_input=120)
    path = tmp_path / f"r.{'dat' if fmt == 'ascii' else 'fits'}"
    (r.to_ascii if fmt == "ascii" else r.to_fits)(path)
    t = Table.read(path, **_READ[fmt])
    assert "final_residuals" not in t.colnames
    assert t["resid_ep_slope"][0] == pytest.approx(1e-4)
    assert t["resid_fe1_fe2"][0] == pytest.approx(-0.01)
    assert t["resid_feh_in_out"][0] == pytest.approx(0.002)
    assert t["resid_rw_slope"][0] == pytest.approx(3e-5)
    assert t["teff"][0] == pytest.approx(6027.0)


@pytest.mark.parametrize("fmt", ["ascii", "fits"])
def test_writers_with_unknown_diagnostics(tmp_path, fmt):
    r = _result()          # every diagnostic None
    path = tmp_path / f"r.{'dat' if fmt == 'ascii' else 'fits'}"
    (r.to_ascii if fmt == "ascii" else r.to_fits)(path)
    if fmt == "ascii":     # empty last_moog_status: check the header and NaN fields only
        head, row = path.read_text().splitlines()
        assert "resid_ep_slope" in head.split() and "final_residuals" not in head
        assert row.split().count("nan") >= 6
        return
    t = Table.read(path, **_READ[fmt])
    assert np.isnan(float(t["resid_ep_slope"][0]))
    assert np.isnan(float(t["n_iters"][0]))
    assert str(t["last_moog_status"][0]).strip() in ("", "--")


def test_to_dict_keeps_the_vector():
    r = _result(final_residuals=(1.0, 2.0, 3.0, 4.0))
    assert r.to_dict()["final_residuals"] == [1.0, 2.0, 3.0, 4.0]
