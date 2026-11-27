"""The abundance-line EW pass gets the same width information as the iron
pass (resolution, vsini, vmac). It used to get none, so its line-width gate
fell back to 0.10 A: HD 206860 (vsini ~10.6 km/s, HARPS) kept 147 Fe lines
with vsini given but only 2 Cr and 2 Ni lines and no Na/Mg/Si/Ca/Ti."""
from types import SimpleNamespace

import numpy as np

import species.abundances as ab


def test_determine_abundances_forwards_broadening(monkeypatch):
    seen = {}

    def fake_measure(wavelength, flux, snr, linelist_path, **kw):
        seen.update(kw)
        return []          # no valid EWs: determine_abundances returns early

    monkeypatch.setattr(ab, "measure_equivalent_widths", fake_measure)
    params = SimpleNamespace(converged=True)
    cfg = SimpleNamespace(linelist_ab_path="x")
    out = ab.determine_abundances(params, np.ones(3), np.ones(3), 100.0, cfg,
                                  None, None, resolution=115000.0, vsini=10.6,
                                  vmac=3.0)
    assert out == {}
    assert seen == {"resolution": 115000.0, "vsini": 10.6, "vmac": 3.0}


def test_width_bound_admits_a_10_kms_rotator_only_with_vsini():
    from species.ew import _max_line_sigma, expected_line_sigma
    wl = 6000.0
    sigma_line = expected_line_sigma(wl, resolution=115000.0, vsini=10.6)
    assert _max_line_sigma(wl, 0.0, 0.0, 0.0) < sigma_line * 1.3   # old path: cut
    assert _max_line_sigma(wl, 115000.0, 10.6, 0.0) > sigma_line * 2
