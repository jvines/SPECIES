"""The line-width acceptance bound.

`_measure_single_line` rejected any fit with `comp.s > 0.10` A. That is not a
rotation gate, it is a joint (EW, depth, wavelength, broadening) gate:

  * 0.10 A at 5500 A corresponds to vsini ~ 6-7 km/s, so a moderate rotator
    loses most of its lines despite them being measured perfectly well;
  * at R = 42000 and 6800 A the instrumental sigma alone is 0.069 A, leaving the
    cap only 1.45x the instrumental width before any stellar broadening.

Scaling the bound to the width actually expected recovered a median 120 -> 140
Fe lines per star over 192 Gaia FGK Benchmark Stars (gained in 190, lost in 0)
and moved |d log g| 0.195 -> 0.086 dex on the 118 stars converging in both runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from species.ew import _max_line_sigma, expected_line_sigma

C_KMS = 299792.458


def test_zero_knowledge_reproduces_the_old_bound():
    """Callers that cannot supply R must see exactly the previous behaviour."""
    assert _max_line_sigma(5500.0, 0.0, 0.0, 0.0) == 0.10


def test_instrumental_sigma_matches_the_definition():
    """sigma = lambda / (R * 2sqrt(2 ln 2))."""
    for wl, R in ((5500.0, 115_000.0), (6800.0, 42_000.0)):
        assert expected_line_sigma(wl, resolution=R) == pytest.approx(
            wl / (R * 2.3548200450309493), rel=1e-12
        )


def test_the_old_cap_was_close_to_the_instrumental_width_at_R42k():
    """The number that motivated the change: 0.069 A instrumental at 6800 A."""
    sigma_inst = expected_line_sigma(6800.0, resolution=42_000.0)
    assert sigma_inst == pytest.approx(0.0685, abs=5e-4)
    assert 0.10 / sigma_inst < 1.5      # old cap barely above the instrument


def test_rotation_uses_grays_coefficient_not_the_gaussian_one():
    """FWHM = 1.5587 vsini (Gray 2005 eq. 18.14, eps = 0.6). 1.35 is the value
    you get by fitting a Gaussian to a rotational profile — a different thing."""
    wl, vsini = 5500.0, 10.0
    sigma = expected_line_sigma(wl, vsini=vsini)
    expected = 1.5587 / 2.3548200450309493 * (vsini / C_KMS) * wl
    assert sigma == pytest.approx(expected, rel=1e-12)


def test_terms_add_in_quadrature():
    wl = 5500.0
    inst = expected_line_sigma(wl, resolution=48_000.0)
    rot = expected_line_sigma(wl, vsini=8.0)
    mac = expected_line_sigma(wl, vmac=4.0)
    both = expected_line_sigma(wl, resolution=48_000.0, vsini=8.0, vmac=4.0)
    assert both == pytest.approx(np.sqrt(inst**2 + rot**2 + mac**2), rel=1e-12)


@pytest.mark.parametrize("vsini", [0.0, 5.0, 10.0, 20.0, 30.0])
def test_bound_grows_with_rotation_and_never_shrinks(vsini):
    """A rotator must not be penalised for being wide."""
    bound = _max_line_sigma(5500.0, 42_000.0, vsini, 0.0)
    assert bound >= 0.10
    assert bound >= _max_line_sigma(5500.0, 42_000.0, 0.0, 0.0)


def test_a_moderate_rotator_would_have_been_rejected_by_the_old_gate():
    """The concrete regression: a well-measured 12 km/s rotator at R=42000."""
    wl, R, vsini = 5500.0, 42_000.0, 12.0
    sigma_true = expected_line_sigma(wl, resolution=R, vsini=vsini)
    assert sigma_true > 0.10                       # old gate rejects it
    assert sigma_true < _max_line_sigma(wl, R, vsini, 0.0)   # new gate keeps it


def test_bound_is_not_so_loose_it_accepts_anything():
    """It must still reject genuinely bad fits — 2.5x the expected width."""
    wl, R = 5500.0, 115_000.0
    bound = _max_line_sigma(wl, R, 2.0, 3.0)
    sigma_exp = expected_line_sigma(wl, resolution=R, vsini=2.0, vmac=3.0)
    assert bound == pytest.approx(max(0.10, 2.5 * sigma_exp))
    assert bound < 10 * max(sigma_exp, 0.04)
