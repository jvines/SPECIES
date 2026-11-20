"""Fe rows of the abundance table (SPEC-4).

The table's Fe I sat 0.2-0.4 dex above the fitted [Fe/H] (+0.33 on the HARPS
Sun at the converged 5829/4.52/-0.00/0.70). Two causes:

* lines_ab.dat carries synthesis-style component lists -- satellites such as
  Fe I 5543.974 (log gf -6.06) beside 5543.936 (-1.14), or Fe II entries at
  10.4 eV -- and each entry was measured as a line of its own. The Gaussian
  fit found the same feature for all of them, MOOG returned +1 to +9 dex for
  the satellites, and the 1.5-sigma clip kept enough to lift the mean.
* its real Fe I lines have log gf a median 0.13 dex below the solar-calibrated
  values of linelist.dat for the same transitions.

Fixes: unresolved same-species components are folded into one line, the Fe
rows come from the solver's own lines and EWs, and the error is the measured
line scatter / sqrt(n).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from species import abundances as abmod
from species.abundances import fold_unresolved_blends, line_scatter_error
from species.atmosphere import AtmosphericParameters
from species.config import Settings
from species.ew import EWResult, LineList
from species.moog.parser import AbfindResult, LineResult

LINES_AB = Path(abmod.__file__).parent / "data" / "linelists" / "lines_ab.dat"


def _ll(rows):
    wl, ep, gf, ion = (np.array(c, dtype=float) for c in zip(*rows))
    return LineList(
        wavelengths=wl, excitation_potential=ep, loggf=gf,
        atomic_number=np.array(["X"] * len(rows)), ionization=ion,
    )


def _ew(wl, ew=50.0):
    return EWResult(wavelength=wl, ew=ew, ew_median=ew, ew_err_plus=1.0, ew_err_minus=1.0)


# --------------------------------------------------------------------------
# fold_unresolved_blends
# --------------------------------------------------------------------------

def test_satellite_is_dropped_and_barely_changes_gf():
    ll = _ll([(5543.936, 4.218, -1.140, 26.0), (5543.974, 4.154, -6.058, 26.0)])
    ews, out = fold_unresolved_blends([_ew(5543.936), _ew(5543.974)], ll, 5777.0)
    assert [r.wavelength for r in ews] == [5543.936]
    assert out.loggf[0] == pytest.approx(-1.140, abs=1e-4)


def test_same_ep_pair_sums_gf_exactly():
    """Na I 5688.193/5688.205: one feature, gf_eff = gf1 + gf2."""
    ll = _ll([(5688.193, 2.104, -1.406, 11.0), (5688.205, 2.104, -0.452, 11.0)])
    ews, out = fold_unresolved_blends([_ew(5688.193), _ew(5688.205)], ll, 5500.0)
    assert [r.wavelength for r in ews] == [5688.205]
    assert out.loggf[1] == pytest.approx(np.log10(10**-1.406 + 10**-0.452), abs=1e-12)
    assert out.loggf[0] == -1.406  # the original list is not modified in place
    assert ll.loggf[1] == -0.452


def test_strongest_by_boltzmann_weight_not_by_gf():
    """At 5151.9 the 1.0 eV entry dominates the 5.0 eV one by 9 dex."""
    ll = _ll([(5151.885, 5.033, -8.850, 26.0), (5151.911, 1.011, -3.322, 26.0)])
    ews, _ = fold_unresolved_blends([_ew(5151.885), _ew(5151.911)], ll, 5777.0)
    assert [r.wavelength for r in ews] == [5151.911]


def test_different_species_or_far_lines_are_not_merged():
    ll = _ll([
        (6000.00, 3.0, -1.0, 26.0), (6000.05, 3.0, -1.0, 26.1),  # different ion
        (6100.00, 3.0, -1.0, 26.0), (6100.15, 3.0, -1.0, 26.0),  # 0.15 A apart
    ])
    ews, out = fold_unresolved_blends([_ew(w) for w in ll.wavelengths], ll, 5777.0)
    assert sorted(r.wavelength for r in ews) == sorted(ll.wavelengths.tolist())
    np.testing.assert_array_equal(out.loggf, ll.loggf)


def test_failed_fit_at_dominant_falls_back_to_the_group():
    ll = _ll([(5638.262, 4.220, -0.870, 26.0), (5638.339, 4.584, -2.929, 26.0)])
    bad = EWResult(wavelength=5638.262, ew=0, ew_median=0, ew_err_plus=0, ew_err_minus=0)
    ews, _ = fold_unresolved_blends([bad, _ew(5638.339, ew=75.0)], ll, 5777.0)
    assert len(ews) == 1
    assert ews[0].wavelength == 5638.262 and ews[0].ew == 75.0


def test_bundled_list_has_one_entry_per_resolvable_feature():
    ll = LineList.from_file(LINES_AB)
    ews, _ = fold_unresolved_blends([_ew(float(w)) for w in ll.wavelengths], ll, 5777.0)
    by_ion: dict[float, list[float]] = {}
    for r in ews:
        i = int(np.where(ll.wavelengths == r.wavelength)[0][0])
        by_ion.setdefault(float(ll.ionization[i]), []).append(r.wavelength)
    for ion, wls in by_ion.items():
        assert np.all(np.diff(sorted(wls)) > 0.10), ion
    # the 52 Fe I / 15 Fe II entries are 41 / 10 features
    assert len(by_ion[26.0]) == 41
    assert len(by_ion[26.1]) == 10


# --------------------------------------------------------------------------
# line_scatter_error
# --------------------------------------------------------------------------

def test_error_is_measured_scatter_over_sqrt_n():
    ab = np.array([7.40, 7.50, 7.60, 7.45, 7.55])
    assert line_scatter_error(ab) == pytest.approx(np.std(ab, ddof=1) / np.sqrt(5))


def test_error_floor_and_small_samples():
    assert line_scatter_error(np.full(4, 7.5)) == pytest.approx(0.05 / 2)
    assert line_scatter_error(np.array([7.5])) == pytest.approx(0.2)
    assert line_scatter_error(np.array([7.5, 7.5])) == pytest.approx(0.2 / np.sqrt(2))
    assert line_scatter_error(np.array([7.0, 8.0])) == pytest.approx(np.std([7.0, 8.0], ddof=1) / np.sqrt(2))


# --------------------------------------------------------------------------
# Fe rows from the solver's abfind
# --------------------------------------------------------------------------

@dataclass
class _FakeSession:
    model_path: Path
    _summary: Path

    def run_abfind(self, read_mode="moog"):
        self._summary.write_text("")


class _FakeMoog:
    def __init__(self, tmp):
        self.tmp = tmp

    @contextmanager
    def open_session(self, _linelist):
        yield _FakeSession(self.tmp / "model", self.tmp / "summary.out")


class _FakeGrid:
    def interpolate_and_write(self, *a, **k):
        return True


def _line(ab, ion):
    return LineResult(6000.0, ion, 4.0, -1.0, 50.0, -5.0, ab, 0.0)


def test_fe_rows_come_from_the_solver_lines(monkeypatch, tmp_path):
    # lines_ab says Fe I is +0.33, the solver's lines say [Fe/H] = 0.00
    monkeypatch.setattr(abmod, "measure_equivalent_widths", lambda *a, **k: [_ew(5228.376)])
    monkeypatch.setattr(abmod, "_parse_multi_species", lambda *a, **k: {
        "FeI": (7.83, 0.03, 44), "FeII": (7.84, 0.06, 10), "NiI": (6.23, 0.04, 20),
    })
    fe_i = [7.42, 7.58, 7.42, 7.58, 7.42, 7.58]  # nothing clipped at 1.5 sigma
    fe_ii = [7.48, 7.52, 7.50]
    abfind = AbfindResult(lines={
        "FeI": [_line(a, 26.0) for a in fe_i], "FeII": [_line(a, 26.1) for a in fe_ii],
    })
    cfg = Settings(output_dir=tmp_path)
    params = AtmosphericParameters(teff=5777, logg=4.44, feh=0.0, vt=1.0, converged=True)
    args = (params, np.zeros(10), np.ones(10), 100.0, cfg, _FakeMoog(tmp_path), _FakeGrid())

    without = abmod.determine_abundances(*args)
    assert without["FeI"].abundance == pytest.approx(0.33)

    res = abmod.determine_abundances(*args, fe_abfind=abfind)
    assert res["FeI"].abundance == pytest.approx(np.mean(fe_i) - 7.50)
    assert res["FeI"].n_lines == len(fe_i)
    assert res["FeI"].uncertainty == pytest.approx(np.std(fe_i, ddof=1) / np.sqrt(len(fe_i)))
    assert res["FeII"].abundance == pytest.approx(0.0)
    assert res["NiI"].abundance == pytest.approx(6.23 - abmod.SOLAR_ABUNDANCE["Ni"])
