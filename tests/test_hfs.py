"""Hyperfine structure for Mn I and Cu I (SPEC-2).

Treating HFS-split lines as single lines in abfind overestimates their
abundances. On the HARPS solar spectrum (sun01) at SPECIES' converged
parameters the old path gave [Mn/H] = +0.42 and [Cu/H] = +0.32; through MOOG
blends with the components in ``hfs_mn_cu.dat`` they are -0.03 and +0.00.
"""

from __future__ import annotations

import shutil
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from species.config import Settings
from species.ew import EWResult, LineList
from species.hfs import (
    HFSComponent,
    components_for,
    load_hfs_table,
    parse_blends_summary,
    write_blends_linelist,
)
from species.moog.par_file import write_blends_par

HFS_IONS = (25.0, 29.0)


@pytest.fixture(scope="module")
def config():
    return Settings()


@pytest.fixture(scope="module")
def table(config):
    return load_hfs_table(config.linelist_hfs_path)


@pytest.fixture(scope="module")
def ab_linelist(config):
    return LineList.from_file(config.linelist_ab_path)


def _hfs_parents(ab_linelist):
    mask = np.isin(ab_linelist.ionization, HFS_IONS)
    return [
        (float(w), float(ep), float(gf), float(ion))
        for w, ep, gf, ion in zip(
            ab_linelist.wavelengths[mask],
            ab_linelist.excitation_potential[mask],
            ab_linelist.loggf[mask],
            ab_linelist.ionization[mask],
        )
    ]


# ---------------------------------------------------------------------------
# The component table
# ---------------------------------------------------------------------------


def test_every_mn_and_cu_line_has_hfs_components(table, ab_linelist):
    parents = _hfs_parents(ab_linelist)
    assert len(parents) >= 8
    for wl, *_ in parents:
        comps = components_for(table, wl)
        assert comps is not None and len(comps) >= 2, f"{wl} has no HFS components"


def test_no_other_element_is_split(table, ab_linelist):
    others = ab_linelist.wavelengths[~np.isin(ab_linelist.ionization, HFS_IONS)]
    assert all(components_for(table, w) is None for w in others)


def test_components_sum_to_the_line_gf(table, ab_linelist):
    """sum(gf_i) must equal the total gf the rest of the line list uses."""
    for wl, _, loggf, _ in _hfs_parents(ab_linelist):
        comps = components_for(table, wl)
        total = np.log10(np.sum(10.0 ** np.array([c.loggf for c in comps])))
        assert total == pytest.approx(loggf, abs=1e-3), wl


def test_components_keep_the_line_wavelength_and_level(table, ab_linelist):
    """gf-weighted centroid at the lines_ab.dat air wavelength, same EP and ion."""
    for wl, ep, _, ion in _hfs_parents(ab_linelist):
        comps = components_for(table, wl)
        w = np.array([c.wavelength for c in comps])
        g = 10.0 ** np.array([c.loggf for c in comps])
        assert np.sum(g * w) / g.sum() == pytest.approx(wl, abs=2e-3)
        assert np.all(np.abs(w - wl) < 0.2)
        assert all(c.excitation_potential == pytest.approx(ep, abs=1e-3) for c in comps)
        assert all(c.ion == ion for c in comps)


def test_known_bad_mn_lines_are_gone(ab_linelist):
    """5255.38 (EP 5.52, log gf -8.78) took the EW of Mn I 5255.33 and returned
    A(Mn) ~ 15.7; the 5505.87 feature is dominated by Fe I 5505.88."""
    mn = ab_linelist.wavelengths[ab_linelist.ionization == 25.0]
    assert not np.any(np.isclose(mn, 5255.38, rtol=0, atol=1e-3))
    assert not np.any(np.isclose(mn, 5505.866, rtol=0, atol=1e-3))


# ---------------------------------------------------------------------------
# MOOG blends I/O
# ---------------------------------------------------------------------------


def test_blends_linelist_format(tmp_path):
    comps = [HFSComponent(5105.4965, 1.389, -2.7468, 29.0),
             HFSComponent(5105.5062, 1.389, -2.9348, 29.0),
             HFSComponent(5105.5646, 1.389, -2.1283, 29.0)]
    path = tmp_path / "b.txt"
    write_blends_linelist(path, [(comps, 88.94)])
    rows = path.read_text().splitlines()[1:]
    assert len(rows) == 3
    # MOOG reads (7e10.3): first member positive with the EW, others negative
    fields = [[r[i:i + 10] for i in range(0, 70, 10)] for r in rows]
    assert float(fields[0][0]) == pytest.approx(5105.4965)
    assert float(fields[0][6]) == pytest.approx(88.94)
    for f in fields[1:]:
        assert float(f[0]) < 0
        assert float(f[1]) == 29.0


def test_blends_par_names_the_element(tmp_path):
    par = write_blends_par(
        tmp_path / "batch.par", model_path=Path("m.atm"), linelist_path=Path("l.txt"),
        summary_out=Path("s.out"), standard_out=Path("o.out"), atom=25.0,
    ).read_text().splitlines()
    assert par[0] == "blends"
    i = par.index("blenlimits")
    assert par[i + 1].split()[-1] == "25"


_BLENDS_SUMMARY = """\
hfs
          Teff =  5777 log g = 4.44                   vt= 1.00 M/H= 0.00
Abundance Results for Species Cu I         (input abundance =   4.190)
wavelength         ID      EP   logGF     EWin   logRWin     abund   delavg
  5105.497   29.00000   1.389  -2.747    88.94    -4.759   999.990  999.990
  5105.565   29.00000   1.389  -2.128    88.94    -4.759     4.075   -0.088
  5218.193   29.00000   3.817  -0.046    51.97    -5.002     4.099   -0.064
  5218.202   29.00000   3.817  -0.506    51.97    -5.002   999.990  999.990
  5700.287   29.00000   1.642  -3.102    30.16    -5.276     4.292    0.129
average abundance =  4.163     std. deviation =  0.098     #lines =   3
E.P. correlation:  slope =  -0.013  intercept =   4.196  corr. coeff. =  -0.169
"""


def test_parse_blends_summary_skips_unassigned_members(tmp_path):
    p = tmp_path / "summary.out"
    p.write_text(_BLENDS_SUMMARY)
    out = parse_blends_summary(p)
    assert list(out) == ["CuI"]
    assert [ab for _, ab in out["CuI"]] == [4.075, 4.099, 4.292]


# ---------------------------------------------------------------------------
# Routing in determine_abundances
# ---------------------------------------------------------------------------


class _FakeSession:
    def __init__(self, run_dir: Path, linelist_path: Path, calls: list):
        self.model_path = run_dir / "model.atm"
        self._summary = run_dir / "summary.out"
        self.calls = calls
        self.abfind_lines = linelist_path.read_text()

    def run_abfind(self, read_mode="linearregression"):
        self.calls.append(("abfind", None))
        self._summary.write_text(
            "Abundance Results for Species Ni I         (input abundance =   6.220)\n"
            "wavelength         ID      EP   logGF     EWin   logRWin     abund   delavg\n"
            "  5115.39   28.00000   3.834  -0.110    60.00    -5.000     6.220    0.000\n"
        )

    def run_blends(self, linelist_path, atom):
        self.calls.append(("blends", atom))
        text = Path(linelist_path).read_text().splitlines()[1:]
        assert all(float(r[10:20]) == atom for r in text)
        name = {25.0: "Mn I", 29.0: "Cu I"}[atom]
        rows = [r for r in text if float(r[:10]) > 0]  # one result per blend
        p = self.model_path.parent / "blends_summary.out"
        p.write_text(
            f"Abundance Results for Species {name}         (input abundance =   5.000)\n"
            + "".join(f"  {float(r[:10]):8.3f}   {atom:.5f}   1.000  -1.000    50.00"
                      f"    -5.000     {4.5:.3f}    0.000\n" for r in rows)
            + "average abundance =  4.500\n"
        )
        return p


def test_mn_and_cu_go_through_blends_not_abfind(config, tmp_path, monkeypatch):
    import species.abundances as ab
    from species.atmosphere import AtmosphericParameters

    linelist = LineList.from_file(config.linelist_ab_path)
    wanted = {5115.392, 5105.537, 5218.197, 5377.607}
    ews = [EWResult(float(w), 50.0, 50.0, 2.0, 2.0) for w in linelist.wavelengths
           if round(float(w), 3) in wanted]
    assert len(ews) == len(wanted)
    monkeypatch.setattr(ab, "measure_equivalent_widths", lambda *a, **k: ews)

    calls: list = []
    sessions: list = []

    class FakeRunner:
        @contextmanager
        def open_session(self, linelist_path):
            s = _FakeSession(tmp_path, Path(linelist_path), calls)
            sessions.append(s)
            yield s

    class FakeGrid:
        def interpolate_and_write(self, *a, **k):
            return True

    cfg = config.model_copy(update={"output_dir": tmp_path})
    params = AtmosphericParameters(teff=5777, logg=4.44, feh=0.0, vt=1.0, converged=True)
    res = ab.determine_abundances(params, np.zeros(1), np.zeros(1), 100.0, cfg,
                                  FakeRunner(), FakeGrid())

    assert ("blends", 25.0) in calls and ("blends", 29.0) in calls
    abfind_text = sessions[0].abfind_lines
    assert "5115.39" in abfind_text
    for w in ("5105.54", "5218.20", "5377.61"):
        assert w not in abfind_text, f"{w} still in the abfind list"
    assert res["CuI"].n_lines == 2 and res["MnI"].n_lines == 1
    assert res["CuI"].abundance == pytest.approx(4.5 - 4.19)
    assert "NiI" in res


# ---------------------------------------------------------------------------
# Real MOOG
# ---------------------------------------------------------------------------


def _moog_or_skip(config):
    exe = shutil.which(config.moog_binary) or (
        config.moog_binary if Path(config.moog_binary).exists() else None)
    if exe is None:
        pytest.skip("MOOG not available")
    return exe


@pytest.mark.slow
def test_blends_lowers_cu_5105_in_the_sun(config, table, tmp_path):
    """Cu I 5105.5 at its solar EW: abfind returns ~4.6, HFS ~4.1 (A_sun = 4.19)."""
    from species.moog.atmosphere_grid import AtmosphereGrid
    from species.moog.wrapper import MOOGRunner

    exe = _moog_or_skip(config)
    linelist = tmp_path / "abfind.txt"
    linelist.write_text("abfind\n  5105.54    29.0        1.39     -1.542"
                        "                       88.9400\n")
    runner = MOOGRunner(exe, config.moog_data_dir)
    with runner.open_session(linelist) as session:
        AtmosphereGrid.get(config.atlas9_dir).interpolate_and_write(
            5777.0, 4.44, 0.0, 1.0, session.model_path, fe_solar=7.50)
        single = session.run_abfind(read_mode="moog")
        assert single is not None
        abfind_ab = next(float(r.split()[6]) for r in session._summary.read_text().splitlines()
                         if r.strip().startswith("5105."))
        blends_in = tmp_path / "blends.txt"
        write_blends_linelist(blends_in, [(components_for(table, 5105.537), 88.94)])
        hfs_ab = next(ab for _, ab in parse_blends_summary(
            session.run_blends(blends_in, 29.0))["CuI"])

    assert abfind_ab - hfs_ab > 0.3
    assert hfs_ab == pytest.approx(4.19, abs=0.2)
