"""Per-line table (SPEC-8): EWs, atomic data and abfind abundances are written, not dropped."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from species.abundances import parse_abfind_line_rows
from species.atmosphere import AtmosphericParameters
from species.ew import EWResult, LineList
from species.io.results import (
    LINE_TABLE_COLUMNS, AnalysisMetadata, AnalysisResult, LineRecord, build_line_records,
)


def _linelist():
    return LineList(
        wavelengths=np.array([5501.47, 5522.45, 6084.11, 6247.56]),
        excitation_potential=np.array([0.96, 4.21, 3.20, 3.89]),
        loggf=np.array([-3.05, -1.47, -3.79, -2.30]),
        atomic_number=np.array([26, 26, 26, 26]),
        ionization=np.array([26.0, 26.0, 26.1, 26.1]),
    )


def _ews():
    return [
        EWResult(5501.47, 112.3, 112.0, 2.1, 2.5),
        EWResult(5522.45, 44.0, 44.1, 1.0, 1.2),
        EWResult(6084.11, 21.7, 21.5, 0.9, 0.8),
        EWResult(6247.56, 0.0, 0.0, 0.0, 0.0),   # failed fit: not in the table
    ]


def test_join_keeps_unused_lines_with_nan_abundance():
    moog = [(5501.47, 26.0, 0.96, -3.05, 112.3, 7.48), (6084.11, 26.1, 3.20, -3.79, 21.7, 7.52)]
    rec = build_line_records(_ews(), [_linelist()], moog)
    assert [r.wavelength for r in rec] == [5501.47, 5522.45, 6084.11]
    by = {r.wavelength: r for r in rec}
    assert by[5501.47].used and by[5501.47].abundance == pytest.approx(7.48)
    assert by[5501.47].ew_err == pytest.approx(2.5)            # larger MC error
    assert not by[5522.45].used and np.isnan(by[5522.45].abundance)
    assert by[5522.45].ew == pytest.approx(44.0) and by[5522.45].loggf == pytest.approx(-1.47)
    assert by[6084.11].species == pytest.approx(26.1)


def _result(lines):
    p = AtmosphericParameters(teff=5777.0, logg=4.44, feh=0.0, vt=1.0, converged=True)
    return AnalysisResult(star_name="SUN", params=p, metadata=AnalysisMetadata(), lines=lines)


LINES = [LineRecord(5501.47, 26.0, 0.96, -3.05, 112.3, 2.5, 7.48, True),
         LineRecord(5522.45, 26.0, 4.21, -1.47, 44.0, 1.2, float("nan"), False),
         LineRecord(5711.09, 12.0, 4.35, -1.72, 105.0, 3.0, 7.61, True)]


def test_fits_has_lines_hdu(tmp_path):
    path = _result(LINES).to_fits(tmp_path / "r.fits")
    with fits.open(path) as h:
        assert "LINES" in h
    t = Table.read(path, hdu="LINES")
    assert tuple(t.colnames) == LINE_TABLE_COLUMNS and len(t) == 3
    ab = np.ma.filled(np.ma.asarray(t["abundance"], dtype=float), np.nan)
    assert ab[2] == pytest.approx(7.61) and np.isnan(ab[1])
    assert list(t["used"]) == [True, False, True]


def test_ascii_writes_sidecar(tmp_path):
    r = _result(LINES)
    path = r.to_ascii(tmp_path / "r.dat")
    side = AnalysisResult.line_table_path(path)
    assert side.name == "r_lines.dat"
    t = Table.read(side, format="ascii.basic")
    assert t["ew_err"][0] == pytest.approx(2.5) and t["species"][2] == pytest.approx(12.0)


def test_no_lines_no_sidecar(tmp_path):
    path = _result([]).to_ascii(tmp_path / "r.dat")
    assert not AnalysisResult.line_table_path(path).exists()


def test_line_records_are_plain_dicts():
    d = _result(LINES).line_records()
    assert d[0] == dict(zip(LINE_TABLE_COLUMNS, (5501.47, 26.0, 0.96, -3.05, 112.3, 2.5, 7.48, True)))


SUMMARY = """Abundance Results for Species Mg I        (input abundance =   7.600)
 wavelength        ID      EP     logGF     EWin   logRWin   abund    del avg
   5711.090   12.00000   4.350   -1.720   105.00    -4.735   7.610   0.010
   6318.720   12.00000   5.110   -1.940    48.20    -5.118   7.590  -0.010
average abundance =   7.600  std. deviation =   0.014  #lines =   2
OH NO! ANOTHER FAILED ITERATION!
Abundance Results for Species Ti II       (input abundance =   4.950)
 wavelength        ID      EP     logGF     EWin   logRWin   abund    del avg
   5418.770   22.10000   1.580   -2.130    49.10    -5.043   4.980   0.000
"""


def test_parse_line_rows(tmp_path):
    p = tmp_path / "summary.out"
    p.write_text(SUMMARY)
    rows = parse_abfind_line_rows(p)
    assert set(rows) == {"MgI", "TiII"}
    assert rows["MgI"][0] == pytest.approx((5711.09, 12.0, 4.35, -1.72, 105.0, 7.61))
    assert len(rows["MgI"]) == 2 and rows["TiII"][0][5] == pytest.approx(4.98)
