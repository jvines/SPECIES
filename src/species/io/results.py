"""Result dataclasses and serialization for SPECIES analysis output.

Provides ``AnalysisResult`` — the top-level container returned by
``Analyzer.run()`` — with methods to serialize to dict, astropy Table,
ASCII, and FITS formats.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

from species.atmosphere import AtmosphericParameters
from species.errors import ParameterErrors
from species.ew import EWResult
from species.moog import AbfindResult

# Column names for AtmosphericParameters.final_residuals in tabular output.
_RESIDUAL_COLUMNS = ("resid_ep_slope", "resid_fe1_fe2", "resid_feh_in_out", "resid_rw_slope")
# Keys whose unknown value is written as "" rather than NaN.
_STRING_COLUMNS = frozenset({"star_name", "method", "instrument", "last_moog_status"})


@dataclass
class BroadeningLineData:
    """Per-line broadening fit data for diagnostic plots."""

    name: str               # e.g. "FeI 6027.05"
    wave_obs: np.ndarray    # observed wavelength window
    flux_obs: np.ndarray    # observed normalized flux
    wave_synth: np.ndarray  # synthetic wavelength
    flux_synth: np.ndarray  # best-fit broadened synthetic flux
    v_grid: np.ndarray      # vsini grid values
    chi2: np.ndarray        # chi-squared at each grid point
    vsini: float            # best-fit vsini for this line
    vmac: float             # macroturbulence used


@dataclass
class BroadeningResult:
    """Broadening parameters (vsini, vmac)."""

    vsini: float = 0.0
    vsini_err: float = 0.0
    vmac: float = 0.0
    vmac_err: float = 0.0
    per_line: list[BroadeningLineData] = field(default_factory=list)


@dataclass
class ElementAbundanceResult:
    """Chemical abundance for a single element."""

    element: str
    abundance: float  # [X/H]
    uncertainty: float
    n_lines: int


@dataclass
class LineRecord:
    """One measured line: atomic data, EW and its MOOG abundance.

    ``abundance`` is MOOG abfind's A(X) at the final atmosphere, NaN for a line
    that was measured but did not enter MOOG (quality cuts, failed fit) or went
    through MOOG blends. ``ew_err`` is the larger of the MC EW errors.
    """

    wavelength: float      # Angstrom
    species: float         # MOOG species code, e.g. 26.0 = Fe I, 26.1 = Fe II
    ep: float              # excitation potential chi (eV)
    loggf: float
    ew: float              # mA (the value handed to MOOG)
    ew_err: float          # mA
    abundance: float       # A(X) from abfind at the final atmosphere
    used: bool             # entered the MOOG abfind run


LINE_TABLE_COLUMNS = ("wavelength", "species", "ep", "loggf", "ew", "ew_err", "abundance", "used")


def build_line_records(
    ew_results: list[EWResult] | None,
    linelists: list[Any],
    moog_rows: list[tuple],
) -> list[LineRecord]:
    """Join measured EWs, atomic data and MOOG per-line abundances.

    ``linelists`` are :class:`species.ew.LineList` objects (the Fe list and the
    abundance list); ``ew_results`` the EWs measured on them; ``moog_rows`` the
    ``(wavelength, species_id, ep, loggf, ew, abundance)`` rows MOOG returned.
    Lines are matched on wavelength rounded to 0.01 A (the precision of the MOOG
    line-list file) and species.
    """
    err = {}
    for r in ew_results or []:
        if r.is_valid:
            err[round(float(r.wavelength), 2)] = (float(r.ew),
                                                   float(max(r.ew_err_plus, r.ew_err_minus)))
    used = {}
    for w, sp, ep, gf, ew, ab in moog_rows:
        used[(round(float(w), 2), round(float(sp), 1))] = (float(ep), float(gf), float(ew), float(ab))
    out, seen = [], set()
    for ll in linelists:
        for i in range(len(ll.wavelengths)):
            w = round(float(ll.wavelengths[i]), 2)
            sp = round(float(ll.ionization[i]), 1)
            key = (w, sp)
            if key in seen or (w not in err and key not in used):
                continue
            seen.add(key)
            ew, ew_err = err.get(w, (np.nan, np.nan))
            if key in used:
                ep, gf, ew_moog, ab = used[key]
                out.append(LineRecord(w, sp, ep, gf, ew_moog, ew_err, ab, True))
            else:
                out.append(LineRecord(w, sp, float(ll.excitation_potential[i]),
                                      float(ll.loggf[i]), ew, ew_err, np.nan, False))
    # MOOG rows whose transition is in neither list (should not happen) are kept too
    for (w, sp), (ep, gf, ew, ab) in used.items():
        if (w, sp) not in seen:
            out.append(LineRecord(w, sp, ep, gf, ew, err.get(w, (np.nan, np.nan))[1], ab, True))
    out.sort(key=lambda r: (r.species, r.wavelength))
    return out


@dataclass
class AnalysisMetadata:
    """Metadata about the analysis run."""

    instrument: str = "unknown"
    rv: float = 0.0  # km/s from CCF
    snr: float = 0.0
    method: str = "per_parameter"


@dataclass
class AnalysisResult:
    """Complete result from a SPECIES analysis.

    This is the main output container returned by ``Analyzer.run()``.
    """

    star_name: str
    params: AtmosphericParameters
    errors: ParameterErrors | None = None
    abundances: dict[str, ElementAbundanceResult] = field(default_factory=dict)
    broadening: BroadeningResult | None = None
    ew_results: list[EWResult] | None = None
    abfind: AbfindResult | None = None
    metadata: AnalysisMetadata = field(default_factory=AnalysisMetadata)
    lines: list[LineRecord] = field(default_factory=list)

    def line_records(self) -> list[dict[str, Any]]:
        """Per-line table as a list of plain dicts (JSON/database friendly)."""
        return [asdict(r) for r in self.lines]

    def to_line_table(self) -> Any:
        """Per-line table (wavelength, species, chi, log gf, EW, sigma_EW, abundance)."""
        from astropy.table import Table
        cols = {c: [getattr(r, c) for r in self.lines] for c in LINE_TABLE_COLUMNS}
        t = Table(cols, dtype=[float, float, float, float, float, float, float, bool])
        t["wavelength"].unit = "Angstrom"
        t["ep"].unit = "eV"
        t["ew"].unit = t["ew_err"].unit = "mAngstrom"
        return t

    @staticmethod
    def line_table_path(path: Path) -> Path:
        """Sidecar path of the ASCII per-line table for a summary at ``path``."""
        path = Path(path)
        return path.with_name(f"{path.stem}_lines{path.suffix}")

    def to_dict(self) -> dict[str, Any]:
        """Flat dictionary suitable for database storage or JSON serialization."""
        d: dict[str, Any] = {
            "star_name": self.star_name,
            "teff": self.params.teff,
            "logg": self.params.logg,
            "feh": self.params.feh,
            "vmic": self.params.vt,
            "n_fe_i": self.params.n_fe_i,
            "n_fe_ii": self.params.n_fe_ii,
            "converged": self.params.converged,
            "method": self.params.method,
            "final_residuals": (
                list(self.params.final_residuals)
                if self.params.final_residuals is not None else None
            ),
            "n_iters": self.params.n_iters,
            "hit_iter_cap": self.params.hit_iter_cap,
            "last_moog_status": self.params.last_moog_status,
            "n_fe_lines_input": self.params.n_fe_lines_input,
            "instrument": self.metadata.instrument,
            "rv": self.metadata.rv,
            "snr": self.metadata.snr,
        }

        if self.errors:
            d.update({
                "teff_error": self.errors.err_teff,
                "logg_error": self.errors.err_logg,
                "feh_error": self.errors.err_feh,
                "vmic_error": self.errors.err_vt,
            })

        if self.broadening:
            d.update({
                "vsini": self.broadening.vsini,
                "vsini_error": self.broadening.vsini_err,
                "vmac": self.broadening.vmac,
                "vmac_error": self.broadening.vmac_err,
            })

        # Abundances as flat keys
        for name, ab in self.abundances.items():
            d[f"{name}_abundance"] = ab.abundance
            d[f"{name}_error"] = ab.uncertainty
            d[f"{name}_nlines"] = ab.n_lines

        return d

    def to_table(self) -> Any:
        """Return as an astropy Table (single row)."""
        from astropy.table import Table
        d = self.to_dict()
        # One scalar, typed column per key: the fixed-width and FITS writers
        # reject vector columns and object (None-holding) columns.
        res = d.pop("final_residuals", None)
        for i, name in enumerate(_RESIDUAL_COLUMNS):
            d[name] = float(res[i]) if res is not None else None
        cols = {}
        for k, v in d.items():
            if v is None:
                v = "" if k in _STRING_COLUMNS else np.nan
            cols[k] = [v]
        return Table(cols)

    def to_ascii(self, path: Path) -> Path:
        """Write ASCII summary file."""
        from astropy.io import ascii as asc
        table = self.to_table()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        asc.write(table, path, format="fixed_width", delimiter=None, overwrite=True)
        if self.lines:
            asc.write(self.to_line_table(), self.line_table_path(path), format="fixed_width",
                      delimiter=None, overwrite=True)
        return path

    def to_fits(self, path: Path) -> Path:
        """Write FITS table."""
        table = self.to_table()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        table.write(path, format="fits", overwrite=True)
        if self.lines:
            from astropy.io import fits
            hdu = fits.table_to_hdu(self.to_line_table())
            hdu.name = "LINES"
            with fits.open(path, mode="append") as hdul:
                hdul.append(hdu)
        return path
