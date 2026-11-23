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
        return path

    def to_fits(self, path: Path) -> Path:
        """Write FITS table."""
        table = self.to_table()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        table.write(path, format="fits", overwrite=True)
        return path
