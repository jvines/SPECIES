"""Parse MOOG output files into structured results.

Replaces the duplicated ``compute_average_abundance()`` function that appeared
in Atmos.py, CalcErrors_new.py, and CalcErrorsGiant_new.py.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from astropy.stats import sigma_clip
from scipy.stats import linregress

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LineResult:
    """Parsed data for a single spectral line from MOOG output."""

    wavelength: float
    species_id: float  # e.g. 26.0 = Fe I, 26.1 = Fe II
    ep: float  # excitation potential (eV)
    loggf: float
    ew_obs: float  # observed equivalent width (mA)
    log_rw: float  # log(reduced equivalent width)
    abundance: float  # derived abundance
    deviation: float  # deviation from mean


@dataclass
class AbfindResult:
    """Parsed results from a MOOG ``abfind`` run."""

    fe_i_abundance: float = -99.0
    fe_ii_abundance: float = -99.0
    ep_slope: float = -99.0
    rw_slope: float = -99.0
    fe_i_fe_ii_diff: float = 0.0
    n_failed: int = 0
    n_fe_i: int = 0
    n_fe_ii: int = 0
    lines: dict[str, list[LineResult]] = field(default_factory=lambda: {"FeI": [], "FeII": []})


# ---------------------------------------------------------------------------
# Regex patterns (compiled once)
# ---------------------------------------------------------------------------

_RE_FAILED_1 = re.compile(r"OH NO! ANOTHER FAILED ITERATION!")
_RE_FAILED_2 = re.compile(r"CANNOT DECIDE ON A LINE WAVELENGTH STEP FOR")
_RE_SPECIES = re.compile(r"Abundance Results for Species (Fe I\s|Fe II)")
_RE_HAS_ALPHA = re.compile(r"[a-z]")
_RE_LINE_DATA = re.compile(r"[\d]+\s+[\d]+.+")
_RE_EP_CORR = re.compile(r"E\.P\. correlation")
_RE_RW_CORR = re.compile(r"R\.W\. correlation")
_RE_AVG_AB = re.compile(r"average abundance")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_abfind_output(
    output_path: Path,
    mode: str = "linearregression",
) -> AbfindResult:
    """Parse a MOOG abfind output file.

    Parameters
    ----------
    output_path
        Path to the MOOG ``*_out.test`` file.
    mode
        How to compute EP/RW slopes: ``"linearregression"`` (default),
        ``"odr"``, or ``"moog"`` (use MOOG's own reported values).

    Returns
    -------
    AbfindResult
        Structured result with abundances, slopes, and per-line data.
    """
    raw_lines: dict[str, list[str]] = {"FeI": [], "FeII": []}
    n_failed = 0
    ep_moog = -99.0
    rw_moog = -99.0
    avg_ab: dict[str, float] = {"FeI": -99.0, "FeII": -99.0}

    names_map = {"Fe I": "FeI", "Fe II": "FeII"}
    current_species: str | None = None

    with open(output_path) as f:
        for line in f:
            line = line.strip()

            # Count MOOG failures
            if _RE_FAILED_1.search(line) or _RE_FAILED_2.search(line):
                n_failed += 1
                continue

            # Detect species header
            m = _RE_SPECIES.search(line)
            if m:
                matched = m.group(1).strip()
                current_species = names_map.get(matched)
                continue

            # MOOG-reported correlations (used in "moog" mode)
            if _RE_EP_CORR.search(line) and current_species == "FeI":
                ep_moog = float(line.split()[4])
                continue

            if _RE_RW_CORR.search(line) and current_species == "FeI":
                rw_moog = float(line.split()[4])
                continue

            if _RE_AVG_AB.search(line) and current_species is not None:
                avg_ab[current_species] = float(line.split()[3])
                continue

            # Line data rows (no alphabetic characters, matches numeric pattern)
            if current_species is not None and not _RE_HAS_ALPHA.search(line):
                if _RE_LINE_DATA.search(line):
                    raw_lines[current_species].append(line)

    # Parse raw line strings into LineResult objects
    parsed_lines: dict[str, list[LineResult]] = {"FeI": [], "FeII": []}
    for species_key, lines_list in raw_lines.items():
        for raw in lines_list:
            cols = raw.split()
            if len(cols) >= 7:
                parsed_lines[species_key].append(
                    LineResult(
                        wavelength=float(cols[0]),
                        species_id=float(cols[1]),
                        ep=float(cols[2]),
                        loggf=float(cols[3]),
                        ew_obs=float(cols[4]),
                        log_rw=float(cols[5]),
                        abundance=float(cols[6]),
                        deviation=float(cols[7]) if len(cols) > 7 else 0.0,
                    )
                )

    # Compute EP/RW slopes and mean abundances
    result = AbfindResult(
        n_failed=n_failed,
        n_fe_i=len(parsed_lines["FeI"]),
        n_fe_ii=len(parsed_lines["FeII"]),
        lines=parsed_lines,
    )

    if mode == "moog":
        result.ep_slope = ep_moog
        result.rw_slope = rw_moog
        result.fe_i_abundance = avg_ab["FeI"]
        result.fe_ii_abundance = avg_ab["FeII"]

    elif mode == "linearregression":
        _compute_slopes_linregress(result, parsed_lines)

    elif mode == "odr":
        # ODR mode requires EW data from external file — fall back to linregress
        # The ODR variant will be implemented when ew.py provides the EW data
        _compute_slopes_linregress(result, parsed_lines)

    result.fe_i_fe_ii_diff = result.fe_i_abundance - result.fe_ii_abundance
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_slopes_linregress(
    result: AbfindResult,
    parsed_lines: dict[str, list[LineResult]],
) -> None:
    """Compute EP/RW slopes via linear regression with sigma clipping."""
    # Fe I — slopes + clipped mean
    fe_i = parsed_lines["FeI"]
    if len(fe_i) >= 3:
        ab = np.array([lr.abundance for lr in fe_i])
        ep_arr = np.array([lr.ep for lr in fe_i])
        rw_arr = np.array([lr.log_rw for lr in fe_i])

        clipped = sigma_clip(ab, maxiters=1)
        mask = ~clipped.mask

        if np.sum(mask) >= 3:
            idx_ep = np.argsort(ep_arr[mask])
            idx_rw = np.argsort(rw_arr[mask])
            result.ep_slope = linregress(ep_arr[mask][idx_ep], ab[mask][idx_ep]).slope
            result.rw_slope = linregress(rw_arr[mask][idx_rw], ab[mask][idx_rw]).slope
            result.fe_i_abundance = float(np.mean(ab[mask]))
        else:
            result.fe_i_abundance = float(np.mean(ab))
    elif len(fe_i) > 0:
        result.fe_i_abundance = float(np.mean([lr.abundance for lr in fe_i]))

    # Fe II — median (no slopes)
    fe_ii = parsed_lines["FeII"]
    if len(fe_ii) > 0:
        result.fe_ii_abundance = float(np.median([lr.abundance for lr in fe_ii]))


def parse_abundance_output(
    output_path: Path,
    fe_solar: float = 7.50,
) -> dict[str, tuple[float, float, int]]:
    """Parse a MOOG abundance output file (multi-element).

    Returns a dict mapping element name (e.g. ``"NaI"``) to
    ``(mean_abundance, std, n_lines)`` relative to solar.
    """
    results: dict[str, tuple[float, float, int]] = {}

    species_pattern = re.compile(r"Abundance Results for Species\s+(\S+\s*\S*)")
    current_species: str | None = None
    abundances: list[float] = []

    with open(output_path) as f:
        for line in f:
            line = line.strip()

            m = species_pattern.search(line)
            if m:
                # Save previous species
                if current_species is not None and abundances:
                    _store_abundance(results, current_species, abundances, fe_solar)
                current_species = m.group(1).strip().replace(" ", "")
                abundances = []
                continue

            if _RE_FAILED_1.search(line) or _RE_FAILED_2.search(line):
                continue

            # Line data
            if current_species and not _RE_HAS_ALPHA.search(line):
                cols = line.split()
                if len(cols) >= 7:
                    abundances.append(float(cols[6]))

    # Don't forget the last species
    if current_species is not None and abundances:
        _store_abundance(results, current_species, abundances, fe_solar)

    return results


def _store_abundance(
    results: dict[str, tuple[float, float, int]],
    species: str,
    abundances: list[float],
    fe_solar: float,
) -> None:
    """Sigma-clip and store mean abundance for one species."""
    ab = np.array(abundances)
    if len(ab) >= 3:
        clipped = sigma_clip(ab, sigma=1.5, maxiters=1)
        ab_clean = ab[~clipped.mask]
    else:
        ab_clean = ab

    if len(ab_clean) > 0:
        mean_ab = float(np.mean(ab_clean)) - fe_solar
        std_ab = float(np.std(ab_clean)) if len(ab_clean) > 1 else 0.0
        results[species] = (mean_ab, std_ab, len(ab_clean))
