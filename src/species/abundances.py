"""Chemical abundance determination for non-iron elements.

Ported from Atmos.py ``runMOOG_ab()`` and CalcParams.py ``calc_ab()``.
Runs MOOG in abfind mode with a multi-element line list to determine
abundances of Na, Mg, Al, Si, Ca, Ti, Cr, Mn, Ni, Cu, Zn, etc.

The flow:
1. Measure EWs for the abundance linelist (same Gaussian fitting as for Fe)
2. Filter into MOOG-format linelist
3. Run MOOG abfind at the converged atmospheric parameters
4. Parse per-species abundances from MOOG output
5. Sigma-clip, compute mean [X/H] relative to solar (Asplund 2009)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.stats import sigma_clip

from species.atmosphere import AtmosphericParameters
from species.config import Settings
from species.data.solar import ELEMENTS, SOLAR_ABUNDANCE
from species.ew import EWResult, LineList, create_moog_linelist, measure_equivalent_widths
from species.moog.atmosphere_grid import AtmosphereGrid
from species.moog.wrapper import MOOGError, MOOGRunner, MOOGSession
from species.moog.par_file import write_abfind_par

logger = logging.getLogger(__name__)


@dataclass
class ElementAbundance:
    """Abundance result for a single chemical element."""

    element: str  # e.g. "NaI", "MgI", "TiII"
    abundance: float  # [X/H] relative to solar
    uncertainty: float
    n_lines: int
    solar_reference: float  # A(X)_sun from Asplund 2009


def determine_abundances(
    params: AtmosphericParameters,
    wavelength: np.ndarray,
    flux: np.ndarray,
    snr: float | np.ndarray,
    config: Settings,
    moog: MOOGRunner,
    grid: AtmosphereGrid,
) -> dict[str, ElementAbundance]:
    """Determine chemical abundances from a multi-element line list.

    Parameters
    ----------
    params
        Converged atmospheric parameters.
    wavelength, flux
        1D rest-frame corrected spectrum.
    snr
        Signal-to-noise ratio.
    config
        SPECIES configuration.
    moog
        MOOG runner instance.
    grid
        Atmosphere grid for model interpolation.

    Returns
    -------
    dict[str, ElementAbundance]
        Mapping from ion name (e.g. ``"NaI"``) to abundance result.
    """
    if not params.converged:
        logger.warning("Atmospheric parameters did not converge; skipping abundances")
        return {}

    # 1. Measure EWs for the abundance linelist
    logger.info("Measuring EWs for abundance linelist")
    ew_results = measure_equivalent_widths(
        wavelength, flux, snr, config.linelist_ab_path,
    )
    valid_ew = [r for r in ew_results if r.is_valid]
    if not valid_ew:
        logger.warning("No valid EWs measured for abundance lines")
        return {}

    # 2. Create MOOG-format linelist
    ab_linelist = LineList.from_file(config.linelist_ab_path)
    moog_ab_path = config.output_dir / "moog_ab_linelist.txt"
    create_moog_linelist(ew_results, ab_linelist, moog_ab_path,
                         ew_min=5.0, ew_max=200.0, max_relative_error=0.8)

    # 3. Run MOOG abfind at converged parameters
    try:
        with moog.open_session(moog_ab_path) as session:
            ok = grid.interpolate_and_write(
                params.teff, params.logg, params.feh, params.vt,
                session.model_path, fe_solar=config.fe_solar,
            )
            if not ok:
                logger.warning("Grid interpolation failed for abundances")
                return {}

            session.run_abfind(read_mode="moog")

            # 4. Parse the MOOG summary output for ALL species
            summary_path = session._summary
            if not summary_path.exists():
                logger.warning("No MOOG output for abundance determination")
                return {}

            raw_abundances = _parse_multi_species(summary_path, config.fe_solar)

    except MOOGError as e:
        logger.warning("MOOG failed for abundance determination: %s", e)
        return {}

    # 5. Build ElementAbundance objects
    abundances: dict[str, ElementAbundance] = {}
    for ion_name, (mean_ab, std_ab, n_lines) in raw_abundances.items():
        # Map ion name to element for solar reference
        element = _ion_to_element(ion_name)
        solar = SOLAR_ABUNDANCE.get(element, 0.0)

        abundances[ion_name] = ElementAbundance(
            element=ion_name,
            abundance=float(mean_ab - solar),
            uncertainty=float(std_ab),
            n_lines=n_lines,
            solar_reference=solar,
        )

    logger.info("Determined abundances for %d ions", len(abundances))
    return abundances


# ---------------------------------------------------------------------------
# MOOG output parsing for multi-species
# ---------------------------------------------------------------------------

# Regex patterns
# Capture only the species name (e.g. "Fe I"), not the "(input abundance = ...)" part
_RE_SPECIES = re.compile(r"Abundance Results for Species\s+(\S+\s*\S*?)\s*(?:\(|\.)")
_RE_HAS_ALPHA = re.compile(r"[a-z]")
_RE_LINE_DATA = re.compile(r"[\d]+\s+[\d]+.+")
_RE_FAILED = re.compile(r"OH NO! ANOTHER FAILED|CANNOT DECIDE")


def _parse_multi_species(
    output_path: Path,
    fe_solar: float,
) -> dict[str, tuple[float, float, int]]:
    """Parse MOOG output for multiple species.

    Returns dict mapping ion name (e.g. ``"NaI"``) to
    ``(mean_abundance, std, n_lines)``. Abundances are on the A(X) scale,
    NOT relative to solar.
    """
    species_data: dict[str, list[float]] = {}
    current_species: str | None = None

    with open(output_path) as f:
        for line in f:
            line = line.strip()

            if _RE_FAILED.search(line):
                continue

            m = _RE_SPECIES.search(line)
            if m:
                raw_name = m.group(1).strip()
                current_species = raw_name.replace(" ", "")
                if current_species not in species_data:
                    species_data[current_species] = []
                continue

            # Line data (no alphabetic chars, matches numeric pattern)
            if current_species and not _RE_HAS_ALPHA.search(line):
                if _RE_LINE_DATA.search(line):
                    cols = line.split()
                    if len(cols) >= 7:
                        ab = float(cols[6])
                        dev = float(cols[7]) if len(cols) > 7 else 0.0
                        if abs(dev) < 10.0:  # Filter obvious outliers
                            species_data[current_species].append(ab)

    # Sigma-clip and compute mean for each species
    results: dict[str, tuple[float, float, int]] = {}
    for species, ab_list in species_data.items():
        if len(ab_list) == 0:
            continue

        ab_arr = np.array(ab_list)
        if len(ab_arr) >= 3:
            clipped = sigma_clip(ab_arr, sigma=1.5, maxiters=1)
            ab_clean = ab_arr[~clipped.mask]
        else:
            ab_clean = ab_arr

        if len(ab_clean) > 0:
            mean_ab = float(np.mean(ab_clean))
            # Uncertainty via uncertainties-style propagation (±0.2 per line)
            std_ab = 0.2 / np.sqrt(len(ab_clean)) if len(ab_clean) > 1 else 0.2
            results[species] = (mean_ab, std_ab, len(ab_clean))

    return results


def _ion_to_element(ion_name: str) -> str:
    """Extract element name from ion name: 'NaI' → 'Na', 'TiII' → 'Ti'."""
    # Find where the Roman numerals start
    for i, c in enumerate(ion_name):
        if c == "I" or c == "V":
            return ion_name[:i]
    return ion_name
