"""Chemical abundance determination for non-iron elements.

Ported from Atmos.py ``runMOOG_ab()`` and CalcParams.py ``calc_ab()``.
Runs MOOG in abfind mode with a multi-element line list to determine
abundances of Na, Mg, Al, Si, Ca, Ti, Cr, Mn, Ni, Cu, Zn, etc. Lines with
hyperfine structure (Mn I, Cu I) are run through MOOG blends with their HFS
components instead; see :mod:`species.hfs`.

The flow:
1. Measure EWs for the abundance linelist (same Gaussian fitting as for Fe)
2. Filter into MOOG-format linelist
3. Run MOOG abfind at the converged atmospheric parameters, and MOOG blends
   for the HFS lines
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
from species.ew import (
    EWResult,
    LineList,
    create_moog_linelist,
    measure_equivalent_widths,
    select_moog_lines,
)
from species.hfs import (
    HFSComponent,
    components_for,
    load_hfs_table,
    parse_blends_summary,
    write_blends_linelist,
)
from species.moog.atmosphere_grid import AtmosphereGrid
from species.moog.wrapper import MOOGError, MOOGRunner, MOOGSession

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

    # 2. Create MOOG-format linelists: abfind for single lines, blends for
    #    lines with hyperfine structure
    ab_linelist = LineList.from_file(config.linelist_ab_path)
    hfs_table = _load_hfs(config)
    single = [r for r in ew_results if components_for(hfs_table, r.wavelength) is None]
    split = [r for r in ew_results if components_for(hfs_table, r.wavelength) is not None]
    moog_ab_path = config.output_dir / "moog_ab_linelist.txt"
    create_moog_linelist(single, ab_linelist, moog_ab_path, **_EW_CUTS)
    hfs_groups = _hfs_groups(split, ab_linelist, hfs_table)

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

            # Hyperfine-split lines: one MOOG blends run per element, at the
            # same model. These replace any abfind result for the ion.
            for ion_name, per_line in _run_hfs_blends(session, hfs_groups).items():
                summary = _summarise(per_line)
                if summary is not None:
                    raw_abundances[ion_name] = summary

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
        summary = _summarise(ab_list)
        if summary is not None:
            results[species] = summary

    return results


def _summarise(ab_list: list[float]) -> tuple[float, float, int] | None:
    """Sigma-clipped mean of per-line abundances: ``(mean, error, n_lines)``."""
    if len(ab_list) == 0:
        return None

    ab_arr = np.array(ab_list, dtype=float)
    if len(ab_arr) >= 3:
        clipped = sigma_clip(ab_arr, sigma=1.5, maxiters=1)
        ab_clean = ab_arr[~clipped.mask]
    else:
        ab_clean = ab_arr

    if len(ab_clean) == 0:
        return None
    mean_ab = float(np.mean(ab_clean))
    # Uncertainty via uncertainties-style propagation (±0.2 per line)
    std_ab = 0.2 / np.sqrt(len(ab_clean)) if len(ab_clean) > 1 else 0.2
    return mean_ab, std_ab, len(ab_clean)


# ---------------------------------------------------------------------------
# Hyperfine structure: MOOG blends
# ---------------------------------------------------------------------------

# EW quality cuts for the abundance lines, shared by the abfind and blends paths
_EW_CUTS = {"ew_min": 5.0, "ew_max": 200.0, "max_relative_error": 0.8}

# A blends abundance is credited to the line whose parent wavelength is within
# this distance of the component MOOG reports it on.
_HFS_MATCH_TOL = 0.2  # A


def _load_hfs(config: Settings) -> dict[float, list[HFSComponent]]:
    path = config.linelist_hfs_path
    if path is None or not Path(path).exists():
        logger.warning(
            "HFS table %s not found: Mn I and Cu I lines will be treated as single "
            "lines, which overestimates their abundances", path,
        )
        return {}
    return load_hfs_table(path)


@dataclass
class _HFSGroup:
    """One measured HFS line ready for MOOG blends."""

    wavelength: float  # parent wavelength in lines_ab.dat
    atom: float  # MOOG species code of the parent, e.g. 25.0
    ew: float  # mA
    components: list[HFSComponent]


def _hfs_groups(
    ew_results: list[EWResult],
    linelist: LineList,
    hfs_table: dict[float, list[HFSComponent]],
) -> list[_HFSGroup]:
    """Apply the abfind EW cuts to the HFS lines and attach their components."""
    groups = []
    for idx, ew in select_moog_lines(ew_results, linelist, **_EW_CUTS):
        wl = float(linelist.wavelengths[idx])
        comps = components_for(hfs_table, wl)
        if comps:
            groups.append(_HFSGroup(wl, float(linelist.ionization[idx]), ew, comps))
    return groups


def _run_hfs_blends(
    session: MOOGSession, groups: list[_HFSGroup]
) -> dict[str, list[float]]:
    """Run MOOG blends per element; return ion name -> per-line A(X).

    MOOG blends aborts the whole run when one group fails to converge, so if a
    run returns fewer lines than it was given, the lines are retried one per
    run and only the failing ones are lost.
    """
    by_atom: dict[float, list[_HFSGroup]] = {}
    for g in groups:
        by_atom.setdefault(g.atom, []).append(g)

    results: dict[str, list[float]] = {}
    for atom, atom_groups in sorted(by_atom.items()):
        atom_groups.sort(key=lambda g: g.wavelength)
        found = _blends_once(session, atom, atom_groups)
        if found is None or sum(len(v) for v in found.values()) < len(atom_groups):
            found = {}
            for g in atom_groups:
                one = _blends_once(session, atom, [g])
                if one is None:
                    logger.warning("MOOG blends failed for HFS line %.3f", g.wavelength)
                    continue
                for ion_name, per_line in one.items():
                    found.setdefault(ion_name, []).extend(per_line)
        for ion_name, per_line in found.items():
            results.setdefault(ion_name, []).extend(per_line)
    return results


def _blends_once(
    session: MOOGSession, atom: float, groups: list[_HFSGroup]
) -> dict[str, list[float]] | None:
    linelist = session.model_path.parent / f"blends_{int(atom)}.txt"
    write_blends_linelist(linelist, [(g.components, g.ew) for g in groups])
    try:
        summary = session.run_blends(linelist, atom)
    except MOOGError as e:
        logger.debug("%s", e)
        return None

    parents = np.array([g.wavelength for g in groups])
    out: dict[str, list[float]] = {}
    for ion_name, per_line in parse_blends_summary(summary).items():
        for comp_wl, ab in per_line:
            if np.min(np.abs(parents - comp_wl)) <= _HFS_MATCH_TOL:
                out.setdefault(ion_name, []).append(ab)
    return out


def _ion_to_element(ion_name: str) -> str:
    """Extract element name from ion name: 'NaI' → 'Na', 'TiII' → 'Ti'."""
    # Find where the Roman numerals start
    for i, c in enumerate(ion_name):
        if c == "I" or c == "V":
            return ion_name[:i]
    return ion_name
