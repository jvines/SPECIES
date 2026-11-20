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
from dataclasses import dataclass, replace
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
from species.moog.parser import AbfindResult
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
    fe_abfind: AbfindResult | None = None,
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
    fe_abfind
        The solver's final abfind at ``params``. When given, the Fe I and Fe II
        rows come from its lines and EWs -- the ones [Fe/H] was fitted on, with
        solar-calibrated log gf -- rather than from the Fe lines of the
        abundance list, whose log gf sit a median 0.13 dex below linelist.dat's
        for the same transitions and put the table's Fe above the fitted [Fe/H].

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
    # One entry per feature the EW fit can resolve: unresolved same-species
    # components (synthesis satellites, fine-structure pairs) are folded into
    # the strongest one instead of each being handed the whole feature's EW.
    single, ab_linelist = fold_unresolved_blends(single, ab_linelist, params.teff)
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

    # Fe on the solver's own lines and EWs
    if fe_abfind is not None:
        for ion_name in ("FeI", "FeII"):
            summary = _summarise([lr.abundance for lr in fe_abfind.lines.get(ion_name, [])])
            if summary is not None:
                raw_abundances[ion_name] = summary
            else:
                raw_abundances.pop(ion_name, None)

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
    return mean_ab, line_scatter_error(ab_clean), len(ab_clean)


# Per-line dispersion assumed when it cannot be measured (fewer than 3 lines),
# and the floor on a measured one: a handful of lines can agree by chance.
_SIGMA_LINE_PRIOR = 0.20  # dex
_SIGMA_LINE_FLOOR = 0.05  # dex


def line_scatter_error(ab: np.ndarray) -> float:
    """Error of a mean abundance: measured line-to-line scatter / sqrt(n).

    The scatter (sample std, ddof=1) is floored at 0.05 dex. With fewer than
    three lines it is not estimable, and the old fixed 0.2 dex per line is kept
    as a floor instead.
    """
    ab = np.asarray(ab, dtype=float)
    n = len(ab)
    if n == 0:
        return float("nan")
    scatter = float(np.std(ab, ddof=1)) if n > 1 else 0.0
    floor = _SIGMA_LINE_FLOOR if n >= 3 else _SIGMA_LINE_PRIOR
    return max(scatter, floor) / np.sqrt(n)


# Same-species entries closer than this are one feature to the EW fit, which
# treats any line found within 0.10 A of the target as the target.
_BLEND_TOL = 0.10  # A


def fold_unresolved_blends(
    ew_results: list[EWResult],
    linelist: LineList,
    teff: float,
    tol: float = _BLEND_TOL,
) -> tuple[list[EWResult], LineList]:
    """Collapse unresolved same-species components onto the strongest one.

    lines_ab.dat carries synthesis-style component lists: e.g. Fe I 5543.936
    (log gf -1.14) next to a 5543.974 satellite at log gf -6.06, or Fe II
    5234.623 next to a 10.4 eV entry. Each entry was measured on its own, the
    Gaussian fit found the same feature for both, and MOOG returned +1 to +9
    dex for the satellites -- enough of which survived the 1.5-sigma clip to
    lift the Sun's Fe I by 0.27 dex.

    Entries of one species whose wavelengths chain within ``tol`` form a group.
    The group's EW is the one measured at its strongest component (line
    strength log gf - theta*chi, theta = 5040/Teff), or failing that any valid
    EW in the group, and that component's log gf becomes the group's effective
    value, ``log10(sum_i gf_i 10^(-theta (chi_i - chi_0)))``: exact for weak
    lines of one ion, and a negligible change when the rest are satellites.

    Returns the EW results (one per group, at the strongest component's
    wavelength) and a copy of the line list with the effective log gf.
    """
    theta = 5040.0 / teff
    wl_all = np.asarray(linelist.wavelengths, dtype=float)
    loggf = np.array(linelist.loggf, dtype=float)

    rows = []
    for r in ew_results:
        idx = np.where(wl_all == r.wavelength)[0]
        if len(idx) == 1:
            rows.append((int(idx[0]), r))
    rows.sort(key=lambda t: (float(linelist.ionization[t[0]]), wl_all[t[0]]))

    groups: list[list[tuple[int, EWResult]]] = []
    for row in rows:
        if groups:
            last_idx = groups[-1][-1][0]
            if (
                float(linelist.ionization[last_idx]) == float(linelist.ionization[row[0]])
                and wl_all[row[0]] - wl_all[last_idx] <= tol
            ):
                groups[-1].append(row)
                continue
        groups.append([row])

    folded: list[EWResult] = []
    for group in groups:
        if len(group) == 1:
            folded.append(group[0][1])
            continue
        ix = np.array([i for i, _ in group])
        chi = np.asarray(linelist.excitation_potential, dtype=float)[ix]
        gf = np.asarray(linelist.loggf, dtype=float)[ix]
        k = int(np.argmax(gf - theta * chi))
        dom = int(ix[k])
        loggf[dom] = float(np.log10(np.sum(10.0 ** (gf - theta * (chi - chi[k])))))
        ew = group[k][1]
        if not ew.is_valid:
            ew = next((r for _, r in group if r.is_valid), ew)
        folded.append(replace(ew, wavelength=float(wl_all[dom])))
        logger.debug(
            "Folded %d entries near %.3f A into one line (log gf %.3f -> %.3f)",
            len(group), wl_all[dom], gf[k], loggf[dom],
        )

    return folded, replace(linelist, loggf=loggf)


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
