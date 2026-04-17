"""Generate MOOG parameter (.par) files for abfind and synth modes.

Ported from CalcParams.py and CalcBroadening.py Driver.create_batch().
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent


def write_abfind_par(
    output_path: Path,
    *,
    model_path: Path,
    linelist_path: Path,
    summary_out: Path,
    standard_out: Path,
    smoothed_out: Path | None = None,
    atmosphere: int = 1,
    molecules: int = 1,
    trudamp: int = 1,
    lines: int = 1,
    flux_int: int = 0,
    damping: int = 1,
    freeform: int = 0,
    plot: int = 0,
) -> Path:
    """Write a MOOG abfind parameter file.

    Parameters
    ----------
    output_path
        Where to write the ``.par`` file.
    model_path
        Path to the interpolated atmosphere model (``.atm``).
    linelist_path
        Path to the MOOG-format line list.
    summary_out
        Path for MOOG summary output (``*_out.test``).
    standard_out
        Path for MOOG standard output.
    smoothed_out
        Optional path for smoothed output.

    Returns
    -------
    Path
        The ``output_path`` for chaining.
    """
    smooth_line = f"smoothed_out  '{smoothed_out}'" if smoothed_out else ""

    content = dedent(f"""\
        abfind
        terminal       'x11'
        standard_out   '{standard_out}'
        summary_out    '{summary_out}'
        {smooth_line}
        model_in       '{model_path}'
        lines_in       '{linelist_path}'
        atmosphere     {atmosphere}
        molecules      {molecules}
        trudamp        {trudamp}
        lines          {lines}
        flux/int       {flux_int}
        damping        {damping}
        freeform       {freeform}
        plot           {plot}
    """).strip() + "\n"

    # Remove blank lines from optional fields
    lines_out = [ln for ln in content.splitlines() if ln.strip()]
    output_path.write_text("\n".join(lines_out) + "\n")
    return output_path


def write_synth_par(
    output_path: Path,
    *,
    model_path: Path,
    linelist_path: Path,
    observed_path: Path | None = None,
    summary_out: Path,
    standard_out: Path,
    smoothed_out: Path,
    wave_start: float,
    wave_end: float,
    wave_step: float = 0.02,
    opacity_range: float = 2.0,
    atmosphere: int = 1,
    molecules: int = 1,
    trudamp: int = 1,
    lines: int = 1,
    flux_int: int = 0,
    damping: int = 1,
    freeform: int = 0,
    plot: int = 0,
    n_species: int = 1,
    species_ids: list[float] | None = None,
    abundances: list[float] | None = None,
) -> Path:
    """Write a MOOG synth parameter file.

    Parameters
    ----------
    output_path
        Where to write the ``.par`` file.
    wave_start, wave_end
        Synthesis wavelength range in Angstroms.
    wave_step
        Wavelength step size.
    n_species
        Number of species to vary abundances for.
    species_ids
        Atomic species IDs (e.g. ``[26.0]`` for Fe I).
    abundances
        Abundance offsets for each species.

    Returns
    -------
    Path
        The ``output_path`` for chaining.
    """
    observed_line = f"observed_in    '{observed_path}'" if observed_path else ""

    # Build par file matching MOOG's expected format exactly.
    # Order matters: abundances BEFORE synlimits, species IDs as integers.
    lines_out = [
        "synth",
        f"standard_out  '{standard_out}'",
        f"summary_out   '{summary_out}'",
        f"smoothed_out  '{smoothed_out}'",
    ]
    if observed_path:
        lines_out.append(f"observed_in   '{observed_path}'")
    lines_out.extend([
        f"model_in      '{model_path}'",
        f"lines_in      '{linelist_path}'",
        f"atmosphere    {atmosphere}",
        f"molecules     {molecules}",
        f"trudamp       {trudamp}",
        f"lines         {lines}",
        f"flux/int      {flux_int}",
        f"damping       {damping}",
        f"freeform      {freeform}",
        f"plot          3",
    ])

    # Abundances block (must come before synlimits)
    if species_ids and abundances and len(species_ids) == len(abundances):
        lines_out.append(f"abundances    {n_species}  1")
        for sid, ab in zip(species_ids, abundances):
            lines_out.append(f"   {int(sid)} {ab:.6f}")

    lines_out.append("isotopes      0  1")
    lines_out.append("synlimits")
    lines_out.append(f" {wave_start:.2f} {wave_end:.2f} {wave_step:.4f} {opacity_range:.1f}")

    # MOOG requires obspectrum + plotpars to produce smoothed_out.
    # plot 3 = write smoothed output to file.
    # obspectrum 5 = no observed spectrum overlay.
    # plotpars 1 = smoothing parameters follow (no extra smoothing;
    # the caller applies vmac/vsini broadening in numpy).
    lines_out.append("obspectrum    5")
    lines_out.append("plotpars      1")
    lines_out.append(f" {wave_start:.2f} {wave_end:.2f} 0.05 1.05")
    lines_out.append(" 0.0000  0.0000  0.000  1.000")
    lines_out.append(" gm  0.000  0.0  0.0  0.00  0.0")

    output_path.write_text("\n".join(lines_out) + "\n")
    return output_path
