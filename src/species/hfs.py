"""Hyperfine-structure (HFS) treatment for abundance lines.

Mn I and Cu I lines are split into many hyperfine (and for Cu, isotopic)
components spread over 0.05-0.15 A. Treated as a single line in an EW analysis,
a line of moderate strength is too saturated in the model, so the abundance
needed to reproduce the observed EW is overestimated; for Cu I 5105.5 in the
Sun the error is half a dex. The cure is to synthesise all components together
and match the total EW, which is what MOOG's ``blends`` driver does.

The component table (``data/linelists/hfs_mn_cu.dat``) keeps each line's total
log gf and air wavelength from ``lines_ab.dat`` and takes only the splitting
pattern from the literature; see its header and ``tools/build_hfs_linelist.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# MOOG writes this as the abundance of every blend member except the one it
# assigns the blend's abundance to.
_NO_ABUNDANCE = 999.0
_RE_SPECIES = re.compile(r"Abundance Results for Species\s+(\S+\s*\S*?)\s*(?:\(|\.)")


@dataclass(frozen=True)
class HFSComponent:
    """One hyperfine component of a parent line."""

    wavelength: float  # air, A
    excitation_potential: float  # eV
    loggf: float
    ion: float  # MOOG species code, e.g. 25.0


def _key(wavelength: float) -> float:
    return round(float(wavelength), 3)


def load_hfs_table(path: Path) -> dict[float, list[HFSComponent]]:
    """Read the HFS table, keyed by parent wavelength rounded to 1 mA."""
    table: dict[float, list[HFSComponent]] = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parent, wl, ep, lgf, ion = line.split()[:5]
        table.setdefault(_key(float(parent)), []).append(
            HFSComponent(float(wl), float(ep), float(lgf), float(ion))
        )
    for comps in table.values():
        comps.sort(key=lambda c: c.wavelength)
    return table


def components_for(
    table: dict[float, list[HFSComponent]], wavelength: float
) -> list[HFSComponent] | None:
    """HFS components of the line at ``wavelength``, or None if it is unsplit."""
    return table.get(_key(wavelength))


def write_blends_linelist(
    path: Path, groups: list[tuple[list[HFSComponent], float]], title: str = "hfs"
) -> None:
    """Write a MOOG ``blends`` line list.

    Each group is one observed feature: its components and the measured EW in
    mA. MOOG reads a group as a line with a positive wavelength carrying the EW,
    followed by lines with negative wavelengths that belong to it.
    """
    with open(path, "w") as f:
        f.write(f"{title}\n")
        for comps, ew in groups:
            for k, c in enumerate(comps):
                wl = c.wavelength if k == 0 else -c.wavelength
                width = ew if k == 0 else 0.0
                # MOOG's formatted read is (7e10.3): wl, id, EP, loggf, damp, d0, EW
                f.write(
                    f"{wl:10.4f}{c.ion:10.1f}{c.excitation_potential:10.3f}"
                    f"{c.loggf:10.4f}{'':20s}{width:10.2f}\n"
                )


def parse_blends_summary(path: Path) -> dict[str, list[tuple[float, float]]]:
    """Parse a MOOG ``blends`` summary.

    Returns ion name (``"MnI"``) -> list of ``(component_wavelength, A(X))``,
    one entry per fitted blend. MOOG assigns each blend's abundance to its
    strongest component and writes 999.99 for the rest; those are skipped.
    """
    out: dict[str, list[tuple[float, float]]] = {}
    current: str | None = None
    for raw in Path(path).read_text().splitlines():
        m = _RE_SPECIES.search(raw)
        if m:
            current = m.group(1).replace(" ", "")
            out.setdefault(current, [])
            continue
        if current is None:
            continue
        cols = raw.split()
        if len(cols) < 7:
            continue
        try:
            wl, ab = float(cols[0]), float(cols[6])
        except ValueError:
            continue
        if np.isfinite(ab) and ab < _NO_ABUNDANCE:
            out[current].append((wl, ab))
    return out
