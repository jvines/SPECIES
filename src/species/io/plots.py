"""Diagnostic plot generation for SPECIES analysis results.

Ports the original SPECIES diagnostic plots (Atmos.plot_output_file,
EWComputation.plot_lines) to work with the v4 AnalysisResult API.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.stats import linregress

logger = logging.getLogger(__name__)


def plot_diagnostics(
    result,  # AnalysisResult (imported lazily to avoid circular)
    output_dir: Path,
) -> list[Path]:
    """Generate all diagnostic plots for an analysis result.

    Parameters
    ----------
    result
        An ``AnalysisResult`` from ``Analyzer.run()``.
    output_dir
        Directory to write plots to.

    Returns
    -------
    list[Path]
        Paths to generated plot files.
    """
    import matplotlib
    matplotlib.use("Agg")

    output_dir.mkdir(parents=True, exist_ok=True)
    plots: list[Path] = []

    # MOOG Fe diagnostic (3-panel: abundance vs EP, vs REW, vs wavelength)
    if result.abfind is not None:
        p = _plot_moog_diagnostics(result, output_dir)
        if p:
            plots.append(p)

    # Abundance pattern ([X/H] for all elements)
    if result.params.converged and result.abundances:
        p = _plot_abundances(result, output_dir)
        if p:
            plots.append(p)

    # EW measurements with MC uncertainties
    if result.ew_results:
        p = _plot_ew_measurements(result, output_dir)
        if p:
            plots.append(p)

    return plots


def _plot_moog_diagnostics(result, output_dir: Path) -> Path | None:
    """3-panel MOOG diagnostic: Fe abundance vs EP, vs REW, Fe I/II vs wavelength.

    Faithfully reproduces the original SPECIES ``Atmos.plot_output_file``.
    """
    abfind = result.abfind
    if abfind is None:
        return None

    fe_i_lines = abfind.lines.get("FeI", [])
    fe_ii_lines = abfind.lines.get("FeII", [])

    if not fe_i_lines:
        return None

    try:
        import matplotlib.pyplot as plt

        # Extract Fe I arrays
        ep_i = np.array([l.ep for l in fe_i_lines])
        rw_i = np.array([l.log_rw for l in fe_i_lines])
        ab_i = np.array([l.abundance for l in fe_i_lines])
        wl_i = np.array([l.wavelength for l in fe_i_lines])

        # Fe II arrays
        ab_ii = np.array([l.abundance for l in fe_ii_lines]) if fe_ii_lines else np.array([])
        wl_ii = np.array([l.wavelength for l in fe_ii_lines]) if fe_ii_lines else np.array([])

        fig, ax = plt.subplots(3, 1, figsize=(10, 7))

        # --- Panel 0: Fe I abundance vs EP ---
        isort_ep = np.argsort(ep_i)
        ax[0].plot(ep_i, ab_i, ls="None", marker="o", color="steelblue", ms=5)
        if len(ep_i) > 2:
            slope, intercept, _, _, slope_err = linregress(ep_i[isort_ep], ab_i[isort_ep])
            ax[0].plot(ep_i[isort_ep], slope * ep_i[isort_ep] + intercept,
                       color="red", label=f"EP slope = {slope:.4f} ± {slope_err:.4f}")
        ax[0].set_xlabel("Excitation Potential (eV)")
        ax[0].set_ylabel("Fe I abundance")
        ax[0].legend(loc="upper left", fontsize="x-small")

        # --- Panel 1: Fe I abundance vs REW ---
        isort_rw = np.argsort(rw_i)
        ax[1].plot(rw_i, ab_i, ls="None", marker="o", color="steelblue", ms=5)
        if len(rw_i) > 2:
            slope, intercept, _, _, slope_err = linregress(rw_i[isort_rw], ab_i[isort_rw])
            ax[1].plot(rw_i[isort_rw], slope * rw_i[isort_rw] + intercept,
                       color="red", label=f"REW slope = {slope:.4f} ± {slope_err:.4f}")
        ax[1].set_xlabel("Reduced Equivalent Width")
        ax[1].set_ylabel("Fe I abundance")
        ax[1].legend(loc="upper left", fontsize="x-small")

        # --- Panel 2: Fe I/II abundance vs wavelength ---
        ax[2].plot(wl_i, ab_i, ls="None", marker="o", color="steelblue", ms=5)
        ax[2].axhline(np.mean(ab_i), color="steelblue",
                      label=f"Fe I = {np.mean(ab_i):.4f}")
        if len(ab_ii) > 0:
            ax[2].plot(wl_ii, ab_ii, ls="None", marker="o", color="orange", ms=5)
            ax[2].axhline(np.mean(ab_ii), color="orange",
                          label=f"Fe II = {np.mean(ab_ii):.4f}"
                                f"\ndiff = {np.mean(ab_i) - np.mean(ab_ii):.4f}")
        ax[2].set_xlabel("Wavelength (Å)")
        ax[2].set_ylabel("Abundance")
        ax[2].legend(loc="upper left", ncol=2, fontsize="x-small")

        fig.suptitle(
            f"{result.star_name} — "
            f"Teff={result.params.teff:.0f} K, "
            f"log g={result.params.logg:.2f}, "
            f"[Fe/H]={result.params.feh:.3f}, "
            f"vt={result.params.vt:.2f}",
            fontsize=10,
        )
        fig.subplots_adjust(hspace=0.35, left=0.08, right=0.95, top=0.93, bottom=0.08)

        path = output_dir / f"{result.star_name}_moog_diagnostics.pdf"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception:
        logger.warning("MOOG diagnostic plot failed", exc_info=True)
        return None


def _plot_abundances(result, output_dir: Path) -> Path | None:
    """Plot [X/H] vs element for all measured abundances."""
    if not result.abundances:
        return None

    try:
        import matplotlib.pyplot as plt

        elements = sorted(result.abundances.keys())
        ab_values = [result.abundances[e].abundance for e in elements]
        ab_errors = [result.abundances[e].uncertainty for e in elements]

        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(elements))
        ax.errorbar(x, ab_values, yerr=ab_errors, fmt="o", capsize=3,
                     color="steelblue", markersize=6)
        ax.axhline(0, ls="--", color="gray", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(elements, rotation=45, ha="right")
        ax.set_ylabel("[X/H]")
        ax.set_title(
            f"{result.star_name} — "
            f"T={result.params.teff:.0f} K, "
            f"logg={result.params.logg:.2f}, "
            f"[Fe/H]={result.params.feh:.3f}"
        )

        path = output_dir / f"{result.star_name}_abundances.pdf"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception:
        logger.warning("Abundance plot failed", exc_info=True)
        return None


def _plot_ew_measurements(result, output_dir: Path) -> Path | None:
    """Plot EW measurements with MC uncertainties for all valid lines."""
    ew_results = result.ew_results
    if not ew_results:
        return None

    valid = [r for r in ew_results if r.is_valid]
    if not valid:
        return None

    try:
        import matplotlib.pyplot as plt

        valid.sort(key=lambda r: r.wavelength)
        wls = [r.wavelength for r in valid]
        medians = [r.ew_median for r in valid]
        err_plus = [r.ew_err_plus for r in valid]
        err_minus = [r.ew_err_minus for r in valid]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.errorbar(wls, medians, yerr=[err_minus, err_plus],
                    fmt="o", ms=3, capsize=2, color="steelblue",
                    ecolor="gray", alpha=0.8, lw=0.8)
        ax.set_xlabel("Wavelength (Å)")
        ax.set_ylabel("EW (mÅ)")
        ax.set_title(f"{result.star_name} — EW Measurements ({len(valid)} lines)")

        path = output_dir / f"{result.star_name}_ew_measurements.pdf"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception:
        logger.warning("EW measurement plot failed", exc_info=True)
        return None
