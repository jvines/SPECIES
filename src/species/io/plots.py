"""Diagnostic plot generation for SPECIES analysis results.

Decoupled from computation — takes an AnalysisResult and generates
publication-quality diagnostic plots.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

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
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    plots: list[Path] = []

    if result.ccf_result is not None:
        p = _plot_ccf(result.ccf_result, result.star_name, output_dir)
        if p:
            plots.append(p)

    if result.params.converged and result.ew_results:
        p = _plot_abundances(result, output_dir)
        if p:
            plots.append(p)

    return plots


def _plot_ccf(ccf_result, star_name: str, output_dir: Path) -> Path | None:
    """Plot CCF and Gaussian fit."""
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ccf_result.ccf_velocities, ccf_result.ccf_values, "k-", lw=0.8)

        # Overlay Gaussian fit
        a, mu, sigma, offset = ccf_result.gaussian_params
        v_fine = np.linspace(mu - 5 * abs(sigma), mu + 5 * abs(sigma), 200)
        fit = offset + a * np.exp(-(v_fine - mu) ** 2 / (2 * sigma ** 2))
        ax.plot(v_fine, fit, "r-", lw=1.2, label=f"RV = {mu:.2f} km/s")

        ax.set_xlabel("Velocity (km/s)")
        ax.set_ylabel("CCF")
        ax.set_xlim(mu - 5 * abs(sigma), mu + 5 * abs(sigma))
        ax.legend()
        ax.set_title(f"{star_name} — CCF")

        path = output_dir / f"{star_name}_ccf.pdf"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception:
        logger.debug("CCF plot failed", exc_info=True)
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
        logger.debug("Abundance plot failed", exc_info=True)
        return None
