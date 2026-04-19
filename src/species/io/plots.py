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

        # Per-line fit plots (multi-page PDF grid like original SPECIES)
        p = _plot_line_fits(result, output_dir)
        if p:
            plots.append(p)

    # Broadening diagnostics (vsini grid search + observed vs synthetic)
    if result.broadening and result.broadening.per_line:
        p = _plot_broadening(result, output_dir)
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


def _plot_line_fits(result, output_dir: Path) -> Path | None:
    """Multi-page PDF grid of per-line Gaussian fits.

    For each valid line with ``line_data``, plots:
    - Top panel: raw flux + continuum polynomial
    - Bottom panel: normalized flux + Gaussian fit + MC uncertainty envelope
      + EW annotation + detected line markers

    Faithfully ports the original SPECIES ``EWComputation.plot_lines``.
    """
    ew_results = result.ew_results
    if not ew_results:
        return None

    lines_with_data = [r for r in ew_results if r.is_valid and r.line_data is not None]
    if not lines_with_data:
        return None

    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        path = output_dir / f"{result.star_name}_line_fits.pdf"

        ncols = 6
        nrows = int(np.ceil(len(lines_with_data) / ncols))
        if nrows == 0:
            return None

        with PdfPages(str(path)) as pdf:
            # Process in pages of ncols * nrows_per_page
            lines_per_page = ncols * 4  # 4 rows per page
            for page_start in range(0, len(lines_with_data), lines_per_page):
                page_lines = lines_with_data[page_start:page_start + lines_per_page]
                n_on_page = len(page_lines)
                n_rows_page = int(np.ceil(n_on_page / ncols))

                fig, axes = plt.subplots(
                    n_rows_page * 2, ncols,
                    figsize=(6 * ncols, 3 * n_rows_page),
                    squeeze=False,
                )

                for i, ew_r in enumerate(page_lines):
                    d = ew_r.line_data
                    row = (i // ncols) * 2
                    col = i % ncols

                    ax_cont = axes[row, col]
                    ax_line = axes[row + 1, col]

                    wave = d.wave
                    sx, ex = wave[0], wave[-1]

                    # --- Top: raw flux + continuum ---
                    mean_f = np.mean(d.flux_raw)
                    ax_cont.plot(wave, d.flux_raw - mean_f + 1.0, lw=0.5, color="black")
                    ax_cont.plot(wave, d.continuum - mean_f + 1.0, lw=0.5, color="steelblue")
                    ax_cont.set_xlim(sx, ex)
                    ax_cont.tick_params(labelsize="xx-small", labelbottom=False)
                    if col == 0:
                        ax_cont.set_ylabel("Flux", fontsize="x-small")

                    # --- Bottom: normalized + Gaussian fit ---
                    flux_plot = d.flux_norm + 1.0
                    ax_line.plot(wave, flux_plot, lw=0.4, color="black")
                    ax_line.axhline(1.0, ls=":", color="gray", lw=0.5)
                    ax_line.axvline(ew_r.wavelength, color="red", ls=":", lw=0.5)

                    # Full multi-Gaussian model
                    ax_line.plot(wave, d.full_model + 1.0, color="green", lw=0.5)

                    # Target component Gaussian + MC envelope
                    a, m, s = d.gauss_params
                    ea, em, es = d.gauss_errors
                    xfit = np.linspace(sx, ex, 300)
                    yfit = a * np.exp(-(xfit - m) ** 2 / (2.0 * s ** 2))
                    ax_line.plot(xfit, yfit + 1.0, color="steelblue", lw=0.5)

                    # MC envelope (16/50/84 percentiles)
                    n_mc = 200
                    da = np.random.normal(a, max(ea, 1e-10), n_mc)
                    dm = np.random.normal(m, max(em, 1e-10), n_mc)
                    ds = np.random.normal(s, max(es, 1e-10), n_mc)
                    mc_profiles = da[:, None] * np.exp(
                        -(xfit[None, :] - dm[:, None]) ** 2 / (2.0 * ds[:, None] ** 2)
                    )
                    y16, y50, y84 = np.percentile(mc_profiles, [16, 50, 84], axis=0)
                    ax_line.fill_between(xfit, y16 + 1.0, y50 + 1.0,
                                         alpha=0.4, color="orangered", lw=0.1)
                    ax_line.fill_between(xfit, y50 + 1.0, y84 + 1.0,
                                         alpha=0.4, color="orangered", lw=0.1)

                    # Detected line markers
                    for ll in d.detected_lines:
                        ax_line.axvline(ll, color="gray", ls="-", alpha=0.5, lw=0.5)

                    # EW annotation
                    p16_ew, p50_ew, p84_ew = np.percentile(d.ew_dist, [16, 50, 84])
                    sy, ey = ax_line.get_ylim()
                    sy = max(sy, 0.0)
                    ax_line.text(
                        sx + (ex - sx) * 0.03, sy + (ey - sy) * 0.12,
                        f"EW = {p50_ew:.1f}$^{{+{p84_ew - p50_ew:.1f}}}_{{-{p50_ew - p16_ew:.1f}}}$ mÅ",
                        fontsize="xx-small",
                        bbox=dict(facecolor="white", alpha=0.8, edgecolor="None"),
                    )

                    ax_line.set_xlim(sx, ex)
                    ax_line.set_ylim(max(sy, 0.0), min(1.3, ey))
                    ax_line.tick_params(labelsize="xx-small")
                    if col == 0:
                        ax_line.set_ylabel("Norm. Flux", fontsize="x-small")
                    if row + 2 >= n_rows_page * 2:
                        ax_line.set_xlabel("Wavelength", fontsize="x-small")

                # Hide unused axes
                for i in range(n_on_page, n_rows_page * ncols):
                    row = (i // ncols) * 2
                    col = i % ncols
                    if row < axes.shape[0]:
                        axes[row, col].set_visible(False)
                        axes[row + 1, col].set_visible(False)

                fig.subplots_adjust(
                    bottom=0.04, top=0.97, left=0.03, right=0.99,
                    wspace=0.15, hspace=0.25,
                )
                pdf.savefig(fig)
                plt.close(fig)

        return path
    except Exception:
        logger.warning("Line fit plot failed", exc_info=True)
        return None


def _plot_broadening(result, output_dir: Path) -> Path | None:
    """Broadening diagnostic: per-line chi2 grid + observed vs synthetic.

    Two rows per line:
    - Top: observed (black) vs best-fit synthetic (red) spectrum
    - Bottom: chi-squared vs vsini grid with spline minimum

    Ports the original SPECIES ``CalcBroadening.plot_paper``.
    """
    per_line = result.broadening.per_line
    if not per_line:
        return None

    try:
        import matplotlib.pyplot as plt
        from scipy.interpolate import UnivariateSpline

        n = len(per_line)
        fig, axes = plt.subplots(2, n, figsize=(4 * n, 6), squeeze=False)

        for i, ld in enumerate(per_line):
            # Top: observed vs synthetic
            ax = axes[0, i]
            ax.plot(ld.wave_obs, ld.flux_obs, "k-", lw=0.8, label="Observed")
            synth_interp = np.interp(ld.wave_obs, ld.wave_synth, ld.flux_synth)
            ax.plot(ld.wave_obs, synth_interp, "r-", lw=0.8,
                    label=f"Synth (vsini={ld.vsini:.1f})")
            ax.set_title(ld.name, fontsize=9)
            ax.legend(fontsize="xx-small", loc="lower left")
            ax.set_ylabel("Norm. Flux" if i == 0 else "")
            ax.tick_params(labelsize="x-small")

            # Bottom: chi2 vs vsini
            ax = axes[1, i]
            ax.plot(ld.v_grid, ld.chi2, "o-", color="steelblue", ms=3, lw=0.8)
            ax.axvline(ld.vsini, color="red", ls="--", lw=0.8,
                       label=f"vsini = {ld.vsini:.2f}")
            # Spline fit
            if len(ld.v_grid) >= 4:
                try:
                    tck = UnivariateSpline(ld.v_grid, ld.chi2,
                                           k=min(4, len(ld.v_grid) - 1), s=0)
                    vfine = np.linspace(ld.v_grid[0], ld.v_grid[-1], 100)
                    ax.plot(vfine, tck(vfine), "g-", lw=0.6, alpha=0.7)
                except Exception:
                    pass
            ax.set_xlabel("vsini (km/s)", fontsize=8)
            ax.set_ylabel("χ²" if i == 0 else "")
            ax.legend(fontsize="xx-small")
            ax.tick_params(labelsize="x-small")

        fig.suptitle(
            f"{result.star_name} — Broadening "
            f"(vsini = {result.broadening.vsini:.1f} ± {result.broadening.vsini_err:.1f}, "
            f"vmac = {result.broadening.vmac:.1f} ± {result.broadening.vmac_err:.1f} km/s)",
            fontsize=10,
        )
        fig.tight_layout()

        path = output_dir / f"{result.star_name}_broadening.pdf"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception:
        logger.warning("Broadening plot failed", exc_info=True)
        return None
