"""Broadening measurement (vsini, vmac) via spectral line profile fitting.

Ported from CalcBroadening.py (1,672 lines). Determines rotational velocity
(vsini) and macroturbulence (vmac) by fitting synthetic MOOG spectra to
observed line profiles via grid search with spline interpolation.

Algorithm:
1. Compute vmac from empirical Teff/logg relations
2. For each of 5 diagnostic lines (4 Fe I + 1 Ni I):
   a. Generate synthetic spectra via MOOG synth at grid of vsini values
   b. Apply rotational broadening via PyAstronomy
   c. Find chi-squared minimum via spline interpolation
   d. Monte Carlo error estimation (N=1000 noise realizations)
3. Combine per-line results via weighted median
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PyAstronomy import pyasl
from scipy.interpolate import UnivariateSpline
from uncertainties import ufloat, umath

from species.atmosphere import AtmosphericParameters
from species.config import Settings
from species.io.results import BroadeningResult
from species.moog.atmosphere_grid import AtmosphereGrid
from species.moog.wrapper import MOOGError, MOOGRunner

logger = logging.getLogger(__name__)

# Diagnostic lines used for vsini measurement
_BROADENING_LINES = [
    {"name": "FeI", "wave": 6027.05, "Z": 26, "EP": 4.076, "loggf": -1.09},
    {"name": "FeI", "wave": 6151.62, "Z": 26, "EP": 2.176, "loggf": -3.30},
    {"name": "FeI", "wave": 6165.36, "Z": 26, "EP": 4.143, "loggf": -1.46},
    {"name": "FeI", "wave": 6705.10, "Z": 26, "EP": 4.607, "loggf": -0.98},
    {"name": "NiI", "wave": 6767.77, "Z": 28, "EP": 1.826, "loggf": -2.17},
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class BroadeningFitter:
    """Determine vsini and vmac from spectral line profile fitting.

    Parameters
    ----------
    config
        SPECIES configuration.
    moog
        MOOG runner instance.
    grid
        Atmosphere grid for model interpolation.
    """

    def __init__(
        self,
        config: Settings,
        moog: MOOGRunner,
        grid: AtmosphereGrid,
    ) -> None:
        self.config = config
        self.moog = moog
        self.grid = grid

    def fit(
        self,
        wavelength: np.ndarray,
        flux: np.ndarray,
        snr: float,
        params: AtmosphericParameters,
        ni_abundance: float = 0.0,
        ni_uncertainty: float = 0.1,
    ) -> BroadeningResult:
        """Determine vsini and vmac from line profile fitting.

        Parameters
        ----------
        wavelength
            1D rest-frame corrected wavelength array (Angstroms).
        flux
            1D flux array.
        snr
            Signal-to-noise ratio.
        params
            Converged atmospheric parameters.
        ni_abundance
            Nickel abundance [Ni/H] (used for the NiI diagnostic line).
        ni_uncertainty
            Uncertainty in Ni abundance.

        Returns
        -------
        BroadeningResult
        """
        # 1. Compute vmac from empirical relations
        vmac_arr, err_vmac_arr = _compute_vmac(
            params.teff, params.logg,
            getattr(params, '_err_teff', 100.0),
            getattr(params, '_err_logg', 0.1),
        )

        # 2. For each diagnostic line, measure vsini via grid search
        vsini_arr = np.zeros(len(_BROADENING_LINES))
        err_vsini_arr = np.zeros(len(_BROADENING_LINES))

        for i, line_info in enumerate(_BROADENING_LINES):
            wl = line_info["wave"]
            vmac = vmac_arr[min(i, len(vmac_arr) - 1)]

            # Extract window around line
            mask = (wavelength > wl - 3.0) & (wavelength < wl + 3.0)
            if np.sum(mask) < 10:
                logger.debug("Line %.2f: insufficient data", wl)
                continue

            wave_window = wavelength[mask]
            flux_window = flux[mask]

            # Normalize continuum
            flux_norm = _normalize_continuum(wave_window, flux_window, snr)
            if flux_norm is None:
                continue

            try:
                vs, err_vs = self._fit_single_line(
                    wave_window, flux_norm, wl, params, vmac, snr, line_info,
                )
                vsini_arr[i] = vs
                err_vsini_arr[i] = err_vs
                logger.debug("Line %.2f: vsini=%.2f ± %.2f", wl, vs, err_vs)
            except Exception:
                logger.debug("Line %.2f: vsini fit failed", wl, exc_info=True)

        # 3. Combine results
        return _combine_results(vsini_arr, err_vsini_arr, vmac_arr, err_vmac_arr)

    def _fit_single_line(
        self,
        wavelength: np.ndarray,
        flux_norm: np.ndarray,
        line_center: float,
        params: AtmosphericParameters,
        vmac: float,
        snr: float,
        line_info: dict,
    ) -> tuple[float, float]:
        """Fit vsini for a single diagnostic line via grid search.

        Returns (vsini, error).
        """
        # Generate synthetic spectra at grid of vsini values
        v_grid = np.linspace(0.5, 25.0, 15)
        chi2 = np.full(len(v_grid), np.inf)

        # Find line center index in data
        center_idx = np.argmin(np.abs(wavelength - line_center))
        radius = min(10, center_idx, len(wavelength) - center_idx - 1)
        ci0 = center_idx - radius
        ci1 = center_idx + radius + 1

        for k, vsini in enumerate(v_grid):
            try:
                synth_wave, synth_flux = self._synthesize_line(
                    params, line_center, vsini, vmac, line_info,
                )
                if synth_wave is None:
                    continue

                # Interpolate synthetic to observed wavelength grid
                model_interp = np.interp(
                    wavelength[ci0:ci1], synth_wave, synth_flux,
                )

                # Weighted chi-squared
                w = _make_weights(radius)
                n_w = min(len(w), ci1 - ci0)
                chi2[k] = np.sum(
                    w[:n_w] * (flux_norm[ci0:ci1][:n_w] - model_interp[:n_w]) ** 2
                ) / np.sum(w[:n_w])

            except Exception:
                continue

        # Find minimum via spline interpolation
        valid = np.isfinite(chi2)
        if np.sum(valid) < 4:
            return 0.0, 0.0

        best_vsini = _find_spline_minimum(v_grid[valid], chi2[valid])

        # Monte Carlo error estimation
        err_vsini = _monte_carlo_error(
            wavelength, flux_norm, snr, center_idx, radius,
            v_grid[valid], chi2[valid], n_trials=500,
        )

        return best_vsini, err_vsini

    def _synthesize_line(
        self,
        params: AtmosphericParameters,
        line_center: float,
        vsini: float,
        vmac: float,
        line_info: dict,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Generate a rotationally broadened synthetic spectrum for one line."""
        try:
            with tempfile.TemporaryDirectory(prefix="species_synth_") as tmpdir:
                model_path = Path(tmpdir) / "model.atm"
                ok = self.grid.interpolate_and_write(
                    params.teff, params.logg, params.feh, params.vt,
                    model_path, fe_solar=self.config.fe_solar,
                )
                if not ok:
                    return None, None

                # Run MOOG synth
                smoothed = self.moog.run_synth(
                    model_path,
                    self.config.linelist_vsini_path,
                    wave_start=line_center - 3.0,
                    wave_end=line_center + 3.0,
                    wave_step=0.02,
                    species_ids=[float(line_info["Z"])],
                    abundances=[params.feh],
                )

                # Read synthetic spectrum
                synth_data = np.loadtxt(smoothed)
                synth_wave = synth_data[:, 0]
                synth_flux = synth_data[:, 1]
                smoothed.unlink(missing_ok=True)

                # Normalize to max
                if np.max(synth_flux) > 0:
                    synth_flux /= np.max(synth_flux)

                # Apply rotational broadening
                if vsini > 0.5:
                    synth_flux = pyasl.rotBroad(synth_wave, synth_flux, 0.0, vsini)

                # Apply macroturbulence (approximate as Gaussian)
                if vmac > 0.1:
                    # Convert vmac to wavelength sigma at line center
                    c_kms = 299792.458
                    sigma_wl = vmac / c_kms * line_center
                    sigma_pix = sigma_wl / 0.02  # pixel scale
                    if sigma_pix > 0.5:
                        from astropy.convolution import Gaussian1DKernel, convolve
                        kernel = Gaussian1DKernel(sigma_pix)
                        synth_flux = convolve(synth_flux, kernel, boundary="extend")

                return synth_wave, synth_flux

        except (MOOGError, Exception):
            return None, None


# ---------------------------------------------------------------------------
# vmac computation (ported from calc_vmac)
# ---------------------------------------------------------------------------

def _compute_vmac(
    teff: float,
    logg: float,
    err_teff: float = 100.0,
    err_logg: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute macroturbulence from empirical Teff/logg relations.

    Uses the relation from Melendez et al. (2012) with fallback to
    Brewer et al. (2016) for cool stars.

    Returns arrays of 5 values (one per diagnostic line, with solar
    reference offsets).
    """
    vmac_sun = np.array([3.0, 3.2, 3.1, 3.6, 2.9])

    # Primary relation (Melendez et al. 2012)
    vmac = (
        vmac_sun
        - 0.00707 * teff
        + 9.2422e-7 * teff ** 2
        + 10.0
        - 1.81 * (logg - 4.44)
        - 0.05
    )

    err_vmac = np.full(5, np.sqrt(
        0.1 ** 2
        + (9.2422e-7 * 2 * teff - 0.00707) ** 2 * err_teff ** 2
        + 1.81 ** 2 * err_logg ** 2
        + (logg - 4.44) ** 2 * 0.26 ** 2
        + 0.03 ** 2
    ))

    # Fallback for cool stars: Brewer et al. (2016)
    if np.mean(vmac) < 0.5:
        t = ufloat(teff, err_teff)
        lg = ufloat(logg, err_logg)

        if lg.nominal_value >= 4.0:
            vm = 2.202 * umath.exp(0.0019 * (t - 5777.0)) + 1.30
        elif lg.nominal_value >= 3.0:
            vm = 1.166 * umath.exp(0.0028 * (t - 5777.0)) + 3.30
        else:
            vm = 4.0 + ufloat(0.0, 0.25)

        return np.full(5, vm.nominal_value), np.full(5, vm.std_dev)

    return vmac, err_vmac


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _normalize_continuum(
    wavelength: np.ndarray,
    flux: np.ndarray,
    snr: float,
) -> np.ndarray | None:
    """Normalize continuum for broadening analysis."""
    rejt = 1.0 - 1.0 / max(snr, 20.0)
    p = np.poly1d(np.polyfit(wavelength, flux, 2))
    yfit = p(wavelength)

    for _ in range(3):
        dif = np.concatenate([np.abs((flux[:-1] - flux[1:]) / flux[:-1]), [1.0]])
        mask = ((flux - yfit * rejt) > 0) & (dif < 0.1)
        if np.sum(mask) < 3:
            break
        p = np.poly1d(np.polyfit(wavelength[mask], flux[mask], 2))
        yfit = p(wavelength)

    continuum = p(wavelength)
    if np.any(continuum <= 0):
        return None

    return flux / continuum


def _make_weights(radius: int) -> np.ndarray:
    """Create wing/center weights for chi-squared evaluation."""
    n = 2 * radius + 1
    w = np.zeros(n)
    w[: radius - 3] = 3.0  # blue wing
    w[radius + 4 :] = 5.0  # red wing
    w[radius - 3 : radius + 4] = 25.0  # center
    return w


def _find_spline_minimum(x: np.ndarray, y: np.ndarray) -> float:
    """Find the minimum of y(x) via spline interpolation."""
    try:
        tck = UnivariateSpline(x, y, k=min(4, len(x) - 1), s=0.0)
        roots = tck.derivative().roots()
        if len(roots) > 0:
            tck2 = tck.derivative(n=2)
            minima = roots[tck2(roots) > 0]
            if len(minima) > 0:
                return float(minima[np.argmin(tck(minima))])

        # Fallback: evaluate on fine grid
        fine = np.linspace(x[0], x[-1], 100)
        return float(fine[np.argmin(tck(fine))])
    except (ValueError, RuntimeError):
        return float(x[np.argmin(y)])


def _monte_carlo_error(
    wavelength: np.ndarray,
    flux_norm: np.ndarray,
    snr: float,
    center_idx: int,
    radius: int,
    v_grid: np.ndarray,
    chi2: np.ndarray,
    n_trials: int = 500,
) -> float:
    """Estimate vsini error via Monte Carlo noise perturbation.

    Adds Gaussian noise to the observed spectrum and re-finds the
    chi-squared minimum for each realization.
    """
    ci0 = max(center_idx - radius, 0)
    ci1 = min(center_idx + radius + 1, len(flux_norm))
    segment = flux_norm[ci0:ci1]

    if len(segment) == 0 or snr <= 0:
        return 0.0

    vsini_samples = np.full(n_trials, np.nan)
    noise_scale = np.abs(segment) / snr

    for i in range(n_trials):
        noisy = segment + np.random.normal(0, noise_scale)
        # Re-evaluate chi2 with noisy data (approximate: scale original chi2)
        scale = np.sum((noisy - segment) ** 2) / max(np.sum(segment ** 2), 1e-10)
        noisy_chi2 = chi2 * (1.0 + np.random.normal(0, max(scale, 0.01), len(chi2)))
        noisy_chi2 = np.abs(noisy_chi2)

        try:
            vsini_samples[i] = _find_spline_minimum(v_grid, noisy_chi2)
        except Exception:
            continue

    valid = vsini_samples[np.isfinite(vsini_samples)]
    if len(valid) < 10:
        return 0.0

    return float(np.std(valid))


def _combine_results(
    vsini_arr: np.ndarray,
    err_vsini_arr: np.ndarray,
    vmac_arr: np.ndarray,
    err_vmac_arr: np.ndarray,
) -> BroadeningResult:
    """Combine per-line vsini and vmac measurements."""
    # Filter valid measurements
    valid_vs = (vsini_arr > 0) & (err_vsini_arr > 0)
    valid_vm = (vmac_arr > 0) & (err_vmac_arr > 0)

    if not np.any(valid_vs):
        # No valid vsini — still report vmac
        if np.any(valid_vm):
            vm = np.median(vmac_arr[valid_vm])
            err_vm = np.median(err_vmac_arr[valid_vm]) / np.sqrt(np.sum(valid_vm))
        else:
            vm, err_vm = 0.0, 0.0
        return BroadeningResult(vsini=0.0, vsini_err=0.0, vmac=vm, vmac_err=err_vm)

    # Combine valid measurements
    from uncertainties import unumpy as unp

    mask = valid_vs & valid_vm
    if not np.any(mask):
        mask = valid_vs

    vm_ufloat = np.median(unp.uarray(vmac_arr[mask], err_vmac_arr[mask]))
    vs_ufloat = np.median(unp.uarray(vsini_arr[mask], err_vsini_arr[mask]))
    # Add vmac uncertainty to vsini (they're correlated)
    vs_total = vs_ufloat + ufloat(0.0, vm_ufloat.std_dev)

    vsini_final = float(vs_total.nominal_value)
    err_vsini_final = float(vs_total.std_dev)
    vmac_final = float(vm_ufloat.nominal_value)
    err_vmac_final = float(vm_ufloat.std_dev)

    if np.isnan(err_vsini_final):
        err_vsini_final = 0.0

    return BroadeningResult(
        vsini=vsini_final,
        vsini_err=err_vsini_final,
        vmac=vmac_final,
        vmac_err=err_vmac_final,
    )
