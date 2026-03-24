"""Cross-correlation function for radial velocity measurement and rest-frame correction.

Ported from Ccf.py. Uses a binary mask (default G2) cross-correlated against
the observed spectrum to determine the radial velocity shift.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PyAstronomy import pyasl
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)

C_KMS = 299792.458  # Speed of light in km/s

# Continuum-only wavelength regions for SNR estimation (iron-free zones).
# Shared between CCF and instrument readers — defined once here.
SNR_RANGES = [
    (5979.25, 5983.13),
    (6066.49, 6075.55),
    (6195.95, 6198.74),
    (6386.36, 6392.14),
    (5257.83, 5258.58),
    (5256.03, 5256.70),
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CCFResult:
    """Result of a cross-correlation radial velocity measurement."""

    rv: float  # Radial velocity in km/s
    ccf_velocities: np.ndarray  # Velocity grid
    ccf_values: np.ndarray  # CCF values
    gaussian_params: tuple[float, float, float, float]  # (amplitude, center, sigma, offset)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cross_correlate(
    wavelength: np.ndarray,
    flux: np.ndarray,
    mask_path: Path,
    rv_range: float = 200.0,
    rv_step: float = 0.05,
) -> CCFResult:
    """Cross-correlate a spectrum with a binary mask to determine radial velocity.

    Parameters
    ----------
    wavelength
        1D wavelength array in Angstroms (should cover 5000-6250 A).
    flux
        1D flux array (continuum-normalized preferred).
    mask_path
        Path to binary mask file (e.g. G2.mas).
    rv_range
        Search range in km/s (symmetric around 0).
    rv_step
        Velocity step in km/s.

    Returns
    -------
    CCFResult
        Radial velocity and CCF profile.
    """
    # Build template from binary mask
    x1 = np.arange(wavelength[0] - rv_range, wavelength[0], wavelength[1] - wavelength[0])
    x2 = np.arange(wavelength[-1], wavelength[-1] + rv_range, wavelength[-1] - wavelength[-2])
    xtem = np.concatenate([x1, wavelength, x2])

    lines1, lines2, flux_l = np.genfromtxt(mask_path, dtype=None, unpack=True)
    mask = (lines1 > wavelength[0]) & (lines2 < wavelength[-1])
    lines1_sel = lines1[mask]
    lines2_sel = lines2[mask]
    flux_l_sel = flux_l[mask]

    ftem = np.zeros(xtem.size)
    for i, f in enumerate(flux_l_sel):
        indices = np.where((xtem >= lines1_sel[i]) & (xtem <= lines2_sel[i]))[0]
        if indices.size > 0:
            ftem[indices] = f

    # Cross-correlate
    rv_arr, cc = pyasl.crosscorrRV(wavelength, flux, xtem, ftem, -rv_range, rv_range, rv_step)

    # Fit Gaussian to CCF minimum
    idx_min = np.argmin(cc)
    x_min = rv_arr[idx_min]
    y_min = cc[idx_min]

    try:
        popt, _ = curve_fit(_gaussian, rv_arr, cc, p0=(y_min, x_min, 1.0, 0.0))
    except RuntimeError:
        logger.warning("Gaussian fit to CCF failed, using minimum directly")
        popt = (y_min, x_min, 1.0, 0.0)

    return CCFResult(
        rv=float(popt[1]),
        ccf_velocities=rv_arr,
        ccf_values=cc,
        gaussian_params=tuple(popt),
    )


def compute_snr(wavelength: np.ndarray, flux: np.ndarray) -> float:
    """Estimate signal-to-noise ratio from continuum regions.

    Uses the Beta-Sigma estimator from PyAstronomy on predefined
    iron-free wavelength windows.

    Parameters
    ----------
    wavelength
        1D wavelength array in Angstroms.
    flux
        1D flux array.

    Returns
    -------
    float
        Estimated SNR, capped at 300.
    """
    beq = pyasl.BSEqSamp()
    sn_values = []

    for wl_min, wl_max in SNR_RANGES:
        mask = (wavelength >= wl_min) & (wavelength <= wl_max)
        segment = flux[mask]
        if segment.size <= 4:
            continue

        # Shift to positive if any negative values
        if np.min(segment) < 0.0:
            segment = segment - np.min(segment)

        median_val = np.median(segment)
        try:
            sigma, _ = beq.betaSigma(segment, 1, 1, returnMAD=True)
            if sigma > 0:
                sn_values.append(median_val / sigma)
        except Exception:
            pass

    if not sn_values:
        return 0.0

    valid = np.array(sn_values)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return 0.0

    return float(min(np.median(valid), 300.0))


def apply_doppler_correction(
    wavelength: np.ndarray,
    rv: float,
) -> np.ndarray:
    """Apply Doppler correction to shift wavelength array to rest frame.

    Parameters
    ----------
    wavelength
        Wavelength array in Angstroms (1D or 2D for echelle orders).
    rv
        Radial velocity in km/s.

    Returns
    -------
    np.ndarray
        Corrected wavelength array.
    """
    if rv / C_KMS == -1.0:
        return wavelength
    return wavelength / (1.0 + rv / C_KMS)


# ---------------------------------------------------------------------------
# Continuum normalization for CCF
# ---------------------------------------------------------------------------

def normalize_for_ccf(wavelength: np.ndarray, flux: np.ndarray) -> np.ndarray:
    """Subtract continuum and normalize for cross-correlation.

    Uses iterative 2nd-order polynomial fit with rejection threshold.

    Parameters
    ----------
    wavelength, flux
        1D arrays.

    Returns
    -------
    np.ndarray
        Normalized flux (continuum-subtracted, ~0 in continuum).
    """
    rejt = 0.98
    p = np.poly1d(np.polyfit(wavelength, flux, 2))
    yfit = p(wavelength)

    for _ in range(3):
        with np.errstate(divide="ignore", invalid="ignore"):
            dif = np.concatenate([np.abs((flux[:-1] - flux[1:]) / flux[:-1]), [1.0]])
        dif = np.nan_to_num(dif, nan=1.0, posinf=1.0, neginf=1.0)
        mask = ((flux - yfit * rejt) > 0) & (dif < 0.1)
        if np.sum(mask) < 3:
            break
        p = np.poly1d(np.polyfit(wavelength[mask], flux[mask], 2))
        yfit = p(wavelength)

    return flux / p(wavelength) - 1.0


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _gaussian(x: np.ndarray, a: float, mu: float, sigma: float, offset: float) -> np.ndarray:
    """Gaussian function for CCF fitting."""
    return offset + a * np.exp(-(x - mu) ** 2 / (2.0 * sigma ** 2))
