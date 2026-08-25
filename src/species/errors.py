"""Unified error propagation for atmospheric parameters.

Replaces CalcErrors_new.py (dwarf, 226 lines) and CalcErrorsGiant_new.py
(228 lines) with a single parameterized module. The two files were 95%
identical, differing only in polynomial coefficients and error caps.

Uses the ``uncertainties`` package for automatic error propagation through
polynomial transfer functions that map measured MOOG diagnostics (EP slope,
RW slope, Fe I-II difference) to parameter uncertainties.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from uncertainties import ufloat, unumpy

from species.atmosphere import AtmosphericParameters
from species.moog.parser import AbfindResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Coefficients for dwarf and giant error transfer functions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ErrorCoefficients:
    """Polynomial transfer function coefficients for error propagation.

    Each coefficient pair (z, z_err) defines a polynomial whose coefficients
    are themselves uncertain, enabling automatic error propagation.
    """

    # Microturbulence error: f(vt) applied to RW slope
    vt_z1: list[float]
    vt_z1_err: list[float]
    vt_z2: list[float]
    vt_z2_err: list[float]

    # Temperature error: f(var) applied to EP slope
    # For dwarfs: var = vt; for giants: var = [Fe/H]
    teff_z1: list[float]
    teff_z1_err: list[float]
    teff_z2: list[float]
    teff_z2_err: list[float]
    teff_z1_variable: str  # "vt" for dwarf, "feh" for giant

    # logg error: f(var) applied to Fe I-II difference
    # For dwarfs: var = vt; for giants: var = logg
    logg_z1: list[float]
    logg_z1_err: list[float]
    logg_z2: list[float]
    logg_z2_err: list[float]
    logg_z1_variable: str  # "vt" for dwarf, "logg" for giant
    logg_scale: float  # 0.55 for dwarf, 1.0 for giant

    # Error caps
    max_err_vt: float
    max_err_teff: tuple[float, float]  # (min_floor, max_cap)
    max_err_feh: float
    max_err_logg: float

    # Default fallback errors (when regression data unavailable)
    default_err_rw: float
    default_err_ep: float
    default_err_dif: float
    default_ab_uncertainty: float  # Per-line abundance uncertainty for unumpy


DWARF = ErrorCoefficients(
    # Microturbulence
    vt_z1=[-3.22, 9.643, -8.795],
    vt_z1_err=[0.132, 0.259, 0.123],
    vt_z2=[1.051, -0.075],
    vt_z2_err=[0.004, 0.004],
    # Temperature (z1 variable = vt)
    teff_z1=[1834.257, -2785.165, -4870.889],
    teff_z1_err=[180.095, 332.9, 145.764],
    teff_z2=[1.003, -15.512],
    teff_z2_err=[0.001, 4.815],
    teff_z1_variable="vt",
    # logg (z1 variable = vt)
    logg_z1=[0.031, -0.343, -1.793],
    logg_z1_err=[0.023, 0.045, 0.021],
    logg_z2=[0.984, 0.05],
    logg_z2_err=[0.002, 0.008],
    logg_z1_variable="vt",
    logg_scale=0.55,
    # Caps
    max_err_vt=1.0,
    max_err_teff=(50.0, 300.0),
    max_err_feh=0.6,
    max_err_logg=0.6,
    # Defaults
    default_err_rw=0.02,
    default_err_ep=0.007,
    default_err_dif=0.1,
    default_ab_uncertainty=0.5,
)

GIANT = ErrorCoefficients(
    # Microturbulence
    vt_z1=[-3.25, 8.8089, -7.2177],
    vt_z1_err=[0.318, 0.79886, 0.494],
    vt_z2=[1.0726, -0.0987],
    vt_z2_err=[0.005, 0.006],
    # Temperature (z1 variable = feh)
    teff_z1=[-2592.84, -5028.971],
    teff_z1_err=[222.543, 29.47],
    teff_z2=[1.002, -6.914],
    teff_z2_err=[0.006, 27.305],
    teff_z1_variable="feh",
    # logg (z1 variable = logg)
    logg_z1=[-2.379, 29.588, -137.375, 282.347, -219.058],
    logg_z1_err=[0.516, 6.374, 29.443, 60.222, 46.022],
    logg_z2=[0.978, 0.068],
    logg_z2_err=[0.009, 0.029],
    logg_z1_variable="logg",
    logg_scale=1.0,
    # Caps (giants are more permissive)
    max_err_vt=1.0,
    max_err_teff=(0.0, 300.0),  # No floor for giants
    max_err_feh=1.0,
    max_err_logg=1.0,
    # Defaults
    default_err_rw=0.03,
    default_err_ep=0.008,
    default_err_dif=0.1,
    default_ab_uncertainty=1.0,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ParameterErrors:
    """Uncertainties for atmospheric parameters."""

    err_teff: float = 0.0
    err_logg: float = 0.0
    err_feh: float = 0.0
    err_vt: float = 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_errors(
    params: AtmosphericParameters,
    abfind_result: AbfindResult,
    stellar_class: Literal["dwarf", "giant"] = "dwarf",
    hold: list[str] | None = None,
    initial_errors: dict[str, float] | None = None,
    use_photometric_teff: bool = False,
    use_fixed_vt: bool = False,
) -> ParameterErrors:
    """Compute parameter uncertainties using polynomial error transfer functions.

    Parameters
    ----------
    params
        Converged atmospheric parameters from the fitting.
    abfind_result
        MOOG abfind output with per-line data.
    stellar_class
        ``"dwarf"`` or ``"giant"`` — selects coefficient set.
    hold
        List of held parameter names.
    initial_errors
        Errors for held parameters: ``{"temperature": 80.0, ...}``.
    use_photometric_teff
        If True, temperature error comes from photometric estimate.
    use_fixed_vt
        If True, vt was held at a fixed value (error = 0.1).

    Returns
    -------
    ParameterErrors
    """
    if hold is None:
        hold = []
    if initial_errors is None:
        initial_errors = {}

    coeff = GIANT if stellar_class == "giant" else DWARF

    # Extract regression data from MOOG output
    ep, rw, err_ep, err_rw, err_dif, err_ab_fe_i = _extract_regression_data(
        abfind_result, coeff
    )

    t_moog = params.teff
    logg_moog = params.logg
    feh_moog = params.feh
    vt_moog = params.vt

    # --- Microturbulence error ---
    if "velocity" in hold:
        micro_i = ufloat(vt_moog, initial_errors.get("velocity", 0.1))
    elif use_fixed_vt:
        micro_i = ufloat(vt_moog, 0.1)
    else:
        z1 = unumpy.uarray(coeff.vt_z1, coeff.vt_z1_err)
        z2 = unumpy.uarray(coeff.vt_z2, coeff.vt_z2_err)
        rw_i = ufloat(rw, err_rw)
        c1 = np.polyval(z1, vt_moog)
        c2 = np.polyval(z2, vt_moog)
        micro_i = np.polyval([c1, c2], rw_i)

    # --- Temperature error ---
    if use_photometric_teff or "temperature" in hold:
        T_i = ufloat(t_moog, initial_errors.get("temperature", 100.0))
    else:
        z1 = unumpy.uarray(coeff.teff_z1, coeff.teff_z1_err)
        z2 = unumpy.uarray(coeff.teff_z2, coeff.teff_z2_err)
        ep_i = ufloat(ep, err_ep)
        # Key difference: dwarf uses vt, giant uses [Fe/H] for z1
        z1_var = vt_moog if coeff.teff_z1_variable == "vt" else feh_moog
        c1 = np.polyval(z1, z1_var)
        c2 = np.polyval(z2, t_moog)
        T_i = np.polyval([c1, c2], ep_i)

    # --- Metallicity error ---
    if "metallicity" in hold:
        feh_i = ufloat(feh_moog, initial_errors.get("metallicity", 0.1))
    else:
        feh_i = ufloat(feh_moog, err_ab_fe_i)

    # --- logg error ---
    if "pressure" in hold or "gravity" in hold:
        logg_i = ufloat(logg_moog, initial_errors.get("gravity", 0.1))
    else:
        z1 = unumpy.uarray(coeff.logg_z1, coeff.logg_z1_err)
        z2 = unumpy.uarray(coeff.logg_z2, coeff.logg_z2_err)
        dif_i = ufloat(0.0, err_dif)
        # Key difference: dwarf uses vt, giant uses logg for z1
        z1_var = vt_moog if coeff.logg_z1_variable == "vt" else logg_moog
        c1 = np.polyval(z1, z1_var)
        c2 = np.polyval(z2, logg_moog)
        logg_i = np.polyval([c1, c2], dif_i)

    # Apply caps
    err_vt = min(micro_i.s, coeff.max_err_vt)
    floor, cap = coeff.max_err_teff
    err_teff = min(max(T_i.s, floor), cap)
    err_feh = min(feh_i.s, coeff.max_err_feh)
    err_logg = min(logg_i.s * coeff.logg_scale, coeff.max_err_logg)

    logger.info(
        "Errors: T=%.1f logg=%.3f [Fe/H]=%.3f vt=%.3f",
        err_teff, err_logg, err_feh, err_vt,
    )

    return ParameterErrors(
        err_teff=err_teff,
        err_logg=err_logg,
        err_feh=err_feh,
        err_vt=err_vt,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_regression_data(
    result: AbfindResult,
    coeff: ErrorCoefficients,
) -> tuple[float, float, float, float, float, float]:
    """Extract EP slope, RW slope, and their errors from MOOG per-line data.

    Returns (ep, rw, err_ep, err_rw, err_dif, err_ab_fe_i).
    """
    from astropy.stats import sigma_clip
    from scipy.stats import linregress

    fe_i_lines = result.lines.get("FeI", [])
    fe_ii_lines = result.lines.get("FeII", [])

    ep = result.ep_slope
    rw = result.rw_slope

    # Defaults
    err_ep = coeff.default_err_ep
    err_rw = coeff.default_err_rw
    err_dif = coeff.default_err_dif
    err_ab_fe_i = 0.1

    if len(fe_i_lines) >= 3:
        ab = np.array([lr.abundance for lr in fe_i_lines])
        ep_arr = np.array([lr.ep for lr in fe_i_lines])
        rw_arr = np.array([lr.log_rw for lr in fe_i_lines])

        clipped = sigma_clip(ab, maxiters=1)
        mask = ~clipped.mask

        if np.sum(mask) >= 3:
            ab_clean = ab[mask]
            ep_clean = ep_arr[mask]
            rw_clean = rw_arr[mask]

            idx_ep = np.argsort(ep_clean)
            idx_rw = np.argsort(rw_clean)

            res_ep = linregress(ep_clean[idx_ep], ab_clean[idx_ep])
            res_rw = linregress(rw_clean[idx_rw], ab_clean[idx_rw])

            if res_ep.stderr > 0:
                err_ep = res_ep.stderr
            if res_rw.stderr > 0:
                err_rw = res_rw.stderr

            # Mean abundance uncertainty using unumpy. This one matches the
            # original exactly: 0.5/sqrt(N_FeI), a pure line-count statistic
            # that does not depend on S/N. Questionable, but it is what the
            # published errors were computed with, so it stays until there is a
            # validation target to change it against.
            m = unumpy.uarray(ab_clean, np.full(len(ab_clean), coeff.default_ab_uncertainty))
            err_ab_fe_i = float(np.mean(m).s)

            # Uncertainty on Fe I - Fe II: the line-to-line SCATTER of the
            # clipped Fe I abundances, not the standard error of a mean.
            #
            # v4 used the SEM of a pooled Fe I + Fe II array with an assumed
            # 0.5 dex per line -- 0.5/sqrt(N_I + N_II), about 0.038 for a
            # typical 150 + 20 line star, against a real scatter of 0.10-0.15.
            # err_dif drives the log g uncertainty through the polynomial
            # transfer function, so log g errors came out 3-4x too small;
            # Soto & Jenkins 2018 Table 1 publishes dwarf sigma(log g) of
            # 0.24-0.75 dex.
            #
            # CalcErrors_new.py used np.std here and had the SEM route sitting
            # commented out immediately above it, so the port reversed a
            # deliberate choice. The scatter is the right scale for a difference
            # of means only because the two samples share their dominant
            # systematics; it is not a formal propagation, it is the empirical
            # quantity the polynomial coefficients were calibrated against.
            if len(fe_ii_lines) >= 2:
                err_dif = float(np.std(ab_clean))

    return ep, rw, err_ep, err_rw, err_dif, err_ab_fe_i
