"""Simplified photometric initial parameter estimation.

Stripped-down port of PhotometricRelations_class.py. Provides initial
Teff/logg guesses from photometric colors when the user doesn't supply
them. Uses Vizier catalog queries (astroquery) when available.

With Broyden's method, initial guesses matter less — the solver
converges from the default (5500, 4.36, 0.0, 1.23) in ~15 iterations.
A photometric initial guess may save 2-3 iterations for unusual stars.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class InitialEstimate:
    """Photometric initial parameter estimate."""

    teff: float = 5500.0
    logg: float = 4.36
    feh: float = 0.0
    vt: float = 1.23
    is_giant: bool = False
    source: str = "default"


def estimate_initial_params(
    star_name: str | None = None,
    ra: float | None = None,
    dec: float | None = None,
    is_giant: bool = False,
) -> InitialEstimate:
    """Estimate initial atmospheric parameters from photometry.

    Attempts to query Vizier for photometric data and derive Teff from
    color-temperature relations. Falls back to sensible defaults.

    Parameters
    ----------
    star_name
        Star name for Vizier resolution (e.g. "HD 209458").
    ra, dec
        Coordinates in degrees (used if star_name is not given).
    is_giant
        If True, use giant-star defaults and relations.

    Returns
    -------
    InitialEstimate
        Initial parameters for the convergence solver.
    """
    # Default values
    if is_giant:
        default = InitialEstimate(teff=4500.0, logg=2.5, feh=0.0, vt=1.5, is_giant=True, source="default_giant")
    else:
        default = InitialEstimate(teff=5500.0, logg=4.36, feh=0.0, vt=1.23, is_giant=False, source="default_dwarf")

    if star_name is None and (ra is None or dec is None):
        return default

    # Try Vizier lookup
    try:
        from astroquery.vizier import Vizier
        import astropy.units as u
        from astropy.coordinates import SkyCoord
    except ImportError:
        logger.debug("astroquery not available, using default initial params")
        return default

    try:
        photometry = _query_vizier(star_name, ra, dec, Vizier, SkyCoord, u)
        if not photometry:
            return default

        teff = _estimate_teff(photometry, is_giant)
        if teff is None:
            return default

        logg = _estimate_logg(teff, is_giant)
        vt = _estimate_vt(teff, logg)

        return InitialEstimate(
            teff=teff, logg=logg, feh=0.0, vt=vt,
            is_giant=is_giant, source="photometric",
        )

    except Exception:
        logger.debug("Photometric lookup failed", exc_info=True)
        return default


# ---------------------------------------------------------------------------
# Vizier queries
# ---------------------------------------------------------------------------

def _query_vizier(star_name, ra, dec, Vizier, SkyCoord, u) -> dict:
    """Query Vizier for 2MASS + Gaia photometry."""
    photometry: dict = {}

    if ra is not None and dec is not None:
        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    elif star_name:
        try:
            coord = SkyCoord.from_name(star_name)
        except Exception:
            return photometry
    else:
        return photometry

    v = Vizier(columns=["*"], row_limit=1)

    # 2MASS
    try:
        result = v.query_region(coord, radius=5 * u.arcsec, catalog="II/246/out")
        if result:
            tab = result[0]
            if "Jmag" in tab.colnames:
                photometry["J"] = float(tab["Jmag"][0])
            if "Hmag" in tab.colnames:
                photometry["H"] = float(tab["Hmag"][0])
            if "Kmag" in tab.colnames:
                photometry["K"] = float(tab["Kmag"][0])
    except Exception:
        pass

    # Gaia DR3
    try:
        result = v.query_region(coord, radius=5 * u.arcsec, catalog="I/355/gaiadr3")
        if result:
            tab = result[0]
            if "Teff" in tab.colnames:
                photometry["gaia_teff"] = float(tab["Teff"][0])
            if "logg" in tab.colnames:
                photometry["gaia_logg"] = float(tab["logg"][0])
    except Exception:
        pass

    return photometry


# ---------------------------------------------------------------------------
# Temperature estimation from colors
# ---------------------------------------------------------------------------

def _estimate_teff(photometry: dict, is_giant: bool) -> float | None:
    """Estimate Teff from available photometry."""
    # Prefer Gaia Teff if available
    if "gaia_teff" in photometry:
        teff = photometry["gaia_teff"]
        if 3000 < teff < 10000:
            logger.info("Using Gaia DR3 Teff = %.0f K", teff)
            return teff

    # J-K color-temperature relation (Casagrande et al. 2010, Gonzalez Hernandez & Bonifacio 2009)
    if "J" in photometry and "K" in photometry:
        jk = photometry["J"] - photometry["K"]
        if 0.1 < jk < 1.0:
            # Rough relation for FGK dwarfs
            teff = 5030.0 + 3020.0 * (0.38 - jk)
            teff = np.clip(teff, 3500, 8500)
            logger.info("Estimated Teff from J-K = %.2f: %.0f K", jk, teff)
            return float(teff)

    return None


def _estimate_logg(teff: float, is_giant: bool) -> float:
    """Rough logg estimate from Teff and luminosity class."""
    if is_giant:
        return min(3.5, max(0.5, 6.5 - teff / 2000.0))

    # Mamajek-style MS relation
    if teff > 6500:
        return 4.2
    elif teff > 5000:
        return 4.4
    else:
        return 4.6


def _estimate_vt(teff: float, logg: float) -> float:
    """Estimate microturbulence from Teff and logg.

    Uses the empirical relation from Tsantaki et al. (2013).
    """
    if 3.6 < logg < 4.8 and 4400 < teff < 6400:
        vt = 6.932e-4 * teff - 0.348 * logg - 1.437
        return max(vt, 0.5)
    return 1.23
