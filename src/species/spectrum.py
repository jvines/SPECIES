"""Spectrum loading, instrument-specific readers, and rest-frame correction.

Ported from Spectra.py, Instruments.py, and the rest-frame parts of Ccf.py.
Replaces the old pattern of per-instrument functions that each read FITS,
call restframe(), and write _res.fits files. Instead, Spectrum is an immutable
data container, and rest-frame correction returns a new Spectrum.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from astropy.io import fits
from PyAstronomy import pyasl
from scipy import interpolate

from species.ccf import (
    CCFResult,
    apply_doppler_correction,
    compute_snr,
    cross_correlate,
    normalize_for_ccf,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core data container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Spectrum:
    """A loaded stellar spectrum.

    Attributes
    ----------
    wavelength : np.ndarray
        Wavelength array in Angstroms. Shape is (N,) for 1D or (N_orders, N_pixels) for echelle.
    flux : np.ndarray
        Flux array matching wavelength shape.
    snr : float or np.ndarray
        Signal-to-noise ratio (scalar or per-order).
    instrument : str
        Instrument name (e.g. ``"FEROS"``, ``"HARPS"``).
    star_name : str
        Star identifier.
    rv : float or None
        Radial velocity in km/s if rest-frame corrected, else None.
    header : dict
        FITS header as plain dict (not astropy Header — serialization-safe).
    ccf_result : CCFResult or None
        CCF details if rest-frame correction was performed.
    """

    wavelength: np.ndarray
    flux: np.ndarray
    snr: float | np.ndarray
    instrument: str = "unknown"
    star_name: str = ""
    rv: float | None = None
    header: dict = field(default_factory=dict)
    ccf_result: CCFResult | None = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_fits(
        cls,
        path: str | Path,
        instrument: str | None = None,
        star_name: str | None = None,
    ) -> Spectrum:
        """Load a spectrum from a FITS file.

        Attempts to auto-detect the instrument from FITS header keywords
        if ``instrument`` is not specified.

        Parameters
        ----------
        path
            Path to the FITS file.
        instrument
            Instrument name. If ``None``, auto-detected from header.
        star_name
            Star name for labeling. Derived from filename if not given.
        """
        path = Path(path)
        if star_name is None:
            star_name = path.stem

        hdu_list = fits.open(path)
        header0 = hdu_list[0].header

        if instrument is None:
            instrument = _detect_instrument(header0, path)

        reader = _READERS.get(instrument.upper(), _read_generic)
        wavelength, flux, snr, header_dict = reader(hdu_list, header0, path)
        hdu_list.close()

        return cls(
            wavelength=wavelength,
            flux=flux,
            snr=snr if snr is not None else 0.0,
            instrument=instrument.upper(),
            star_name=star_name,
            header=header_dict,
        )

    @classmethod
    def from_arrays(
        cls,
        wavelength: np.ndarray,
        flux: np.ndarray,
        snr: float | np.ndarray = 0.0,
        instrument: str = "unknown",
        star_name: str = "",
    ) -> Spectrum:
        """Create a Spectrum from pre-loaded arrays.

        This is the primary integration point for ExoAutomata, which
        passes pre-combined spectra directly.
        """
        return cls(
            wavelength=np.asarray(wavelength, dtype=np.float64),
            flux=np.asarray(flux, dtype=np.float64),
            snr=snr,
            instrument=instrument.upper(),
            star_name=star_name,
        )

    # ------------------------------------------------------------------
    # Operations (return new Spectrum)
    # ------------------------------------------------------------------

    def to_rest_frame(self, mask_path: Path) -> Spectrum:
        """Apply CCF rest-frame correction.

        Cross-correlates the spectrum (in the 5000-6250 A range) with the
        given binary mask, determines the radial velocity, and returns a
        new Spectrum with corrected wavelengths.

        Parameters
        ----------
        mask_path
            Path to binary mask file (e.g. G2.mas).

        Returns
        -------
        Spectrum
            New Spectrum with Doppler-corrected wavelengths and ``rv`` set.
        """
        # Work on 1D representation
        wave_1d, flux_1d = self._get_1d()

        # Select CCF range
        ccf_mask = (wave_1d >= 5000.0) & (wave_1d <= 6250.0)
        if np.sum(ccf_mask) == 0:
            logger.warning("No data in 5000-6250 A range for CCF; skipping rest-frame correction")
            return self

        wave_range = wave_1d[ccf_mask]
        flux_range = flux_1d[ccf_mask]

        if np.mean(flux_range) == 0.0:
            logger.warning("Zero-flux in CCF range; skipping rest-frame correction")
            return self

        # Normalize and cross-correlate
        flux_norm = normalize_for_ccf(wave_range, flux_range)
        ccf_result = cross_correlate(wave_range, flux_norm, mask_path)

        logger.info("RV = %.3f km/s", ccf_result.rv)

        # Apply Doppler correction to full wavelength array
        wave_corrected = apply_doppler_correction(self.wavelength, ccf_result.rv)

        # Compute SNR on corrected 1D spectrum if not already set
        snr = self.snr
        if isinstance(snr, (int, float)) and snr == 0.0:
            wave_1d_corr = apply_doppler_correction(wave_1d, ccf_result.rv)
            snr = compute_snr(wave_1d_corr, flux_1d)
            if np.isnan(snr):
                snr = 0.0

        return Spectrum(
            wavelength=wave_corrected,
            flux=self.flux,
            snr=snr,
            instrument=self.instrument,
            star_name=self.star_name,
            rv=ccf_result.rv,
            header=self.header,
            ccf_result=ccf_result,
        )

    def combine_orders(self) -> Spectrum:
        """Combine echelle orders into a single 1D spectrum.

        Uses spline interpolation to resample all orders onto a uniform
        wavelength grid.

        Returns
        -------
        Spectrum
            1D spectrum. No-op if already 1D.
        """
        if self.wavelength.ndim == 1:
            return self

        wave_1d, flux_1d = _combine_orders_arrays(self.wavelength, self.flux)

        # Collapse SNR to scalar
        snr = self.snr
        if hasattr(snr, "__len__"):
            valid = np.array(snr)
            valid = valid[np.isfinite(valid) & (valid > 0)]
            snr = float(np.median(valid)) if valid.size > 0 else 0.0

        return Spectrum(
            wavelength=wave_1d,
            flux=flux_1d,
            snr=snr,
            instrument=self.instrument,
            star_name=self.star_name,
            rv=self.rv,
            header=self.header,
            ccf_result=self.ccf_result,
        )

    def _get_1d(self) -> tuple[np.ndarray, np.ndarray]:
        """Return 1D wavelength/flux arrays, combining orders if needed."""
        if self.wavelength.ndim == 1:
            return self.wavelength, self.flux
        return _combine_orders_arrays(self.wavelength, self.flux)

    def to_fits(self, path: str | Path) -> Path:
        """Write the spectrum to a FITS file (2-row format: wavelength + flux).

        Parameters
        ----------
        path
            Output file path.

        Returns
        -------
        Path
            The written path.
        """
        path = Path(path)
        if self.wavelength.ndim == 1:
            data = np.zeros((2, len(self.wavelength)))
            data[0] = self.wavelength
            data[1] = self.flux
        else:
            data = np.zeros((2, *self.wavelength.shape))
            data[0] = self.wavelength
            data[1] = self.flux

        hdr = fits.Header()
        if self.rv is not None:
            hdr["RV"] = (self.rv, "Radial velocity (km/s)")
        if isinstance(self.snr, (int, float)):
            hdr["SNR"] = (float(self.snr), "Signal-to-noise ratio")
        elif hasattr(self.snr, "__len__"):
            for i, s in enumerate(self.snr):
                hdr[f"SNR{i}"] = (float(s), f"SNR order {i}")
        hdr["INSTRUME"] = (self.instrument, "Instrument name")

        path.parent.mkdir(parents=True, exist_ok=True)
        fits.writeto(path, data=data, header=hdr, overwrite=True)
        return path


# ---------------------------------------------------------------------------
# Order combination (ported from Instruments.py combine_orders)
# ---------------------------------------------------------------------------

def _combine_orders_arrays(
    wave: np.ndarray,
    flux: np.ndarray,
    reverse: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine 2D echelle orders into a single 1D spectrum via spline interpolation."""
    ranges_wl = []
    splines = []
    deltas = []
    all_bounds = []

    for w, f in zip(wave, flux):
        valid = ~np.isnan(f) & (w != 0.0)
        w_clean = w[valid]
        f_clean = f[valid]
        if w_clean.size < 6:
            continue

        ranges_wl.append((w_clean[0], w_clean[-1]))
        deltas.append(float(np.mean(np.diff(w_clean))))
        splines.append(interpolate.UnivariateSpline(w_clean, f_clean, k=5, s=0))
        all_bounds.extend([w_clean[0], w_clean[-1]])

    if not deltas:
        return wave.ravel(), flux.ravel()

    if reverse:
        ranges_wl.reverse()
        deltas.reverse()
        splines.reverse()

    wl_min = min(all_bounds)
    wl_max = max(all_bounds)
    n_points = int(np.ceil(abs(wl_max - wl_min) / max(deltas)))

    wave_out = np.linspace(wl_min, wl_max, n_points)
    flux_out = np.zeros(n_points)

    for (wl_lo, wl_hi), spline in zip(ranges_wl, splines):
        mask = (wave_out >= wl_lo) & (wave_out < wl_hi)
        flux_out[mask] = spline(wave_out[mask])

    return wave_out, flux_out


# ---------------------------------------------------------------------------
# Instrument readers
# ---------------------------------------------------------------------------

def _detect_instrument(header: fits.Header, path: Path) -> str:
    """Try to detect instrument from FITS header or filename."""
    for key in ("INSTRUME", "INSTRUMENT"):
        if key in header:
            val = str(header[key]).strip().upper()
            # Normalize common names
            for name in ("FEROS", "HARPS", "UVES", "HIRES", "CORALIE", "PFS", "NRES"):
                if name in val:
                    return name
            return val

    # Fallback: parse filename convention ({star}_{instrument}.fits)
    stem = path.stem.lower()
    for name in ("feros", "harps", "uves", "hires", "coralie", "aat", "pfs", "lconres"):
        if f"_{name}" in stem:
            return name.upper()

    return "GENERIC"


def _header_to_dict(header: fits.Header) -> dict:
    """Convert FITS header to a plain dict (serialization-safe)."""
    d = {}
    for key in header:
        if key:
            try:
                d[key] = header[key]
            except Exception:
                pass
    return d


def _read_harps(
    hdu_list: fits.HDUList, header0: fits.Header, path: Path
) -> tuple[np.ndarray, np.ndarray, float | np.ndarray | None, dict]:
    """Read HARPS FITS files (multiple format variants)."""
    snr = None
    wave, flux = None, None

    if hdu_list[0].data is None:
        # Binary table format
        try:
            wave = hdu_list[1].data["WAVE"][0]
            flux = hdu_list[1].data["FLUX"][0]
        except (ValueError, KeyError):
            wave = hdu_list[1].data["wav"]
            flux = hdu_list[1].data["flux"]

        if "HIERARCH ESO DRS SPE EXT SN0" in header0:
            snr = np.median([header0[f"HIERARCH ESO DRS SPE EXT SN{i}"] for i in range(72)])

    else:
        data = hdu_list[0].data
        shape = data.shape

        if len(shape) == 1:
            # 1D spectrum with WCS
            snr = _extract_harps_snr(header0)
            wave, flux = pyasl.read1dFitsSpec(str(path))

        elif len(shape) == 2 and shape[0] == 2:
            # 2-row: wavelength + flux
            wave = data[0]
            flux = data[1]
            valid = ~np.isnan(flux)
            wave, flux = wave[valid], flux[valid]
            snr = _extract_harps_snr(header0)

        elif len(shape) == 2 and shape[1] == 2:
            # Transposed: (N, 2)
            wave = data[:, 0]
            flux = data[:, 1]
            valid = ~np.isnan(flux)
            wave, flux = wave[valid], flux[valid]
            snr = _extract_harps_snr(header0)

        else:
            # Multi-order DRS format
            flux = data[5]
            wave = data[0]
            sn = data[8]
            snr = np.array([
                np.median(sno[sno > 0]) for sno in sn if np.sum(sno > 0) > 0
            ])

    return wave, flux, snr, _header_to_dict(header0)


def _extract_harps_snr(header: fits.Header) -> float | None:
    """Extract HARPS SNR from various header keyword formats."""
    for prefix in ("HIERARCH ESO DRS SPE EXT SN", "ESO DRS SPE EXT SN", "ESO DRS CAL EXT SN"):
        if f"{prefix}0" in header:
            return np.median([header[f"{prefix}{i}"] for i in range(72)])
    if "SNR" in header:
        return header["SNR"]
    return None


def _read_feros(
    hdu_list: fits.HDUList, header0: fits.Header, path: Path
) -> tuple[np.ndarray, np.ndarray, float | np.ndarray | None, dict]:
    """Read FEROS FITS files (multiple format variants)."""
    snr = None
    wave, flux = None, None

    if hdu_list[0].data is None:
        wave = hdu_list[1].data["WAVE"][0]
        flux = hdu_list[1].data["FLUX"][0]
        valid = ~np.isnan(flux)
        wave, flux = wave[valid], flux[valid]
        if "SNR" in header0:
            snr = header0["SNR"]

    else:
        data = hdu_list[0].data
        shape = data.shape

        if len(shape) >= 2:
            if len(shape) == 2 and shape[0] == 2:
                wave = data[0]
                flux = data[1]

            elif "PIPELINE" in header0 or (len(shape) == 3 and "NAXIS3" in header0):
                n3 = header0.get("NAXIS3", shape[0] if len(shape) == 3 else 0)
                if n3 > 2:
                    wave = data[0]
                    flux = data[5]
                    sn = data[8]
                    # Check for empty flux (all zeros)
                    if len(np.unique(flux)) <= 1:
                        flux = data[1]
                    snr = np.array([
                        np.median(sno[sno > 0]) for sno in sn if np.sum(sno > 0) > 0
                    ])
                else:
                    wave = data[0]
                    flux = data[1]

            else:
                # Multi-extension FEROS (39 orders)
                wave_parts, flux_parts = [], []
                for i in range(min(39, len(hdu_list))):
                    hdr = hdu_list[i].header
                    d = hdu_list[i].data
                    if d is None:
                        continue
                    if "CRPIX1" in hdr and "CDELT1" in hdr and "CRVAL1" in hdr:
                        w = _wcs_wavelength(hdr)
                        wave_parts.append(w)
                        flux_parts.append(d)
                if wave_parts:
                    wave = np.array(wave_parts)
                    flux = np.array(flux_parts)

        else:
            # 1D spectrum
            try:
                wave, flux = pyasl.read1dFitsSpec(str(path))
            except TypeError:
                wave = _wcs_wavelength(header0)
                flux = np.array(hdu_list[0].data)

    if snr is None and "SNR" in header0:
        snr = header0["SNR"]

    return wave, flux, snr, _header_to_dict(header0)


def _read_generic(
    hdu_list: fits.HDUList, header0: fits.Header, path: Path
) -> tuple[np.ndarray, np.ndarray, float | np.ndarray | None, dict]:
    """Generic FITS reader for unsupported instruments."""
    snr = header0.get("SNR")
    wave, flux = None, None

    # Try WCS keywords first
    if "CRPIX1" in header0 and "CDELT1" in header0 and "CRVAL1" in header0:
        wave, flux = pyasl.read1dFitsSpec(str(path))
    elif hdu_list[0].data is not None:
        data = hdu_list[0].data
        if data.ndim == 2 and data.shape[0] == 2:
            wave = data[0]
            flux = data[1]
            valid = ~np.isnan(flux)
            wave, flux = wave[valid], flux[valid]
        elif data.ndim == 2 and data.shape[1] == 2:
            wave = data[:, 0]
            flux = data[:, 1]
            valid = ~np.isnan(flux)
            wave, flux = wave[valid], flux[valid]
        elif data.ndim == 3 and data.shape[-1] == 2:
            wave = data.T[0].T
            flux = data.T[1].T
        elif data.ndim == 3:
            wave = data[0]
            flux = data[1]

    if wave is None:
        raise ValueError(f"Could not read spectrum from {path}: unrecognized FITS format")

    return wave, flux, snr, _header_to_dict(header0)


def _read_uves(
    hdu_list: fits.HDUList, header0: fits.Header, path: Path
) -> tuple[np.ndarray, np.ndarray, float | np.ndarray | None, dict]:
    """Read UVES FITS files. Handles blue+red arm combination."""
    # UVES can come as separate blue/red or combined
    # The combined case is handled by the generic reader
    return _read_generic(hdu_list, header0, path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wcs_wavelength(header: fits.Header) -> np.ndarray:
    """Build wavelength array from WCS keywords."""
    n = int(header["NAXIS1"])
    crpix = float(header["CRPIX1"])
    cdelt = float(header["CDELT1"])
    crval = float(header["CRVAL1"])
    return (np.arange(n) + 1.0 - crpix) * cdelt + crval


# ---------------------------------------------------------------------------
# Reader registry
# ---------------------------------------------------------------------------

_READERS = {
    "HARPS": _read_harps,
    "FEROS": _read_feros,
    "UVES": _read_uves,
    "HIRES": _read_generic,
    "CORALIE": _read_generic,
    "AAT": _read_generic,
    "PFS": _read_generic,
    "LCONRES": _read_generic,
    "NRES": _read_generic,
    "GENERIC": _read_generic,
}
