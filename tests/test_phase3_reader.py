"""ESO Phase 3 1D spectra (UVES science data products) are read."""
import numpy as np
from astropy.io import fits

from species.spectrum import Spectrum


def _phase3(path, unit="angstrom", reduced=True):
    n = 2000
    w = np.linspace(4800.0, 6800.0, n)
    if unit == "nm":
        w = w / 10.0
    f = 1000.0 + np.zeros(n)
    f[10] = np.nan
    q = np.zeros(n, dtype=np.int32)
    q[20] = 1
    cols = [fits.Column("WAVE", format=f"{n}D", unit=unit, array=[w]),
            fits.Column("FLUX", format=f"{n}D", array=[5e5 + 0 * f]),
            fits.Column("QUAL", format=f"{n}J", array=[q])]
    if reduced:
        cols += [fits.Column("FLUX_REDUCED", format=f"{n}D", array=[f]),
                 fits.Column("QUAL_REDUCED", format=f"{n}J", array=[q])]
    ph = fits.PrimaryHDU()
    ph.header["INSTRUME"] = "UVES"
    ph.header["SNR"] = 180.0
    fits.HDUList([ph, fits.BinTableHDU.from_columns(cols, name="SPECTRUM")]).writeto(path)
    return path


def test_uves_phase3_reduced_flux(tmp_path):
    s = Spectrum.from_fits(_phase3(tmp_path / "u.fits"))
    assert s.instrument == "UVES"
    assert s.snr == 180.0
    assert len(s.wavelength) == 1998          # NaN and QUAL-flagged pixels dropped
    assert np.allclose(s.flux, 1000.0)        # FLUX_REDUCED preferred
    assert 4799 < s.wavelength.min() < s.wavelength.max() < 6801


def test_phase3_nm_and_flux_only(tmp_path):
    s = Spectrum.from_fits(_phase3(tmp_path / "n.fits", unit="nm", reduced=False))
    assert 4799 < s.wavelength.min() and s.wavelength.max() < 6801   # nm -> Angstrom
    assert np.allclose(s.flux, 5e5)
