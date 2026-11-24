"""MAROON-X reader: synthetic HDFStores, plus the public DRS sample when MAROONX_SAMPLE is set."""

from __future__ import annotations

import os

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("tables")

from species.spectrum import Spectrum, instrument_resolution, vacuum_to_air

NPIX = 600
BLUE_ORDERS = [91, 92, 93]
RED_ORDERS = [70, 71]
SENTINEL = 1.0e6


def _arm(orders, lam0_nm, rng):
    """One arm's MultiIndex frame: quantity columns, [Fiber, Order] rows of arrays."""
    rows = {}
    for fiber in (2, 6):
        for k, o in enumerate(orders):
            lo = lam0_nm + 12.0 * k
            w = np.linspace(lo, lo + 10.0, NPIX)
            x = np.linspace(-1, 1, NPIX)
            blaze = 1000.0 * (1.0 - 0.8 * x**2)
            line = 1.0 - 0.6 * np.exp(-0.5 * ((w - (lo + 5.0)) / 0.02) ** 2)
            flux = blaze * line + rng.normal(0, 3, NPIX)
            var = np.full(NPIX, 100.0)
            if fiber == 6:
                flux[100:105] = 777.0     # interpolated filler
                var[100:105] = SENTINEL
            rows[(fiber, o)] = {
                "wavelengths": w, "optimal_extraction": flux,
                "optimal_var": var, "box_extraction": flux,
            }
    idx = pd.MultiIndex.from_tuples(list(rows), names=["Fiber", "Order"])
    return pd.DataFrame(list(rows.values()), index=idx)


def _header(arm):
    return pd.Series({
        "ARM": arm, "EXPTIME": "900.0", "BERV_FLUXWEIGHTED_FRD": "3858.70",
        "JD_UTC_FLUXWEIGHTED_FRD": "2461278.9278449",
        "MAROONX TELESCOPE TARGETNAME": "TEST STAR 1", "MAROONX TELESCOPE AIRMASS": "1.1",
        "MAROONX TELESCOPE PROGRAMID": "SYNTH",
    })


@pytest.fixture()
def maroonx_file(tmp_path):
    rng = np.random.default_rng(1)
    path = tmp_path / "20990101T000000Z_SOOOE_x_0900.hd5"
    with pd.HDFStore(path, "w") as s:
        s["spec_blue"] = _arm(BLUE_ORDERS, 500.0, rng)
        s["spec_red"] = _arm(RED_ORDERS, 650.0, rng)
        s["header_blue"] = _header("BLUE")
        s["header_red"] = _header("RED")
    return path


@pytest.fixture()
def response_file(tmp_path):
    path = tmp_path / "response.hd5"
    x = np.linspace(-1, 1, NPIX)
    resp = 1.0 - 0.8 * x**2
    with pd.HDFStore(path, "w") as s:
        s["response_blue"] = pd.DataFrame({o: resp for o in BLUE_ORDERS if o != 91})
        s["response_red"] = pd.DataFrame({o: resp for o in RED_ORDERS})
    return path


def test_reads_orders_and_drops_blue_91(maroonx_file, monkeypatch):
    monkeypatch.delenv("MAROONX_RESPONSE", raising=False)
    spec = Spectrum.from_fits(maroonx_file)
    assert spec.instrument == "MAROON-X"
    assert spec.wavelength.shape[0] == 4          # 2 blue (92, 93) + 2 red
    assert spec.wavelength.shape == spec.flux.shape
    firsts = np.nanmin(np.where(spec.wavelength > 0, spec.wavelength, np.nan), axis=1)
    assert np.all(np.diff(firsts) > 0)            # bluest first
    assert spec.resolving_power == 85_000.0 == instrument_resolution("MAROON-X")
    assert spec.star_name == "TEST_STAR_1"


def test_wavelengths_converted_to_air(maroonx_file):
    air = Spectrum.from_maroonx(maroonx_file)
    vac = Spectrum.from_maroonx(maroonx_file, medium="vacuum")
    ratio = vac.wavelength[vac.wavelength > 0] / air.wavelength[air.wavelength > 0]
    assert np.all((ratio > 1.00027) & (ratio < 1.00029))
    assert air.header["WAVELENGTH_MEDIUM"] == "air"
    # 6000 A vacuum is 5998.34 A in standard air
    assert vacuum_to_air(np.array([6000.0]))[0] == pytest.approx(5998.34, abs=0.01)


def test_variance_mask_not_flux(maroonx_file):
    spec = Spectrum.from_maroonx(maroonx_file)
    assert np.isnan(spec.flux[:, 100:105]).all()
    assert not np.any(spec.flux[np.isfinite(spec.flux)] == 777.0)


def test_snr_from_flux_over_sqrt_var(maroonx_file):
    spec = Spectrum.from_maroonx(maroonx_file)
    assert spec.snr.shape == (4,)
    # blaze peak ~1000 / sqrt(100) = 100; median over the order sits lower
    assert np.all((spec.snr > 30) & (spec.snr < 100))


def test_flattening_fallback_without_response(maroonx_file, monkeypatch):
    monkeypatch.delenv("MAROONX_RESPONSE", raising=False)
    spec = Spectrum.from_maroonx(maroonx_file)
    assert spec.header["RESPONSE_APPLIED"] is False
    cont = np.nanmedian(spec.flux, axis=1)
    assert np.allclose(cont, 1.0, atol=0.05)


def test_response_by_argument_and_env(maroonx_file, response_file, monkeypatch):
    monkeypatch.delenv("MAROONX_RESPONSE", raising=False)
    a = Spectrum.from_fits(maroonx_file, response=response_file)
    assert a.header["RESPONSE_APPLIED"] is True
    monkeypatch.setenv("MAROONX_RESPONSE", str(response_file))
    b = Spectrum.from_fits(maroonx_file)
    assert b.header["RESPONSE_APPLIED"] is True
    np.testing.assert_allclose(np.nan_to_num(a.flux), np.nan_to_num(b.flux))
    # response removed the blaze: central pixels sit near 1000
    assert np.nanmedian(a.flux[:, NPIX // 2 - 20: NPIX // 2 + 20]) == pytest.approx(1000, rel=0.1)


def test_missing_response_file_is_an_error(maroonx_file, tmp_path):
    with pytest.raises(FileNotFoundError):
        Spectrum.from_maroonx(maroonx_file, response=tmp_path / "nope.hd5")


def test_header_metadata(maroonx_file):
    h = Spectrum.from_maroonx(maroonx_file).header
    assert h["BERV_KMS"] == pytest.approx(3.8587)
    assert h["JD_UTC_FLUXWEIGHTED_FRD"] == pytest.approx(2461278.9278449)
    assert h["EXPTIME"] == 900.0


def test_combine_orders_runs(maroonx_file):
    one = Spectrum.from_maroonx(maroonx_file).combine_orders()
    assert one.wavelength.ndim == 1 and np.isfinite(one.flux).all()


# --- masterflat blaze (SPEC-6) ---------------------------------------------

@pytest.fixture()
def blaze_file(tmp_path):
    """Masterflat blaze store in the DRS layout: column ``blaze``, [Fiber, Order]."""
    path = tmp_path / "20990101T00_masterflat__FFFFF_x__blaze.hd5"
    x = np.linspace(-1, 1, NPIX)
    blz = 5.0e4 * (1.0 - 0.8 * x**2)      # same shape as the synthetic blaze
    with pd.HDFStore(path, "w") as s:
        for arm, orders in (("blue", BLUE_ORDERS), ("red", RED_ORDERS)):
            idx = pd.MultiIndex.from_tuples([(f, o) for f in (2, 6) for o in orders],
                                            names=["Fiber", "Order"])
            s[f"blaze_{arm}"] = pd.DataFrame({"blaze": [blz] * len(idx)}, index=idx)
    return path


def test_blaze_by_argument_and_env(maroonx_file, blaze_file, monkeypatch):
    monkeypatch.delenv("MAROONX_RESPONSE", raising=False)
    monkeypatch.delenv("MAROONX_BLAZE", raising=False)
    a = Spectrum.from_fits(maroonx_file, blaze=blaze_file)
    assert a.header["BLAZE_APPLIED"] is True
    assert a.header["N_ORDERS_DEBLAZED"] == 4
    assert a.header["RESPONSE_APPLIED"] is False
    monkeypatch.setenv("MAROONX_BLAZE", str(blaze_file))
    b = Spectrum.from_fits(maroonx_file)
    np.testing.assert_allclose(np.nan_to_num(a.flux), np.nan_to_num(b.flux))
    # normalised continuum, and the line at the order centre keeps its depth
    assert np.allclose(np.nanmedian(a.flux, axis=1), 1.0, atol=0.02)
    centre = np.nanmin(a.flux[:, NPIX // 2 - 10: NPIX // 2 + 10], axis=1)
    assert np.allclose(centre, 0.4, atol=0.05)


def test_blaze_flattens_better_than_polynomial_alone(maroonx_file, blaze_file, monkeypatch):
    monkeypatch.delenv("MAROONX_RESPONSE", raising=False)
    monkeypatch.delenv("MAROONX_BLAZE", raising=False)
    poly = Spectrum.from_maroonx(maroonx_file)
    blz = Spectrum.from_maroonx(maroonx_file, blaze=blaze_file)
    # off-line continuum scatter: deblazed is at least as flat
    sl = np.r_[50:250, 350:550]
    assert np.nanstd(blz.flux[:, sl]) <= np.nanstd(poly.flux[:, sl]) * 1.05


def test_response_takes_precedence_over_blaze(maroonx_file, response_file, blaze_file, monkeypatch):
    monkeypatch.delenv("MAROONX_RESPONSE", raising=False)
    monkeypatch.delenv("MAROONX_BLAZE", raising=False)
    s = Spectrum.from_maroonx(maroonx_file, response=response_file, blaze=blaze_file)
    assert s.header["RESPONSE_APPLIED"] is True and s.header["BLAZE_APPLIED"] is False


def test_missing_blaze_file_is_an_error(maroonx_file, tmp_path, monkeypatch):
    monkeypatch.delenv("MAROONX_RESPONSE", raising=False)
    with pytest.raises(FileNotFoundError):
        Spectrum.from_maroonx(maroonx_file, blaze=tmp_path / "nope.hd5")


def test_simbad_star_prefix_stripped(tmp_path):
    rng = np.random.default_rng(2)
    path = tmp_path / "x.hd5"
    h = _header("BLUE"); h["MAROONX TELESCOPE TARGETNAME"] = "* tau Cet"
    with pd.HDFStore(path, "w") as s:
        s["spec_blue"] = _arm(BLUE_ORDERS, 500.0, rng)
        s["spec_red"] = _arm(RED_ORDERS, 650.0, rng)
        s["header_blue"] = h
        s["header_red"] = _header("RED")
    assert Spectrum.from_maroonx(path).star_name == "tau_Cet"


# --- real DRS product --------------------------------------------------------
# Public sample from the EPRV data-standard MAROON-X tutorial (tau Cet, 40 s):
#   http://grinnell.as.arizona.edu/~rvdata/maroonx/20240923T092056Z_SOOOE_x_0040.hd5
#   http://grinnell.as.arizona.edu/~rvdata/maroonx/20240916T19_masterflat__FFFFF_x__blaze.hd5
# Set MAROONX_SAMPLE (and optionally MAROONX_SAMPLE_BLAZE) to the downloaded files.



_SAMPLE = os.environ.get("MAROONX_SAMPLE")
_SAMPLE_BLAZE = os.environ.get("MAROONX_SAMPLE_BLAZE")


@pytest.mark.skipif(not _SAMPLE, reason="MAROONX_SAMPLE not set (real DRS file)")
def test_real_drs_file(monkeypatch):
    monkeypatch.delenv("MAROONX_RESPONSE", raising=False)
    monkeypatch.delenv("MAROONX_BLAZE", raising=False)
    s = Spectrum.from_fits(_SAMPLE)
    assert s.instrument == "MAROON-X"
    assert s.star_name == "tau_Cet"
    assert s.header["N_ORDERS"] == 61            # 33 blue (91 dropped) + 28 red
    w = s.wavelength[s.wavelength > 0]
    assert 4890 < w.min() < 4920 and 9190 < w.max() < 9230   # air, A
    assert np.median(s.snr) > 300
    # continuum (95th percentile per order) at 1; medians sit lower in line-rich orders
    assert np.allclose(np.nanpercentile(s.flux, 95, axis=1), 1.0, atol=0.01)
    assert s.header["BERV_KMS"] == pytest.approx(7.87184, abs=1e-4)
    # Na D2 (5889.95 A air) in the stellar frame, shifted by tau Cet's RV
    # (-16.6 km/s) and the BERV: line core within 0.15 A of the prediction.
    c = 299792.458
    pred = 5889.951 * (1 + (-16.6 - 7.872) / c)
    one = s.combine_orders()
    sel = (one.wavelength > pred - 1.0) & (one.wavelength < pred + 1.0)
    core = one.wavelength[sel][np.nanargmin(one.flux[sel])]
    assert core == pytest.approx(pred, abs=0.15)


@pytest.mark.skipif(not (_SAMPLE and _SAMPLE_BLAZE), reason="MAROONX_SAMPLE(_BLAZE) not set")
def test_real_drs_file_with_masterflat_blaze(monkeypatch):
    monkeypatch.delenv("MAROONX_RESPONSE", raising=False)
    s = Spectrum.from_maroonx(_SAMPLE, blaze=_SAMPLE_BLAZE)
    assert s.header["BLAZE_APPLIED"] is True
    assert s.header["N_ORDERS_DEBLAZED"] == 61
    assert np.allclose(np.nanpercentile(s.flux, 95, axis=1), 1.0, atol=0.01)
    # dividing by the blaze leaves less continuum ripple than the polynomial alone
    poly = Spectrum.from_maroonx(_SAMPLE)

    def ripple(spec):
        return np.median([
            np.nanstd([np.nanpercentile(c, 95) for c in np.array_split(r[np.isfinite(r)], 10)])
            for r in spec.flux
        ])

    assert ripple(s) < 0.5 * ripple(poly)
