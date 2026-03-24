"""Smoke tests for the SPECIES v4 rewrite.

These tests verify the end-to-end pipeline works by running on the
bundled solar spectrum (sun01_harps). They require MOOG and the ATLAS9
grid to be available (run inside Docker).
"""

import pytest
import numpy as np
from pathlib import Path

# Expected solar values (ground truth)
SOLAR_TEFF = 5777.0
SOLAR_LOGG = 4.44
SOLAR_FEH = 0.0
SOLAR_VT = 1.0

# Tolerances
TEFF_TOL = 150.0  # K
LOGG_TOL = 0.3  # dex
FEH_TOL = 0.15  # dex
VT_TOL = 0.3  # km/s


@pytest.fixture
def solar_spectrum_path():
    """Path to the solar spectrum FITS file."""
    candidates = [
        Path("Spectra/sun01_harps_res.fits"),
        Path("Spectra/sun01_harps.fits"),
    ]
    for p in candidates:
        if p.exists():
            return p
    pytest.skip("Solar spectrum not found")


@pytest.fixture
def config():
    """SPECIES configuration for testing."""
    from species.config import Settings
    return Settings()


class TestImports:
    """Verify all modules import cleanly."""

    def test_top_level_import(self):
        from species import __version__
        assert __version__ == "4.0.0a1"

    def test_lazy_imports(self):
        from species import Spectrum, Analyzer, Settings
        assert Spectrum is not None
        assert Analyzer is not None
        assert Settings is not None

    def test_config_defaults(self):
        from species.config import Settings
        s = Settings()
        assert s.fe_solar == 7.50
        assert s.moog_binary == "MOOGSILENT"
        assert s.linelist_path.exists()
        assert s.mask_path.exists()

    def test_moog_imports(self):
        from species.moog import MOOGRunner, AbfindResult, AtmosphereGrid
        from species.moog.par_file import write_abfind_par, write_synth_par
        from species.moog.parser import parse_abfind_output

    def test_all_modules(self):
        from species.ccf import cross_correlate, compute_snr
        from species.spectrum import Spectrum
        from species.ew import measure_equivalent_widths, LineList
        from species.atmosphere import AtmosphereFitter, AtmosphericParameters
        from species.errors import compute_errors, DWARF, GIANT
        from species.broadening import BroadeningFitter
        from species.io.results import AnalysisResult


class TestMOOGParser:
    """Test MOOG output parsing with a sample file."""

    def test_parser_handles_missing_file(self):
        from species.moog.parser import parse_abfind_output
        with pytest.raises(FileNotFoundError):
            parse_abfind_output(Path("/nonexistent/file.test"))

    def test_abfind_result_defaults(self):
        from species.moog.parser import AbfindResult
        r = AbfindResult()
        assert r.fe_i_abundance == -99.0
        assert r.n_failed == 0
        assert r.lines == {"FeI": [], "FeII": []}


class TestAtmosphereGrid:
    """Test ATLAS9 grid loading and interpolation."""

    def test_grid_loads(self, config):
        from species.moog.atmosphere_grid import AtmosphereGrid
        grid = AtmosphereGrid.get(config.atlas9_dir)
        assert grid.grid is not None
        assert "tgrid" in grid.grid
        assert len(grid.grid["tgrid"]) > 0

    def test_solar_interpolation(self, config, tmp_path):
        from species.moog.atmosphere_grid import AtmosphereGrid
        grid = AtmosphereGrid.get(config.atlas9_dir)
        model_path = tmp_path / "solar.atm"
        ok = grid.interpolate_and_write(
            SOLAR_TEFF, SOLAR_LOGG, SOLAR_FEH, SOLAR_VT, model_path,
        )
        assert ok, "Grid interpolation failed for solar parameters"
        assert model_path.exists()
        content = model_path.read_text()
        assert "KURUCZ" in content
        assert "NTAU" in content

    def test_out_of_bounds_fails(self, config, tmp_path):
        from species.moog.atmosphere_grid import AtmosphereGrid
        grid = AtmosphereGrid.get(config.atlas9_dir)
        model_path = tmp_path / "bad.atm"
        ok = grid.interpolate_and_write(
            50000.0, 10.0, 5.0, 0.0, model_path,
        )
        assert not ok, "Should fail for out-of-bounds parameters"


class TestParFile:
    """Test MOOG parameter file generation."""

    def test_abfind_par(self, tmp_path):
        from species.moog.par_file import write_abfind_par
        par = write_abfind_par(
            tmp_path / "test.par",
            model_path=tmp_path / "model.atm",
            linelist_path=tmp_path / "lines.txt",
            summary_out=tmp_path / "summary.out",
            standard_out=tmp_path / "standard.out",
        )
        content = par.read_text()
        assert "abfind" in content
        assert "model_in" in content

    def test_synth_par(self, tmp_path):
        from species.moog.par_file import write_synth_par
        par = write_synth_par(
            tmp_path / "test.par",
            model_path=tmp_path / "model.atm",
            linelist_path=tmp_path / "lines.txt",
            summary_out=tmp_path / "summary.out",
            standard_out=tmp_path / "standard.out",
            smoothed_out=tmp_path / "smoothed.out",
            wave_start=6000.0,
            wave_end=6010.0,
        )
        content = par.read_text()
        assert "synth" in content
        assert "6000.00" in content


class TestMOOGRunner:
    """Test MOOG execution (requires MOOG binary)."""

    def test_moog_not_found(self):
        from species.moog.wrapper import MOOGRunner, MOOGError
        runner = MOOGRunner(moog_binary="/nonexistent/MOOGSILENT")
        with pytest.raises(MOOGError, match="not found"):
            runner.run_abfind(Path("/tmp/model.atm"), Path("/tmp/lines.txt"))

    def test_abfind_solar(self, config, tmp_path):
        """Run MOOG abfind on solar model — the critical integration test."""
        from species.moog.wrapper import MOOGRunner
        from species.moog.atmosphere_grid import AtmosphereGrid

        grid = AtmosphereGrid.get(config.atlas9_dir)
        model_path = tmp_path / "solar.atm"
        ok = grid.interpolate_and_write(
            SOLAR_TEFF, SOLAR_LOGG, SOLAR_FEH, SOLAR_VT, model_path,
        )
        if not ok:
            pytest.skip("Grid interpolation failed")

        runner = MOOGRunner(
            moog_binary=config.moog_binary,
            moog_data_dir=config.moog_data_dir,
        )

        # We need a MOOG-format linelist with EWs
        # Use the bundled EW file if available
        ew_file = Path("EW/sun01_harps.txt")
        if not ew_file.exists():
            pytest.skip("Solar EW file not available")

        # Create MOOG linelist from EWs
        from species.ew import LineList, create_moog_linelist, EWResult
        linelist = LineList.from_file(config.linelist_path)

        # Read existing EW file
        ew_data = np.genfromtxt(ew_file, names=("wl", "ew", "ew_mean", "ew_err1", "ew_err2"))
        ew_results = [
            EWResult(
                wavelength=float(row["wl"]),
                ew=float(row["ew"]),
                ew_median=float(row["ew_mean"]),
                ew_err_plus=float(row["ew_err1"]),
                ew_err_minus=float(row["ew_err2"]),
            )
            for row in ew_data
        ]

        moog_lines = tmp_path / "solar_lines.txt"
        n_fe_i, n_fe_ii = create_moog_linelist(ew_results, linelist, moog_lines)
        assert n_fe_i > 0, "No Fe I lines in MOOG linelist"

        result = runner.run_abfind(model_path, moog_lines)
        assert result.fe_i_abundance > 0, "MOOG returned no Fe I abundance"
        assert result.n_fe_i > 0

        # Solar Fe abundance should be near 7.50
        assert abs(result.fe_i_abundance - 7.50) < 0.3, (
            f"Solar Fe I abundance {result.fe_i_abundance} too far from 7.50"
        )


class TestSpectrum:
    """Test spectrum loading."""

    def test_from_fits(self, solar_spectrum_path):
        from species.spectrum import Spectrum
        spec = Spectrum.from_fits(solar_spectrum_path, instrument="HARPS")
        assert spec.wavelength is not None
        assert spec.flux is not None
        assert len(spec.wavelength) > 0

    def test_from_arrays(self):
        from species.spectrum import Spectrum
        wave = np.linspace(5000, 7000, 10000)
        flux = np.ones(10000)
        spec = Spectrum.from_arrays(wave, flux, snr=100.0, instrument="TEST")
        assert spec.instrument == "TEST"
        assert spec.snr == 100.0


class TestEndToEnd:
    """End-to-end pipeline test on solar spectrum."""

    @pytest.mark.slow
    def test_full_pipeline(self, solar_spectrum_path, config):
        """Run the full SPECIES pipeline on the solar spectrum.

        This is THE test — if this passes with correct solar parameters,
        the rewrite is working.
        """
        from species.spectrum import Spectrum
        from species.analyzer import Analyzer

        spec = Spectrum.from_fits(solar_spectrum_path, instrument="HARPS")
        analyzer = Analyzer(spec, output_dir=Path("/tmp/species_test"), config=config)
        analyzer.configure(compute_broadening=False)  # Skip broadening for speed

        result = analyzer.run()

        assert result.params.converged, "Solar spectrum should converge"

        assert abs(result.params.teff - SOLAR_TEFF) < TEFF_TOL, (
            f"Teff={result.params.teff} too far from {SOLAR_TEFF}"
        )
        assert abs(result.params.logg - SOLAR_LOGG) < LOGG_TOL, (
            f"logg={result.params.logg} too far from {SOLAR_LOGG}"
        )
        assert abs(result.params.feh - SOLAR_FEH) < FEH_TOL, (
            f"[Fe/H]={result.params.feh} too far from {SOLAR_FEH}"
        )

        # Test serialization
        d = result.to_dict()
        assert "teff" in d
        assert "logg" in d
        assert d["star_name"] is not None
