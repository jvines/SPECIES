# SPECIES v4

**SPECtroscopic Inference of stEllar parameterS**

A modern stellar spectroscopy pipeline for determining atmospheric parameters, chemical abundances, and broadening from high-resolution echelle spectra.

Originally created by Maritza Soto & James Jenkins ([Soto & Jenkins 2018](http://adsabs.harvard.edu/abs/2018A%26A...615A..76S)). Modernized and maintained by Jose Vines.

## What it does

Given a high-resolution stellar spectrum, SPECIES determines:

- **Atmospheric parameters**: Teff, log g, [Fe/H], microturbulence
- **Chemical abundances**: Na, Mg, Al, Si, Ca, Ti, Cr, Mn, Fe, Ni, Cu, Zn (14 ions)
- **Broadening**: vsini (rotational velocity) and vmac (macroturbulence)

It works by measuring equivalent widths of iron lines, then iteratively solving for the atmospheric parameters that produce excitation balance, ionization balance, and abundance consistency using MOOG (Sneden 1973) under the assumption of local thermodynamic equilibrium (LTE). Atmosphere models are interpolated from ATLAS9 grids (Castelli & Kurucz 2004).

## Installation

```bash
pip install astro-species
species install-moog
species check
```

`install-moog` downloads and compiles MOOG from source. You need `gfortran` installed:

| Platform | Command |
|----------|---------|
| Ubuntu/Debian | `sudo apt install gfortran` |
| Fedora/RHEL | `sudo dnf install gcc-gfortran` |
| macOS | `brew install gcc` |
| Conda | `conda install -c conda-forge gfortran` |

You also need ATLAS9 atmosphere model grids (~3.3 GB). These are not bundled with the package. Set `SPECIES_ATLAS9_DIR` to point to your local copy, or they will be downloaded on first use (TBD).

### Docker

For zero-setup usage:

```bash
docker build -t species .
docker run -v ./spectra:/data species analyze /data/my_star.fits
```

## Quick start

### Python API

```python
from species import Spectrum, Analyzer

# Load a spectrum (auto-detects instrument from FITS header)
spectrum = Spectrum.from_fits("my_star_feros.fits")

# Or from arrays (e.g. from your own reduction pipeline)
spectrum = Spectrum.from_arrays(wavelength, flux, snr=120, instrument="FEROS")

# Run the full analysis
result = Analyzer(spectrum).run()

# Results are structured Python objects
print(result.params.teff)           # 5822.0
print(result.params.logg)           # 4.52
print(result.params.feh)            # -0.003
print(result.abundances["SiI"])     # ElementAbundance(abundance=+0.068, n_lines=20)
print(result.broadening.vsini)      # 2.1 km/s

# Serialize
result.to_fits("results.fits")
result.to_dict()  # for databases, JSON, etc.
```

### Command line

```bash
# Full analysis
species analyze my_star.fits --output ./results

# Skip broadening (faster)
species analyze my_star.fits --no-broadening

# Giant star
species analyze my_star.fits --giant

# EW measurement only
species ew-only my_star.fits

# Check installation
species check
```

### Configuration

All settings can be overridden via environment variables or constructor arguments:

```python
from species import Analyzer, Spectrum, Settings

config = Settings(
    atlas9_dir="/path/to/grids",
    moog_binary="/usr/local/bin/MOOGSILENT",
    is_giant=True,
)

result = Analyzer(spectrum, config=config).run()
```

Environment variables use the `SPECIES_` prefix: `SPECIES_ATLAS9_DIR`, `SPECIES_MOOG_BINARY`, etc.

## Supported instruments

SPECIES reads spectra from:

- **HARPS** (ESO 3.6m)
- **FEROS** (ESO/MPG 2.2m)
- **UVES** (ESO VLT)
- **HIRES** (Keck)
- **CORALIE** (Euler 1.2m)
- **PFS** (Magellan)
- **AAT**
- **LCO NRES**
- Generic FITS (auto-detected from WCS keywords or 2D data layout)

Instrument is auto-detected from the FITS header `INSTRUME` keyword or the filename convention (`starname_instrument.fits`). You can also pass `instrument="FEROS"` explicitly.

## Changes from the original SPECIES

### Algorithmic improvements

**Broyden's method replaces per-parameter bisection.** The original adjusted one parameter at a time (metallicity, then temperature, then gravity, then microturbulence) via bisection, cycling until all four diagnostics converged. This required 600-900 MOOG calls per star and had convergence issues when porting from Python 2 to 3 due to float comparison semantics. The new default uses Broyden's quasi-Newton method, which solves the 4-equation system simultaneously in 10-16 MOOG calls. The residual norm is typically 250x smaller. The old bisection method is still available via `method="per_parameter"`.

**Broadening uses MOOG 5x instead of 75x.** The original ran a separate MOOG synthesis for each vsini grid point (15 values x 5 lines = 75 calls). But the unbroadened spectrum is identical regardless of vsini — rotational broadening is applied afterward. The new code synthesizes once per diagnostic line and applies `pyasl.rotBroad()` in numpy.

**Error propagation unified.** The original had two nearly identical files (`CalcErrors_new.py` for dwarfs, `CalcErrorsGiant_new.py` for giants) differing only in polynomial coefficients. Now a single parameterized module.

### Architecture

| Aspect | Original (v3) | v4 |
|--------|--------------|-----|
| Python version | 2.7/3.x compatible | 3.12+ |
| Packaging | None (clone and run) | `pip install astro-species` |
| API | CLI only (`python SPECIES.py -starlist ...`) | Python library + CLI |
| Configuration | 80+ global variables, 40+ CLI flags | Pydantic Settings + env vars |
| MOOG interface | `os.system("MOOGSILENT")` | `subprocess.run()` with timeout, error handling |
| Type hints | None | Full coverage |
| Tests | None | 17 integration tests |
| Output | Text files at hardcoded paths | Structured dataclasses, FITS, ASCII |
| Parallelism | Per-star (`multiprocessing.Pool`) | Per-star + MOOGSession for per-call efficiency |
| Lines of code | ~9,800 | ~5,800 |

### What was removed

- **Isochrone fitting** (`FindIsochrones.py`): Mass, age, and radius determination is now handled by downstream tools (Lachesis). SPECIES focuses on spectroscopy.
- **Python 2 compatibility**: `from __future__` imports, `past.utils.old_div`, `builtins` shims.
- **MultiNest/emcee dependencies**: These were only used by the isochrone module.
- **`mwdust` dependency**: Extinction correction is handled elsewhere.
- **Full `PhotometricRelations_class.py`**: Replaced with a lightweight Vizier lookup for initial parameter guesses. With Broyden's method, initial guesses matter less.

### Performance

| Metric | Original | v4 |
|--------|----------|-----|
| MOOG calls per star | 600-900 | 15-25 |
| Wall-clock time | 10-20 min | ~1 min |
| Convergence residual | ~0.05 | ~0.0002 |

## Package structure

```
src/species/
    __init__.py          # Public API: Spectrum, Analyzer, Settings
    config.py            # Pydantic Settings (replaces global variables)
    cli.py               # Command-line interface
    analyzer.py          # Main orchestrator
    spectrum.py          # Spectrum loading + instrument readers
    ccf.py               # Cross-correlation for RV / rest-frame correction
    ew.py                # Equivalent width measurement
    atmosphere.py        # Atmospheric parameter fitting (Broyden + bisection)
    errors.py            # Error propagation (unified dwarf/giant)
    abundances.py        # Chemical abundance determination
    broadening.py        # vsini / vmac measurement
    photometry.py        # Vizier initial parameter estimation
    moog/
        wrapper.py       # MOOG subprocess interface + MOOGSession
        parser.py        # MOOG output parsing
        par_file.py      # Parameter file generation
        atmosphere_grid.py  # ATLAS9 grid interpolation
    io/
        results.py       # Result dataclasses + serialization
        plots.py         # Diagnostic plots
    data/
        linelists/       # Bundled iron + abundance line lists
        masks/           # Binary masks for CCF
        solar.py         # Asplund 2009 solar abundances
```

## Citation

If you use SPECIES in your work, please cite:

- Soto & Jenkins 2018, A&A 615, A76 ([ADS](http://adsabs.harvard.edu/abs/2018A%26A...615A..76S))
- Soto et al. 2021, MNRAS 508, 1

## License

MIT
