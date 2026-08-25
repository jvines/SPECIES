"""Command-line interface for SPECIES.

Usage::

    species analyze star_feros.fits --instrument FEROS --output ./results
    species analyze star.fits --no-broadening --no-abundances
    species ew-only star.fits --instrument HARPS
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """SPECIES CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="species",
        description="SPECIES — SPECtroscopic Inference of stEllar parameterS",
    )
    parser.add_argument("--version", action="store_true", help="Show version and exit")

    sub = parser.add_subparsers(dest="command")

    # --- analyze ---
    p_analyze = sub.add_parser("analyze", help="Run full spectroscopic analysis")
    p_analyze.add_argument("spectra", type=Path, nargs="+", help="Path(s) to FITS spectra")
    p_analyze.add_argument("--instrument", "-i", default=None, help="Instrument name (auto-detected if omitted)")
    p_analyze.add_argument("--output", "-o", type=Path, default=Path("./output"), help="Output directory")
    p_analyze.add_argument("--ncores", "-n", type=int, default=1, help="Number of parallel workers")
    p_analyze.add_argument("--giant", action="store_true", help="Treat as giant star")
    p_analyze.add_argument("--no-broadening", action="store_true", help="Skip vsini/vmac measurement")
    p_analyze.add_argument("--no-abundances", action="store_true", help="Skip chemical abundances")
    p_analyze.add_argument("--no-errors", action="store_true", help="Skip error propagation")
    p_analyze.add_argument("--no-restframe", action="store_true", help="Skip rest-frame correction")
    p_analyze.add_argument("--method", choices=["broyden", "per_parameter", "downhill_simplex"], default=None)
    p_analyze.add_argument("--format", choices=["ascii", "fits", "both"], default="both", help="Output format")
    p_analyze.add_argument("-v", "--verbose", action="store_true")

    # --- ew-only ---
    p_ew = sub.add_parser("ew-only", help="Measure equivalent widths only")
    p_ew.add_argument("spectrum", type=Path, help="Path to FITS spectrum")
    p_ew.add_argument("--instrument", "-i", default=None)
    p_ew.add_argument("--output", "-o", type=Path, default=Path("./output"))
    p_ew.add_argument("-v", "--verbose", action="store_true")

    # --- install-moog ---
    p_moog = sub.add_parser("install-moog", help="Download and compile MOOG")
    p_moog.add_argument(
        "--prefix", type=Path, default=None,
        help="Install directory (default: ~/.species/moog)",
    )
    p_moog.add_argument(
        "--commit", default="53d3f645f18c1acfb568c5ed7ea0c4e4551ef5e5",
        help="moog17scat git commit to checkout",
    )
    p_moog.add_argument("-v", "--verbose", action="store_true")

    # --- check ---
    sub.add_parser("check", help="Check SPECIES installation and dependencies")

    args = parser.parse_args(argv)

    if args.version:
        from species import __version__
        print(f"SPECIES v{__version__}")
        return 0

    if args.command is None:
        parser.print_help()
        return 0

    # Setup logging
    level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s", stream=sys.stderr)

    if args.command == "analyze":
        return _cmd_analyze(args)
    elif args.command == "ew-only":
        return _cmd_ew_only(args)
    elif args.command == "install-moog":
        return _cmd_install_moog(args)
    elif args.command == "check":
        return _cmd_check(args)

    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    """Run full analysis pipeline on one or more spectra."""
    from species.spectrum import Spectrum
    from species.analyzer import Analyzer
    from species.config import Settings

    t0 = time.time()

    config = Settings()
    if args.method:
        config.minimization = args.method

    configure_kwargs = dict(
        is_giant=args.giant,
        compute_broadening=not args.no_broadening,
        compute_abundances=not args.no_abundances,
        compute_errors=not args.no_errors,
        compute_rest_frame=not args.no_restframe,
    )

    spectra = [Spectrum.from_fits(p, instrument=args.instrument) for p in args.spectra]

    if len(spectra) == 1:
        # Single star — direct run with full output
        analyzer = Analyzer(spectra[0], output_dir=args.output, config=config)
        analyzer.configure(**configure_kwargs)
        result = analyzer.run()
        _print_result(result)
        _write_result(result, args.output, args.format)
        elapsed = time.time() - t0
        print(f"Time: {elapsed:.1f}s")
        return 0 if result.params.converged else 1

    # Multiple stars — batch mode
    n = len(spectra)
    ncores = min(args.ncores, n)
    print(f"Analyzing {n} spectra ({ncores} worker{'s' if ncores > 1 else ''})...\n")

    results = Analyzer.batch(
        spectra,
        output_dir=args.output,
        config=config,
        n_cores=ncores,
        **configure_kwargs,
    )

    # Summary table
    all_ok = True
    for result in results:
        status = "OK" if result.params.converged else "FAIL"
        if not result.params.converged:
            all_ok = False
        n_ab = len(result.abundances)
        print(
            f"  [{status:4s}] {result.star_name:20s}  "
            f"T={result.params.teff:7.1f}  logg={result.params.logg:5.3f}  "
            f"[Fe/H]={result.params.feh:+6.3f}  vt={result.params.vt:5.3f}  "
            f"({n_ab} abundances)"
        )
        _write_result(result, args.output, args.format)

    elapsed = time.time() - t0
    n_converged = sum(1 for r in results if r.params.converged)
    print(f"\n{n_converged}/{n} converged in {elapsed:.1f}s")
    return 0 if all_ok else 1


def _print_result(result) -> None:
    """Print detailed results for a single star."""
    if result.params.converged:
        print(f"Converged: T={result.params.teff:.0f} K  "
              f"logg={result.params.logg:.2f}  "
              f"[Fe/H]={result.params.feh:.3f}  "
              f"vt={result.params.vt:.3f} km/s")
    else:
        print(f"WARNING: Did not converge. Best estimate: T={result.params.teff:.0f} K  "
              f"logg={result.params.logg:.2f}  [Fe/H]={result.params.feh:.3f}")

    if result.errors:
        print(f"Errors:   ±{result.errors.err_teff:.0f} K  "
              f"±{result.errors.err_logg:.3f}  "
              f"±{result.errors.err_feh:.3f}  "
              f"±{result.errors.err_vt:.3f}")

    if result.abundances:
        print(f"Abundances: {len(result.abundances)} elements")
        for name, ab in sorted(result.abundances.items()):
            print(f"  {name:6s}  [{name}/H]={ab.abundance:+.3f} ± {ab.uncertainty:.3f}  ({ab.n_lines} lines)")

    if result.broadening:
        print(f"Broadening: vsini={result.broadening.vsini:.2f} ± {result.broadening.vsini_err:.2f} km/s  "
              f"vmac={result.broadening.vmac:.2f} ± {result.broadening.vmac_err:.2f} km/s")


def _write_result(result, output_dir: Path, fmt: str) -> None:
    """Write result files for one star."""
    out = Path(output_dir)
    if fmt in ("ascii", "both"):
        result.to_ascii(out / f"{result.star_name}_results.dat")
    if fmt in ("fits", "both"):
        result.to_fits(out / f"{result.star_name}_results.fits")


def _cmd_ew_only(args: argparse.Namespace) -> int:
    """Measure equivalent widths only."""
    from species.spectrum import Spectrum
    from species.analyzer import Analyzer
    from species.config import Settings

    config = Settings()
    spec = Spectrum.from_fits(args.spectrum, instrument=args.instrument)
    analyzer = Analyzer(spec, output_dir=args.output, config=config)

    ew_results = analyzer.run_ew_only()
    valid = [r for r in ew_results if r.is_valid]
    print(f"Measured {len(valid)} valid EWs out of {len(ew_results)} lines")

    for r in valid:
        print(f"  {r.wavelength:8.2f}  EW={r.ew_median:7.2f} +{r.ew_err_plus:.2f} -{r.ew_err_minus:.2f} mA")

    return 0


def _cmd_install_moog(args: argparse.Namespace) -> int:
    """Download and compile MOOG from moog17scat."""
    import shutil
    import subprocess

    prefix = args.prefix or Path.home() / ".species" / "moog"
    repo_url = "https://github.com/jvines/moog17scat.git"
    commit = args.commit

    # Check for gfortran
    gfortran = shutil.which("gfortran")
    if not gfortran:
        print("Error: gfortran not found.")
        print()
        print("Install it with:")
        print("  Ubuntu/Debian:  sudo apt install gfortran")
        print("  Fedora/RHEL:    sudo dnf install gcc-gfortran")
        print("  macOS:          brew install gcc")
        print("  Conda:          conda install -c conda-forge gfortran")
        return 1

    make = shutil.which("make")
    if not make:
        print("Error: make not found. Install build-essential (Linux) or Xcode CLI tools (macOS).")
        return 1

    git = shutil.which("git")
    if not git:
        print("Error: git not found.")
        return 1

    print(f"Installing MOOG to {prefix}")
    print(f"  gfortran: {gfortran}")
    print()

    # Clone
    if prefix.exists():
        print(f"Directory {prefix} already exists.")
        moogsilent = prefix / "MOOGSILENT"
        if moogsilent.exists():
            print(f"MOOG binary found at {moogsilent}")
            response = input("Reinstall? [y/N] ").strip().lower()
            if response != "y":
                _print_config_hint(prefix)
                return 0
        shutil.rmtree(prefix)

    print("Cloning moog17scat...")
    result = subprocess.run(
        ["git", "clone", repo_url, str(prefix)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Error cloning: {result.stderr}")
        return 1

    # Checkout pinned commit
    subprocess.run(
        ["git", "checkout", commit],
        cwd=prefix, capture_output=True, text=True,
    )

    # Compile
    print("Compiling MOOG...")
    result = subprocess.run(
        ["make"],
        cwd=prefix,
        capture_output=not args.verbose,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        print("Compilation failed.")
        if not args.verbose:
            print("Run with -v to see compiler output.")
            if result.stderr:
                print(result.stderr[-500:])
        return 1

    moogsilent = prefix / "MOOGSILENT"
    if not moogsilent.exists():
        print("Error: MOOGSILENT binary was not produced.")
        print("Check the compiler output with: species install-moog -v")
        return 1

    # Verify it computes, not merely that it starts.
    #
    # This used to run the binary with empty input and check that the string
    # "MOOG" appeared in stdout -- i.e. that it printed its own banner. That
    # passes for an installation whose support files are missing, which is the
    # dangerous case: MOOG then falls back to Unsold van der Waals damping and
    # returns abundances shifted by ~0.04 dex, silently. A banner cannot tell
    # you that; an abundance can.
    print("Verifying MOOG produces abundances...")
    ok_run, detail = _verify_moog_computes(moogsilent, prefix)
    if ok_run:
        print(f"MOOG works: {detail}")
    else:
        print(f"Warning: MOOG did not produce a usable abundance ({detail}).")
        print("The binary may still start and print its banner while returning")
        print("wrong numbers. Investigate before running science with it.")

    print()
    print(f"MOOG installed at {moogsilent}")
    _print_config_hint(prefix)
    return 0


def _verify_moog_computes(moogsilent: Path, data_dir: Path) -> tuple[bool, str]:
    """Run a real abfind and check MOOG returns a plausible Fe abundance.

    Interpolates a solar atmosphere from the bundled grid, feeds MOOG three
    Fe I lines with hand-set equivalent widths, and requires an answer in the
    range a solar model must give. This is deliberately a *value* check: an
    installation with missing support files starts fine, prints its banner, and
    returns abundances offset by ~0.04 dex.

    Returns (ok, human-readable detail).
    """
    import tempfile

    from species.moog.atmosphere_grid import AtmosphereGrid
    from species.moog.wrapper import MOOGError, MOOGRunner

    # Three clean, unblended Fe I lines with EWs typical of the solar spectrum.
    lines = (
        # wavelength, species, excitation potential, log gf, EW (mA)
        (6027.050, 26.0, 4.076, -1.09, 63.0),
        (6151.618, 26.0, 2.176, -3.30, 51.0),
        (6165.360, 26.0, 4.143, -1.46, 45.0),
    )

    try:
        with tempfile.TemporaryDirectory(prefix="species_verify_") as tmp:
            tmpdir = Path(tmp)
            model = tmpdir / "solar.atm"
            grid = AtmosphereGrid.get(data_dir)
            if not grid.interpolate_and_write(5777.0, 4.44, 0.0, 1.0, model):
                return False, "could not interpolate a solar atmosphere"

            linelist = tmpdir / "lines.dat"
            linelist.write_text(
                "solar verification lines\n"
                + "".join(
                    f"{wl:10.3f}{sp:10.1f}{ep:10.3f}{gf:10.3f}"
                    f"{'':10}{'':10}{ew:10.1f}\n"
                    for wl, sp, ep, gf, ew in lines
                )
            )

            runner = MOOGRunner(moog_binary=str(moogsilent), moog_data_dir=data_dir)
            result = runner.run_abfind(model, linelist)
    except (MOOGError, OSError, ValueError) as e:
        return False, f"{type(e).__name__}: {e}"

    a_fe = getattr(result, "fe_i_abundance", None)
    if a_fe is None or not (a_fe == a_fe):        # None or NaN
        return False, "no Fe I abundance in the output"
    # A solar model with solar EWs must land near the solar iron abundance.
    # Wide on purpose: this is an installation check, not a calibration.
    if not (6.8 <= a_fe < 8.2):
        return False, f"A(Fe I) = {a_fe:.3f}, outside the plausible solar range"
    return True, f"A(Fe I) = {a_fe:.3f} from {result.n_fe_i} solar Fe I lines"


def _print_config_hint(prefix: Path) -> None:
    """Print instructions for configuring SPECIES to find MOOG."""
    moogsilent = prefix / "MOOGSILENT"
    print()
    print("To use with SPECIES, either:")
    print()
    print(f"  1. Add to PATH:  export PATH=\"{prefix}:$PATH\"")
    print()
    print("  2. Set environment variable:")
    print(f"     export SPECIES_MOOG_BINARY={moogsilent}")
    print(f"     export SPECIES_MOOG_DATA_DIR={prefix}")
    print()
    print("  3. Pass in Python:")
    print(f"     Settings(moog_binary=\"{moogsilent}\", moog_data_dir=\"{prefix}\")")
    print()
    print("Add the export lines to your ~/.bashrc or ~/.zshrc to make permanent.")


def _cmd_check(args: argparse.Namespace) -> int:
    """Check SPECIES installation and dependencies."""
    import shutil

    from species import __version__
    from species.config import Settings

    print(f"SPECIES v{__version__}")
    print()

    config = Settings()
    ok = True

    # Check MOOG
    moog_path = shutil.which(config.moog_binary)
    if moog_path:
        print(f"  MOOG binary:     {moog_path}")
    else:
        # Check common locations
        home_moog = Path.home() / ".species" / "moog" / "MOOGSILENT"
        if home_moog.exists():
            print(f"  MOOG binary:     {home_moog} (not on PATH)")
            print(f"                   Run: export PATH=\"{home_moog.parent}:$PATH\"")
        else:
            print(f"  MOOG binary:     NOT FOUND")
            print(f"                   Run: species install-moog")
            ok = False
    print()

    # Check ATLAS9 grids
    from species.moog.atmosphere_grid import AtmosphereGrid

    grid_dir = config.atlas9_dir
    grid_path = grid_dir / AtmosphereGrid.GRID_FILENAME
    grids_dir = grid_dir / "grids"
    if not grid_path.exists():
        # The bundled copy is what a pip install actually resolves to.
        try:
            from importlib.resources import files
            bundled = Path(str(files("species").joinpath("data", AtmosphereGrid.GRID_FILENAME)))
            if bundled.exists():
                grid_path = bundled
        except Exception:
            pass
    if grid_path.exists():
        size_mb = grid_path.stat().st_size / 1e6
        print(f"  ATLAS9 grid:     {grid_path} ({size_mb:.0f} MB)")
    elif grids_dir.exists():
        print(f"  ATLAS9 grid:     {grids_dir} (raw files; the .nc grid is built on first run)")
    else:
        print(f"  ATLAS9 grid:     NOT FOUND at {grid_dir}")
        print(f"                   Set SPECIES_ATLAS9_DIR to the directory containing grids/")
        ok = False
    print()

    # Check linelists
    for name, path in [
        ("Iron linelist", config.linelist_path),
        ("Abundance linelist", config.linelist_ab_path),
        ("Binary mask", config.mask_path),
    ]:
        if path and path.exists():
            print(f"  {name:20s} {path}")
        else:
            print(f"  {name:20s} NOT FOUND: {path}")
            ok = False
    print()

    # Check optional deps
    for pkg, desc in [
        ("astroquery", "Vizier photometric lookups"),
    ]:
        try:
            __import__(pkg)
            print(f"  {pkg:20s} installed")
        except ImportError:
            print(f"  {pkg:20s} not installed (optional: {desc})")
    print()

    if ok:
        print("All required dependencies found.")
    else:
        print("Some dependencies are missing. See above.")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
