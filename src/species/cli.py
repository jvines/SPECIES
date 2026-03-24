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
    p_analyze.add_argument("spectrum", type=Path, help="Path to FITS spectrum")
    p_analyze.add_argument("--instrument", "-i", default=None, help="Instrument name (auto-detected if omitted)")
    p_analyze.add_argument("--output", "-o", type=Path, default=Path("./output"), help="Output directory")
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
    """Run full analysis pipeline."""
    from species.spectrum import Spectrum
    from species.analyzer import Analyzer
    from species.config import Settings

    t0 = time.time()

    config = Settings()
    if args.method:
        config.minimization = args.method

    spec = Spectrum.from_fits(args.spectrum, instrument=args.instrument)
    analyzer = Analyzer(spec, output_dir=args.output, config=config)
    analyzer.configure(
        is_giant=args.giant,
        compute_broadening=not args.no_broadening,
        compute_abundances=not args.no_abundances,
        compute_errors=not args.no_errors,
        compute_rest_frame=not args.no_restframe,
    )

    result = analyzer.run()

    # Output
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

    # Write output files
    out_dir = Path(args.output)
    if args.format in ("ascii", "both"):
        p = result.to_ascii(out_dir / f"{result.star_name}_results.dat")
        print(f"Wrote: {p}")
    if args.format in ("fits", "both"):
        p = result.to_fits(out_dir / f"{result.star_name}_results.fits")
        print(f"Wrote: {p}")

    elapsed = time.time() - t0
    print(f"Time: {elapsed:.1f}s")
    return 0 if result.params.converged else 1


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

    # Verify it runs
    print("Verifying MOOG binary...")
    try:
        proc = subprocess.run(
            [str(moogsilent)],
            input="\n\n",
            capture_output=True, text=True,
            timeout=5,
        )
        if "MOOG" in proc.stdout:
            print("MOOG binary works.")
        else:
            print("Warning: MOOG binary ran but output was unexpected.")
    except Exception as e:
        print(f"Warning: could not verify MOOG binary: {e}")

    print()
    print(f"MOOG installed at {moogsilent}")
    _print_config_hint(prefix)
    return 0


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
    grid_dir = config.atlas9_dir
    pickle_path = grid_dir / "ATLAS9_grid.pickle"
    grids_dir = grid_dir / "grids"
    if pickle_path.exists():
        size_mb = pickle_path.stat().st_size / 1e6
        print(f"  ATLAS9 grid:     {pickle_path} ({size_mb:.0f} MB)")
    elif grids_dir.exists():
        print(f"  ATLAS9 grid:     {grids_dir} (raw files, pickle will be created on first run)")
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
