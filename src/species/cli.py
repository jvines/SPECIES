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


if __name__ == "__main__":
    sys.exit(main())
