# Changelog

## 4.1.0 — 2026-08-24

Eight defects, all introduced by the v4 rewrite and none present in the
published Python-2 SPECIES. **Three of them changed reported numbers silently**,
so results produced with 4.0.x should be checked against the list below before
being relied on.

### Wrong numbers, no error raised

- **The instrumental profile was never applied.** `par_file.py` writes
  `gm 0.000` — no smoothing — on the understanding that the caller broadens in
  numpy, and the caller only ever applied macroturbulence and rotation. The
  instrumental width was therefore absorbed into the fitted `vsini`, making it a
  property of the spectrograph as much as of the star: at R = 48000 the
  instrumental FWHM is ~6.2 km/s, comparable to a slow rotator's vsini. **Any
  vsini from 4.0.x is an upper limit.** Restored per-instrument resolving powers
  and convolution in quadrature with vmac.

- **`hold` was ignored by the solver but honoured by the error model.**
  `hold=["gravity"]` returned a *freely fitted* log g reported with the
  uncertainty of a held one — a fabricated error bar in an output table. The
  Broyden solver now masks held parameters out of the Jacobian, the Newton step,
  the rank-1 update and the convergence norm.

- **σ(log g) was ~2× too small.** `err_dif` used the standard error of a pooled
  Fe I + Fe II mean (0.5/√(N_I+N_II) ≈ 0.038) where the original used the
  line-to-line *scatter* (0.10–0.15), with the SEM route explicitly commented
  out. `err_dif` drives log g through the polynomial transfer function.

- **`downhill_simplex` analysed every star at the starting metallicity.** The
  objective read `feh = feh0  # Updated below` and never updated it, so all
  MOOG calls ran at [Fe/H] = 0.0 by default and the abundance residual was
  dropped from the objective entirely. For a metal-poor star that propagates
  into log g through the ionisation balance. Not the default method, so most
  users are unaffected. [Fe/H] is now a fixed point at each trial.

### Failed loudly, or produced nothing

- **`Analyzer.batch(n_cores>1)` failed every star.** The worker was a function
  nested inside `batch`, which cannot be pickled, so every submit raised — and
  the CLI then wrote a full set of zero-filled result files. It also passed five
  of the spectrum's fields and dropped `rv`, `header` and `ccf_result`, so a
  worker would silently re-run the CCF on an already-corrected spectrum.

- **`species install-moog` verified nothing.** It ran the binary with empty
  input and checked that the string "MOOG" appeared in stdout — that it printed
  its banner. An installation with missing support files passes that and then
  returns abundances offset ~0.04 dex, because MOOG silently falls back to
  Unsöld van der Waals damping. Verification now interpolates a solar
  atmosphere, runs abfind, and requires a plausible A(Fe I). Missing support
  files now warn.

- **`__version__` was wrong.** pyproject said 4.0.4, `__init__` said 4.0.0, and
  the first test in the suite asserted 4.0.0a1 — so it was red, and nothing ran
  it. Now single-sourced from distribution metadata.

### Changed on purpose

- **The ATLAS9 grid ships as NetCDF, not a pickle** — 13.3 MB against 34.8 MB
  for bit-identical numbers, verified by round trip and by a byte-identical
  interpolated solar `.atm`. A pickle is unversioned, executes arbitrary code on
  load, and is readable only from a compatible Python. Adds a `netCDF4`
  dependency.

- **The EW line-width acceptance bound scales with the expected width.** It was
  a hardcoded 0.10 Å, which is vsini ≈ 6–7 km/s at 5500 Å — a moderate rotator
  lost most of its lines despite them being measured perfectly well. The bound
  is now built from the instrumental profile, rotation (Gray 2005 eq. 18.14) and
  macroturbulence. With no resolving power known it reproduces the old 0.10 Å
  exactly, so behaviour is unchanged for callers that cannot supply R.

  Measured with the same change on the sibling Julia implementation over 192
  Gaia FGK Benchmark Stars: median Fe lines per star 120 → 140, gained in 190
  and lost in 0; stars with too few lines to solve 25 → 12; |Δlog g| 0.195 →
  0.086 dex over the 118 stars converging in both runs.

- **CI exists.** This repository had none.

### Notes

- No API removals. `Spectrum` gains an optional `resolution` field and a
  `resolving_power` property; `measure_equivalent_widths` gains `resolution`,
  `vsini`, `vmac`, all defaulting to the previous behaviour.
- Minor version rather than patch: vsini, σ(log g) and line selection all move.
