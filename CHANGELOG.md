# Changelog

## 4.2.0 — 2026-11-20

Abundances: Mn and Cu were overestimated (no hyperfine structure, a phantom and
two blended Mn lines), and the abundance table's Fe rows sat ~0.33 dex above the
fitted [Fe/H] (blend satellites fitted as separate lines). Both changed reported
numbers silently; Mn, Cu and table Fe from 4.1.0 and earlier should not be used.
Also: solver convergence diagnostics, a MAROON-X reader, and a working
`species install-moog`.

Minor rather than patch: abundances and their errors move.

### Wrong numbers, no error raised

- **Mn and Cu were overestimated (SPEC-2).** Their lines are hyperfine-split
  but went through MOOG `abfind` as single lines. They now go through MOOG
  `blends` with their HFS (and Cu isotopic) components from
  `data/linelists/hfs_mn_cu.dat`. The patterns come from the Gaia-ESO line list
  (Heiter et al. 2021), or from Kurucz for Mn I 5413.67, which Gaia-ESO lists
  unsplit. The total log gf and the wavelength of each line are unchanged. Two
  bad Mn lines were removed from `lines_ab.dat`. The first, 5255.380 (EP
  5.52 eV, log gf −8.78), took the EW of Mn I 5255.33 and returned
  A(Mn) ≈ 15.7. Sigma clipping hid it only when there were three or more lines.
  The second, 5505.866, is dominated by Fe I 5505.88 and gave +1.1 dex in the
  Sun. On the HARPS solar spectrum, at SPECIES' converged parameters, [Mn/H]
  went from +0.42 to −0.03 and [Cu/H] from +0.32 to +0.00. Mn I 5537.75 was then removed as well: Ti I 5537.85, of comparable
  strength, blends it, and it was the high outlier in the Sun and in a metal-rich
  G dwarf (+0.9 dex). With two Mn lines sigma clipping cannot reject it. Mn now
  rests on 5004.89, 5117.93, 5255.33, 5377.61 and 5413.67.

- **The abundance table's Fe sat 0.2–0.4 dex above the fitted [Fe/H]**
  (SPEC-4). On the HARPS Sun at the converged 5829/4.52/−0.00/0.70 it read
  Fe I +0.33, Fe II +0.34. Two causes. (1) `lines_ab.dat` carries
  synthesis-style component lists: satellites such as Fe I 5543.974
  (log gf −6.06) beside 5543.936 (−1.14), or Fe II entries at 10.4 eV. Each
  entry was measured as a line of its own. The Gaussian fit found the same
  feature for every one of them, MOOG returned +1 to +9 dex for the satellites,
  and the single 1.5σ clip kept enough of them to lift the mean. Same-species
  entries within 0.10 Å are now folded into one line: the strongest component
  (log gf − θχ at the star's Teff) carries the EW, with an effective
  log gf = log Σ gf·10^(−θΔχ). That is exact for a same-EP pair such as
  Na I 5688.19/5688.21, and a negligible change when the rest are satellites.
  It is applied to every element except the HFS lines. Fe I goes from 52
  entries to 41 features and Fe II from 15 to 10; Ti I and Cr I lose their
  satellites too. (2) The real Fe I lines of `lines_ab.dat` have log gf a median
  0.13 dex below linelist.dat's solar-calibrated values for the same
  transitions. The Fe I and Fe II rows therefore now come from the solver's
  final abfind, on the lines and EWs [Fe/H] was fitted on
  (`determine_abundances(..., fe_abfind=)`, passed by `Analyzer`). The
  abundance-list Fe lines remain only as a fallback. Results: on the Sun,
  Fe I +0.000 ± 0.005 (101 lines) and Fe II −0.015 ± 0.013. On a MAROON-X epoch
  fitted at [Fe/H] = 0.184 ± 0.050, Fe I +0.187 and Fe II +0.168; it was
  +0.387 and +0.576.

- **Abundance uncertainties are the measured line scatter / √n** (SPEC-4).
  They used to be a fixed 0.2/√n. The scatter is floored at 0.05 dex. With
  fewer than three lines, where it cannot be estimated, the old 0.2 dex per
  line is kept as the floor.

### Fixed

- **`species install-moog` produced a binary that only worked from
  /moog17scat** (SPEC-5). MOOG opens `Barklem.dat`/`BarklemUV.dat` via a path
  compiled in from `moogpath` in `Moog.f`/`Moogsilent.f` (`'/moog17scat'`).
  The installer now rewrites it to the install directory before `make`
  (prefixes over 58 characters are refused, the Fortran buffer is 60). No
  `/moog17scat` symlink needed.

### Added

- **Solver convergence diagnostics** (SPEC-1). `AtmosphericParameters` and
  `AnalysisResult.to_dict()` now carry `final_residuals` (EP slope, Fe I − Fe II,
  [Fe/H] derived − input, RW slope), `n_iters` (MOOG evaluations), `hit_iter_cap`,
  `last_moog_status` and `n_fe_lines_input`, for every solver. Unknown is None,
  never 0.

- **MAROON-X (Gemini-N) reader** (SPEC-3). `Spectrum.from_fits` (and so the CLI
  and `Analyzer.batch`) routes `*.hd5` pandas stores to `Spectrum.from_maroonx`:
  fiber 6, blue order 91 dropped, pixels masked by variance (not flux), edge
  trimming, S/N from flux/sqrt(var), R = 85000. The DRS wavelengths are
  **vacuum**; they are converted to air for the SPECIES line lists. The
  instrument response is optional (`response=` or `MAROONX_RESPONSE`); without
  it each order is flattened by a polynomial envelope. New extra:
  `pip install astro-species[maroonx]` (pandas, PyTables).

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
