"""Main SPECIES analysis orchestrator.

Replaces SPECIES.py + CalcParams.py ``run_iteration()`` with a clean Python API.
This is what ExoAutomata and CLI users call to run an analysis.

Usage::

    from species import Spectrum, Analyzer

    spectrum = Spectrum.from_fits("star_feros.fits", instrument="FEROS")
    analyzer = Analyzer(spectrum, output_dir=Path("./output"))
    result = analyzer.run()

    print(result.params.teff)        # 5777.0
    print(result.abundances["NaI"])  # ElementAbundanceResult(...)
    result.to_dict()                 # For database storage
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from species.atmosphere import AtmosphereFitter, AtmosphericParameters
from species.config import Settings
from species.errors import ParameterErrors, compute_errors
from species.ew import EWResult, LineList, create_moog_linelist, measure_equivalent_widths
from species.io.results import (
    AnalysisMetadata,
    AnalysisResult,
    BroadeningResult,
    ElementAbundanceResult,
)
from species.moog.atmosphere_grid import AtmosphereGrid
from species.moog.wrapper import MOOGRunner
from species.spectrum import Spectrum

logger = logging.getLogger(__name__)


class Analyzer:
    """Main SPECIES analysis orchestrator.

    Runs the full spectroscopic parameter pipeline:
    1. Rest-frame correction (CCF)
    2. Equivalent width measurement
    3. Atmospheric parameter fitting (iterative MOOG)
    4. Error propagation
    5. Chemical abundances
    6. Broadening (vsini, vmac)

    Parameters
    ----------
    spectrum
        Input spectrum to analyze.
    output_dir
        Directory for output files and plots. Created if needed.
    config
        SPECIES configuration. If ``None``, uses defaults + environment variables.
    """

    def __init__(
        self,
        spectrum: Spectrum,
        output_dir: Path | None = None,
        config: Settings | None = None,
    ) -> None:
        self.spectrum = spectrum
        self.config = config or Settings()
        self.output_dir = Path(output_dir or self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-init heavy objects
        self._moog: MOOGRunner | None = None
        self._grid: AtmosphereGrid | None = None

        # Overrides (set via configure())
        self._overrides: dict = {}

    def configure(self, **kwargs) -> Analyzer:
        """Override configuration settings for this analysis.

        Parameters
        ----------
        **kwargs
            Any attribute of ``Settings`` (e.g. ``is_giant=True``,
            ``compute_broadening=False``).

        Returns
        -------
        Analyzer
            Self, for chaining.
        """
        for key, val in kwargs.items():
            if hasattr(self.config, key):
                object.__setattr__(self.config, key, val)
            else:
                self._overrides[key] = val
        return self

    @property
    def moog(self) -> MOOGRunner:
        if self._moog is None:
            self._moog = MOOGRunner(
                moog_binary=self.config.moog_binary,
                moog_data_dir=self.config.moog_data_dir,
                timeout=self.config.moog_timeout,
            )
        return self._moog

    @property
    def grid(self) -> AtmosphereGrid:
        if self._grid is None:
            self._grid = AtmosphereGrid.get(self.config.atlas9_dir)
        return self._grid

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(self) -> AnalysisResult:
        """Run the full analysis pipeline.

        Returns
        -------
        AnalysisResult
            Complete results including parameters, errors, abundances,
            and broadening measurements.
        """
        spec = self.spectrum

        # 1. Rest-frame correction
        if self.config.compute_rest_frame and spec.rv is None:
            logger.info("Applying rest-frame correction via CCF")
            spec = spec.to_rest_frame(self.config.mask_path)

        # Ensure 1D for EW measurement
        spec_1d = spec.combine_orders() if spec.wavelength.ndim > 1 else spec

        # 2. Measure equivalent widths
        logger.info("Measuring equivalent widths")
        ew_results = measure_equivalent_widths(
            spec_1d.wavelength,
            spec_1d.flux,
            spec_1d.snr,
            self.config.linelist_path,
            output_dir=self.output_dir,
        )

        valid_ew = [r for r in ew_results if r.is_valid]
        logger.info("Measured %d valid EWs out of %d lines", len(valid_ew), len(ew_results))

        if len(valid_ew) == 0:
            logger.error("No valid EWs measured; cannot determine parameters")
            return AnalysisResult(
                star_name=spec.star_name,
                params=AtmosphericParameters(),
                metadata=AnalysisMetadata(
                    instrument=spec.instrument,
                    rv=spec.rv or 0.0,
                    snr=float(spec.snr) if not hasattr(spec.snr, "__len__") else float(np.median(spec.snr)),
                ),
            )

        # 3. Create MOOG linelist from EWs
        linelist = LineList.from_file(self.config.linelist_path)
        moog_linelist_path = self.output_dir / "moog_linelist.txt"
        n_fe_i, n_fe_ii = create_moog_linelist(
            ew_results, linelist, moog_linelist_path,
        )
        logger.info("MOOG linelist: %d Fe I, %d Fe II lines", n_fe_i, n_fe_ii)

        # 4. Fit atmospheric parameters
        logger.info("Fitting atmospheric parameters")
        fitter = AtmosphereFitter(self.config, self.moog, self.grid)

        initial = self._overrides.get("initial_params")
        hold = self._overrides.get("hold", [])
        boundaries = self._overrides.get("boundaries")

        params = fitter.fit(
            moog_linelist_path,
            initial_params=initial,
            hold=hold,
            boundaries=boundaries,
        )

        logger.info(
            "Parameters: T=%.0f logg=%.2f [Fe/H]=%.2f vt=%.2f (converged=%s)",
            params.teff, params.logg, params.feh, params.vt, params.converged,
        )

        # 5. Error propagation
        errors: ParameterErrors | None = None
        if self.config.compute_errors and params.converged:
            logger.info("Computing parameter uncertainties")
            # Re-run MOOG to get the final abfind result for error analysis
            abfind = self.moog.run_abfind(
                self._make_model(params),
                moog_linelist_path,
                read_mode=self.config.read_mode,
            )
            stellar_class = "giant" if self.config.is_giant else "dwarf"
            errors = compute_errors(params, abfind, stellar_class=stellar_class, hold=hold)

        # 6. Chemical abundances
        abundances: dict[str, ElementAbundanceResult] = {}
        if self.config.compute_abundances and params.converged:
            logger.info("Determining chemical abundances")
            from species.abundances import determine_abundances
            snr_scalar = float(spec_1d.snr) if not hasattr(spec_1d.snr, "__len__") else float(np.median(spec_1d.snr))
            raw_ab = determine_abundances(
                params, spec_1d.wavelength, spec_1d.flux, snr_scalar,
                self.config, self.moog, self.grid,
            )
            abundances = {
                name: ElementAbundanceResult(
                    element=ab.element, abundance=ab.abundance,
                    uncertainty=ab.uncertainty, n_lines=ab.n_lines,
                )
                for name, ab in raw_ab.items()
            }

        # 7. Broadening (vsini, vmac)
        broadening: BroadeningResult | None = None
        if self.config.compute_broadening and params.converged:
            logger.info("Measuring broadening (vsini, vmac)")
            from species.broadening import BroadeningFitter
            bf = BroadeningFitter(self.config, self.moog, self.grid)
            ni_ab = abundances.get("NiI")
            broadening = bf.fit(
                spec_1d.wavelength,
                spec_1d.flux,
                float(spec_1d.snr) if not hasattr(spec_1d.snr, "__len__") else float(np.median(spec_1d.snr)),
                params,
                ni_abundance=ni_ab.abundance if ni_ab else params.feh,
                ni_uncertainty=ni_ab.uncertainty if ni_ab else 0.1,
            )

        # Build metadata
        snr_val = spec.snr
        if hasattr(snr_val, "__len__"):
            snr_val = float(np.median(snr_val))

        metadata = AnalysisMetadata(
            instrument=spec.instrument,
            rv=spec.rv or 0.0,
            snr=float(snr_val),
            method=params.method,
        )

        return AnalysisResult(
            star_name=spec.star_name,
            params=params,
            errors=errors,
            abundances=abundances,
            broadening=broadening,
            ew_results=ew_results,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def run_ew_only(self) -> list[EWResult]:
        """Only compute equivalent widths (skip parameter fitting)."""
        spec = self.spectrum
        if self.config.compute_rest_frame and spec.rv is None:
            spec = spec.to_rest_frame(self.config.mask_path)
        spec_1d = spec.combine_orders() if spec.wavelength.ndim > 1 else spec
        return measure_equivalent_widths(
            spec_1d.wavelength, spec_1d.flux, spec_1d.snr,
            self.config.linelist_path,
        )

    def run_params_only(self) -> AtmosphericParameters:
        """Compute atmospheric parameters without abundances or broadening."""
        saved = (self.config.compute_abundances, self.config.compute_broadening)
        try:
            self.config.compute_abundances = False
            self.config.compute_broadening = False
            result = self.run()
            return result.params
        finally:
            self.config.compute_abundances, self.config.compute_broadening = saved

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    @staticmethod
    def batch(
        spectra: list[Spectrum],
        output_dir: Path | None = None,
        config: Settings | None = None,
        n_cores: int = 1,
        **configure_kwargs,
    ) -> list[AnalysisResult]:
        """Analyze multiple spectra, optionally in parallel.

        Parameters
        ----------
        spectra
            List of Spectrum objects to analyze.
        output_dir
            Base output directory. Each star gets a subdirectory.
        config
            Shared configuration. If ``None``, uses defaults.
        n_cores
            Number of parallel workers. 1 = sequential.
        **configure_kwargs
            Passed to ``configure()`` for each analyzer (e.g. ``is_giant=True``).

        Returns
        -------
        list[AnalysisResult]
            One result per input spectrum, in the same order.

        Usage::

            spectra = [Spectrum.from_fits(f) for f in fits_files]
            results = Analyzer.batch(spectra, n_cores=4)
        """
        config = config or Settings()
        output_dir = Path(output_dir or config.output_dir)

        if n_cores <= 1 or len(spectra) <= 1:
            results = []
            for spec in spectra:
                star_dir = output_dir / spec.star_name
                analyzer = Analyzer(spec, output_dir=star_dir, config=config)
                if configure_kwargs:
                    analyzer.configure(**configure_kwargs)
                try:
                    results.append(analyzer.run())
                except Exception as e:
                    logger.error("Failed to analyze %s: %s", spec.star_name, e)
                    results.append(AnalysisResult(
                        star_name=spec.star_name,
                        params=AtmosphericParameters(),
                        metadata=AnalysisMetadata(instrument=spec.instrument),
                    ))
            return results

        # Parallel execution
        from concurrent.futures import ProcessPoolExecutor, as_completed

        def _run_one(spec_data: tuple) -> AnalysisResult:
            """Worker function — must be top-level picklable."""
            wave, flux, snr, instrument, star_name, star_dir_str, config_dict, kwargs = spec_data
            cfg = Settings(**config_dict)
            spec = Spectrum.from_arrays(wave, flux, snr=snr, instrument=instrument, star_name=star_name)
            analyzer = Analyzer(spec, output_dir=Path(star_dir_str), config=cfg)
            if kwargs:
                analyzer.configure(**kwargs)
            return analyzer.run()

        # Serialize spectra for pickling (numpy arrays + metadata)
        tasks = []
        for spec in spectra:
            star_dir = output_dir / spec.star_name
            # Extract config as dict for serialization
            config_dict = {
                k: str(v) if isinstance(v, Path) else v
                for k, v in config.model_dump().items()
                if v is not None
            }
            tasks.append((
                spec.wavelength, spec.flux, spec.snr,
                spec.instrument, spec.star_name,
                str(star_dir), config_dict, configure_kwargs,
            ))

        results: list[AnalysisResult | None] = [None] * len(spectra)
        with ProcessPoolExecutor(max_workers=min(n_cores, len(spectra))) as pool:
            futures = {
                pool.submit(_run_one, task): i
                for i, task in enumerate(tasks)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                    logger.info("Completed %s", spectra[idx].star_name)
                except Exception as e:
                    logger.error("Failed %s: %s", spectra[idx].star_name, e)
                    results[idx] = AnalysisResult(
                        star_name=spectra[idx].star_name,
                        params=AtmosphericParameters(),
                        metadata=AnalysisMetadata(instrument=spectra[idx].instrument),
                    )

        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _make_model(self, params: AtmosphericParameters) -> Path:
        """Create a temporary atmosphere model file for given parameters."""
        import tempfile
        model_path = Path(tempfile.mktemp(prefix="species_model_", suffix=".atm"))
        self.grid.interpolate_and_write(
            params.teff, params.logg, params.feh, params.vt,
            model_path, fe_solar=self.config.fe_solar,
        )
        return model_path
