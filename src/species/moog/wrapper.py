"""Clean subprocess wrapper around the MOOGSILENT Fortran binary.

Replaces the ``os.system("MOOGSILENT > temp.log 2>&1 <<EOF\\n...")`` pattern
used throughout the original SPECIES codebase. Each invocation runs in an
isolated temporary directory to enable safe parallel execution.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from species.moog.par_file import write_abfind_par, write_synth_par
from species.moog.parser import AbfindResult, parse_abfind_output

logger = logging.getLogger(__name__)


class MOOGError(Exception):
    """Raised when MOOG fails or produces unusable output."""


class MOOGRunner:
    """Interface to the MOOGSILENT binary.

    Each ``run_*`` call executes in a fresh temporary directory so that
    concurrent runs never collide on input/output files.

    Parameters
    ----------
    moog_binary
        Name or absolute path of the MOOGSILENT executable.
    moog_data_dir
        Directory containing MOOG support files (``Barklem.dat``, etc.).
        These are symlinked into every temporary run directory.
    timeout
        Maximum wall-clock seconds per MOOG invocation.
    """

    # MOOG support files that must be present in the working directory
    _SUPPORT_FILES = (
        "Barklem.dat",
        "BarklemUV.dat",
    )

    def __init__(
        self,
        moog_binary: str = "MOOGSILENT",
        moog_data_dir: Path | None = None,
        timeout: int = 300,
    ) -> None:
        self.moog_binary = moog_binary
        self.moog_data_dir = moog_data_dir
        self.timeout = timeout

    # ------------------------------------------------------------------
    # abfind mode
    # ------------------------------------------------------------------

    def run_abfind(
        self,
        model_path: Path,
        linelist_path: Path,
        *,
        read_mode: str = "linearregression",
    ) -> AbfindResult:
        """Run MOOG in ``abfind`` mode and return parsed results.

        Parameters
        ----------
        model_path
            Path to the interpolated ``.atm`` atmosphere model.
        linelist_path
            Path to the MOOG-format line list (with EW column).
        read_mode
            Slope computation mode: ``"linearregression"``, ``"odr"``, ``"moog"``.

        Returns
        -------
        AbfindResult
            Parsed Fe I/II abundances, EP/RW slopes, and per-line data.

        Raises
        ------
        MOOGError
            If MOOG fails to run or produces no output.
        """
        with tempfile.TemporaryDirectory(prefix="species_moog_") as tmpdir:
            run_dir = Path(tmpdir)
            self._link_support_files(run_dir)

            # Copy input files into the isolated run directory
            local_model = run_dir / "model.atm"
            local_lines = run_dir / "lines.txt"
            shutil.copy2(model_path, local_model)
            shutil.copy2(linelist_path, local_lines)

            summary_out = run_dir / "summary.out"
            standard_out = run_dir / "standard.out"

            par_file = write_abfind_par(
                run_dir / "batch.par",
                model_path=local_model,
                linelist_path=local_lines,
                summary_out=summary_out,
                standard_out=standard_out,
            )

            self._execute(par_file, run_dir)

            if not summary_out.exists():
                raise MOOGError(
                    f"MOOG produced no summary output at {summary_out}. "
                    "Check that the atmosphere model and line list are valid."
                )

            return parse_abfind_output(summary_out, mode=read_mode)

    # ------------------------------------------------------------------
    # synth mode
    # ------------------------------------------------------------------

    def run_synth(
        self,
        model_path: Path,
        linelist_path: Path,
        *,
        wave_start: float,
        wave_end: float,
        wave_step: float = 0.02,
        n_species: int = 1,
        species_ids: list[float] | None = None,
        abundances: list[float] | None = None,
        observed_path: Path | None = None,
    ) -> Path:
        """Run MOOG in ``synth`` mode and return path to smoothed output.

        The caller is responsible for parsing the synthesis output since
        the format depends on the use case (broadening fitting, etc.).

        Parameters
        ----------
        model_path
            Path to the interpolated ``.atm`` atmosphere model.
        linelist_path
            Path to the MOOG-format line list.
        wave_start, wave_end
            Synthesis wavelength range in Angstroms.
        species_ids
            Atomic species IDs for abundance variation.
        abundances
            Abundance offsets for each species.

        Returns
        -------
        Path
            Path to a *copy* of the MOOG smoothed output file. The caller
            owns this file and should clean it up when done.

        Raises
        ------
        MOOGError
            If MOOG fails to run or produces no output.
        """
        with tempfile.TemporaryDirectory(prefix="species_synth_") as tmpdir:
            run_dir = Path(tmpdir)
            self._link_support_files(run_dir)

            local_model = run_dir / "model.atm"
            local_lines = run_dir / "lines.txt"
            shutil.copy2(model_path, local_model)
            shutil.copy2(linelist_path, local_lines)

            local_obs = None
            if observed_path is not None:
                local_obs = run_dir / "observed.dat"
                shutil.copy2(observed_path, local_obs)

            summary_out = run_dir / "summary.out"
            standard_out = run_dir / "standard.out"
            smoothed_out = run_dir / "smoothed.out"

            par_file = write_synth_par(
                run_dir / "batch.par",
                model_path=local_model,
                linelist_path=local_lines,
                observed_path=local_obs,
                summary_out=summary_out,
                standard_out=standard_out,
                smoothed_out=smoothed_out,
                wave_start=wave_start,
                wave_end=wave_end,
                wave_step=wave_step,
                n_species=n_species,
                species_ids=species_ids,
                abundances=abundances,
            )

            self._execute(par_file, run_dir)

            if not smoothed_out.exists():
                raise MOOGError(
                    f"MOOG produced no smoothed output at {smoothed_out}. "
                    "Synthesis may have failed."
                )

            # Copy the result out before the tmpdir is deleted
            persistent_out = Path(tempfile.mktemp(prefix="species_synth_", suffix=".out"))
            shutil.copy2(smoothed_out, persistent_out)
            return persistent_out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute(self, par_file: Path, run_dir: Path) -> None:
        """Execute MOOGSILENT with the given parameter file."""
        stdin_text = f"{par_file.name}\n\n"

        logger.debug("Running MOOG in %s with par file %s", run_dir, par_file.name)

        try:
            proc = subprocess.run(
                [self.moog_binary],
                input=stdin_text,
                cwd=run_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except FileNotFoundError:
            raise MOOGError(
                f"MOOG binary '{self.moog_binary}' not found. "
                "Ensure MOOGSILENT is installed and on your PATH."
            ) from None
        except subprocess.TimeoutExpired:
            raise MOOGError(
                f"MOOG timed out after {self.timeout}s. "
                "The model may not converge for these parameters."
            ) from None

        if proc.returncode != 0:
            logger.warning(
                "MOOG exited with code %d. stderr: %s",
                proc.returncode,
                proc.stderr[:500] if proc.stderr else "(empty)",
            )

    def _link_support_files(self, run_dir: Path) -> None:
        """Symlink MOOG support files into the run directory."""
        if self.moog_data_dir is None:
            return

        for fname in self._SUPPORT_FILES:
            src = self.moog_data_dir / fname
            if src.exists():
                dst = run_dir / fname
                if not dst.exists():
                    dst.symlink_to(src)

    def open_session(self, linelist_path: Path) -> MOOGSession:
        """Create a persistent working directory for a sequence of MOOG calls.

        Use this instead of ``run_abfind()`` when you need to run MOOG many
        times with the same linelist but different atmosphere models (e.g.,
        during the iterative convergence loop). Avoids creating/destroying
        a temporary directory for each call.

        Usage::

            with runner.open_session(linelist_path) as session:
                for params in parameter_iterations:
                    grid.interpolate_and_write(..., session.model_path)
                    result = session.run_abfind()
        """
        return MOOGSession(self, linelist_path)


class MOOGSession:
    """Persistent working directory for a sequence of MOOG calls.

    Eliminates per-call overhead of creating temp directories, symlinking
    support files, and copying the linelist. Only the atmosphere model
    (``.atm`` file) is overwritten between iterations.

    Typical speedup: 2-3x for convergence loops with 100+ iterations.
    """

    def __init__(self, runner: MOOGRunner, linelist_path: Path) -> None:
        self._runner = runner
        self._tmpdir = tempfile.TemporaryDirectory(prefix="species_session_")
        self._run_dir = Path(self._tmpdir.name)

        # Setup once: support files + linelist
        runner._link_support_files(self._run_dir)
        self._linelist = self._run_dir / "lines.txt"
        shutil.copy2(linelist_path, self._linelist)

        # Reusable paths
        self.model_path = self._run_dir / "model.atm"
        self._summary = self._run_dir / "summary.out"
        self._standard = self._run_dir / "standard.out"
        self._par_file = self._run_dir / "batch.par"

    def __enter__(self) -> MOOGSession:
        return self

    def __exit__(self, *exc) -> None:
        self._tmpdir.cleanup()

    def run_abfind(self, read_mode: str = "linearregression") -> AbfindResult:
        """Run MOOG abfind using the model currently at ``self.model_path``.

        The caller must write the atmosphere model to ``self.model_path``
        before each call (e.g., via ``grid.interpolate_and_write()``).

        Returns
        -------
        AbfindResult
        """
        # Remove stale output
        self._summary.unlink(missing_ok=True)
        self._standard.unlink(missing_ok=True)

        write_abfind_par(
            self._par_file,
            model_path=self.model_path,
            linelist_path=self._linelist,
            summary_out=self._summary,
            standard_out=self._standard,
        )

        self._runner._execute(self._par_file, self._run_dir)

        if not self._summary.exists():
            raise MOOGError("MOOG produced no summary output in session")

        return parse_abfind_output(self._summary, mode=read_mode)
