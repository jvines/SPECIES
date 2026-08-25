"""Iterative atmospheric parameter determination via MOOG.

Ported from Atmos.py. Determines Teff, logg, [Fe/H], and microturbulence (vt)
by iteratively running MOOG until Fe I/II excitation and ionization balance
converge. Supports both per-parameter adjustment and Nelder-Mead simplex.

This is the heart of SPECIES — the convergence logic, safety checks, and
retry strategies are preserved faithfully from the original.
"""

from __future__ import annotations

import logging
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from species.config import AtmosQuantity, Settings, Tolerance
from species.moog.atmosphere_grid import AtmosphereGrid
from species.moog.parser import AbfindResult, parse_abfind_output
from species.moog.par_file import write_abfind_par
from species.moog.wrapper import MOOGError, MOOGRunner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AtmosphericParameters:
    """Converged atmospheric parameters from iterative MOOG fitting."""

    teff: float = 0.0
    logg: float = 0.0
    feh: float = 0.0  # [Fe/H]
    vt: float = 0.0  # microturbulence (km/s)
    n_fe_i: int = 0
    n_fe_ii: int = 0
    fe_i_abundance: float = 0.0
    fe_ii_abundance: float = 0.0
    converged: bool = False
    method: str = "per_parameter"


# ---------------------------------------------------------------------------
# Convergence state machine (ported from atmos class)
# ---------------------------------------------------------------------------

class _ConvergenceState:
    """Internal state for the iterative convergence loop.

    Tracks parameter values, MOOG outputs, iteration counts, and all the
    safety-check counters from the original code.
    """

    # Default parameter boundaries
    DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
        "metallicity": (-3.0, 1.0),
        "temperature": (3500.0, 9000.0),
        "gravity": (0.5, 4.9),
        "velocity": (0.0, 5.0),
    }

    PARAM_NAMES = ("metallicity", "temperature", "gravity", "velocity")

    def __init__(
        self,
        initial: tuple[float, float, float, float],
        hold: list[str],
        tolerance: Tolerance,
        boundaries: dict[str, tuple[float, float]] | None = None,
        max_repeat: int = 200,
    ) -> None:
        # Merge boundaries with defaults
        bounds = dict(self.DEFAULT_BOUNDS)
        if boundaries:
            for name, (lo, hi) in boundaries.items():
                if name in bounds:
                    dlo, dhi = bounds[name]
                    bounds[name] = (max(lo, dlo), min(hi, dhi))

        feh, teff, logg, vt = initial
        self.params = {
            "metallicity": AtmosQuantity("metallicity", feh, "metallicity" in hold,
                                          [-999.0, -999.0, -999.0], bounds["metallicity"],
                                          0.25, tolerance.ab, 0.0),
            "temperature": AtmosQuantity("temperature", teff, "temperature" in hold,
                                          [-999.0, -999.0], bounds["temperature"],
                                          50.0, tolerance.ep, 200.0),
            "gravity": AtmosQuantity("gravity", logg, "gravity" in hold,
                                      [-999.0, -999.0], bounds["gravity"],
                                      0.25, tolerance.dif, 0.2),
            "velocity": AtmosQuantity("velocity", vt, "velocity" in hold,
                                       [-999.0, -999.0], bounds["velocity"],
                                       0.25, tolerance.rw, 0.2),
        }
        self.tol = tolerance
        self.moog = [0.0, 0.0, 0.0, 0.0]  # [ab, ep, dif, rw]
        self.n_failed = 0
        self.n_break = 0
        self.n_it = 0
        self.n_it_total = 0
        self.n_repeat = max_repeat
        self.exception = 1  # 1 = converged, 2 = failed
        self.change = "metallicity"
        self.change_prev = "metallicity"
        self.history: list[list[float]] = []
        self._last_result: AbfindResult | None = None  # Track latest MOOG result

    @property
    def values(self) -> tuple[float, float, float, float]:
        """Current (feh, teff, logg, vt)."""
        return (
            self.params["metallicity"].value,
            self.params["temperature"].value,
            self.params["gravity"].value,
            self.params["velocity"].value,
        )

    def is_converged(self) -> bool:
        """Check if all four convergence criteria are met."""
        feh = self.params["metallicity"].value
        return (
            abs(self.moog[0] - feh) <= self.tol.ab
            and abs(self.moog[1]) <= self.tol.ep
            and abs(self.moog[2]) <= self.tol.dif
            and abs(self.moog[3]) <= self.tol.rw
        )

    def is_failed(self) -> bool:
        """Check terminal failure conditions."""
        if self.n_break > 5:
            self.exception = 2
            return True
        if self.n_it >= self.n_repeat:
            self.exception = 2
            return True
        if self.n_it_total >= 500_000:
            self.exception = 2
            return True
        # Check if too many params are out of bounds
        n_out = sum(
            1 for name in ("metallicity", "temperature", "gravity")
            if not (self.params[name].bounds[0] <= self.params[name].value <= self.params[name].bounds[1])
        )
        if n_out >= 3:
            self.exception = 2
            return True
        return False

    def record_moog_output(self, ab: float, ep: float, dif: float, rw: float, n_failed: int) -> None:
        """Store the latest MOOG output."""
        self.moog = [ab, ep, dif, rw]
        self.n_failed = n_failed
        # Zero out held parameters
        for i, name in enumerate(self.PARAM_NAMES):
            if self.params[name].hold:
                self.moog[i] = 0.0 if i != 0 else self.params["metallicity"].value

    def perturb_on_failure(self) -> None:
        """Randomly perturb non-held parameters when MOOG fails."""
        if self.n_failed <= 0:
            return
        for p in self.params.values():
            if not p.hold:
                new_val = np.random.normal(p.value, p.width)
                new_val = np.clip(new_val, p.bounds[0], p.bounds[1])
                p.value = float(new_val)
                p.ranges = [-999.0, -999.0] if len(p.ranges) == 2 else [-999.0, -999.0, p.value]

    def advance_parameter(self) -> None:
        """Move to the next parameter in the cycle."""
        order = ["metallicity", "temperature", "pressure", "velocity"]
        # "pressure" maps to "gravity" in the MOOG output indexing
        idx = order.index(self.change) if self.change in order else 0
        self.change = order[(idx + 1) % 4]

    def new_iteration(self) -> None:
        """Book-keeping for a new iteration."""
        self.n_it_total += 1
        self.history.append(list(self.values))
        self.change_prev = self.change

    def check_repeat(self) -> None:
        """If current params were already tried, advance to next parameter."""
        if list(self.values) in self.history:
            self.advance_parameter()
            self.n_it += 1

    def check_nfailed_on_change(self, change: str) -> None:
        """Increment break counter if the last MOOG call failed."""
        if self.n_failed > 0:
            self.advance_parameter()
            self.n_break += 1

    def check_nbreak(self) -> bool:
        """Return True if we've exceeded the failure limit."""
        if self.n_break > 5:
            self.exception = 2
            return True
        return False

    def check_nrepeat(self) -> bool:
        """Return True if parameters have been repeated too many times."""
        if self.n_it >= self.n_repeat:
            self.exception = 2
            return True
        return False

    def check_nit_total(self) -> bool:
        """Return True if total iterations exceeded."""
        if self.n_it_total >= 500_000:
            self.exception = 2
            return True
        return False


# ---------------------------------------------------------------------------
# Main fitter
# ---------------------------------------------------------------------------

class AtmosphereFitter:
    """Iterative atmospheric parameter determination via MOOG.

    Usage::

        fitter = AtmosphereFitter(config, moog_runner, atm_grid)
        result = fitter.fit(linelist_path, initial_params=(0.0, 5500, 4.36, 1.23))
    """

    def __init__(
        self,
        config: Settings,
        moog: MOOGRunner,
        grid: AtmosphereGrid,
    ) -> None:
        self.config = config
        self.moog = moog
        self.grid = grid

    def fit(
        self,
        linelist_path: Path,
        initial_params: tuple[float, float, float, float] | None = None,
        hold: list[str] | None = None,
        boundaries: dict[str, tuple[float, float]] | None = None,
        method: str | None = None,
    ) -> AtmosphericParameters:
        """Run iterative fitting to determine atmospheric parameters.

        Parameters
        ----------
        linelist_path
            Path to the MOOG-format line list (with EW column).
        initial_params
            Starting (feh, teff, logg, vt). Defaults to (0.0, 5500, 4.36, 1.23).
        hold
            List of parameter names to hold fixed (e.g. ``["gravity"]``).
        boundaries
            Override parameter boundaries.
        method
            ``"per_parameter"`` (default) or ``"downhill_simplex"``.

        Returns
        -------
        AtmosphericParameters
            Converged parameters, or best-effort if convergence failed.
        """
        if initial_params is None:
            initial_params = (0.0, 5500.0, 4.36, 1.23)
        if hold is None:
            hold = []
        if method is None:
            method = self.config.minimization

        if method == "broyden":
            return self._fit_broyden(linelist_path, initial_params, hold, boundaries)
        if method == "downhill_simplex":
            return self._fit_simplex(linelist_path, initial_params, hold, boundaries)
        if method == "per_parameter":
            return self._fit_per_parameter(linelist_path, initial_params, hold, boundaries)
        # Default: try Broyden first, fall back to per_parameter
        result = self._fit_broyden(linelist_path, initial_params, hold, boundaries)
        if result.converged:
            return result
        logger.info("Broyden did not converge, falling back to per_parameter")
        return self._fit_per_parameter(linelist_path, initial_params, hold, boundaries)

    # ------------------------------------------------------------------
    # Per-parameter method
    # ------------------------------------------------------------------

    def _fit_per_parameter(
        self,
        linelist_path: Path,
        initial: tuple[float, float, float, float],
        hold: list[str],
        boundaries: dict[str, tuple[float, float]] | None,
    ) -> AtmosphericParameters:
        """Iterative per-parameter convergence loop.

        Faithful port of the original Atmos.py calc_params() + Metallicity(),
        Temperature(), Pressure(), Velocity() functions. Each parameter runs
        its OWN inner convergence loop, then hands off to the next.

        Uses MOOGSession to reuse the working directory across all iterations.
        """
        state = _ConvergenceState(
            initial, hold, self.config.tolerance, boundaries,
        )

        with self.moog.open_session(linelist_path) as session:
            # Initial MOOG run
            result = self._session_run_moog(state, session)
            if result is None:
                return self._failed_result(state)

            ab = result.fe_i_abundance - self.config.fe_solar
            state.record_moog_output(ab, result.ep_slope, result.fe_i_fe_ii_diff, result.rw_slope, result.n_failed)
            state.params["metallicity"].ranges[2] = ab

            # Metallicity range index (tracks which slot in the 3-element ranges array to write)
            met_idx = 0

            # Outer loop: cycle through parameters
            while True:
                state.new_iteration()

                # Check overall convergence
                if state.is_converged():
                    feh, teff, logg, vt = state.values
                    logger.info(
                        "Converged after %d iterations: T=%.0f logg=%.2f [Fe/H]=%.2f vt=%.2f",
                        state.n_it_total, teff, logg, feh, vt,
                    )
                    return AtmosphericParameters(
                        teff=teff, logg=logg, feh=feh, vt=vt,
                        n_fe_i=result.n_fe_i, n_fe_ii=result.n_fe_ii,
                        fe_i_abundance=result.fe_i_abundance,
                        fe_ii_abundance=result.fe_ii_abundance,
                        converged=True, method="per_parameter",
                    )

                if state.is_failed():
                    break

                change = state.change

                if change == "metallicity":
                    met_idx = self._converge_metallicity(state, session)
                    state.check_nfailed_on_change(change)
                elif change == "temperature":
                    met_idx = self._converge_temperature(state, session, met_idx)
                    state.check_nfailed_on_change(change)
                elif change == "pressure":
                    met_idx = self._converge_pressure(state, session, met_idx)
                    state.check_nfailed_on_change(change)
                else:  # velocity
                    met_idx = self._converge_velocity(state, session, met_idx)
                    state.check_nfailed_on_change(change)

                state.perturb_on_failure()
                if state.check_nbreak():
                    break
                state.check_repeat()
                if state.check_nrepeat():
                    break
                if state.check_nit_total():
                    break

                # Keep result from latest MOOG call
                result = state._last_result or result

        return self._failed_result(state)

    # ------------------------------------------------------------------
    # Per-parameter inner convergence loops (ported from Atmos.py)
    # ------------------------------------------------------------------

    def _converge_metallicity(self, state: _ConvergenceState, session) -> int:
        """Run metallicity to convergence: adjust [Fe/H] until derived == input.

        Ported from Atmos.py Metallicity().
        """
        c = 0
        for _ in range(100_000):
            state.n_it_total += 1
            feh_param = state.params["metallicity"]
            ab = state.moog[0]

            if abs(ab - feh_param.value) <= state.tol.ab:
                state.advance_parameter()
                break

            if c > 50:
                state.advance_parameter()
                break

            # Set [Fe/H] = derived abundance
            feh_antes = feh_param.value
            feh_param.value = float(np.clip(ab, feh_param.bounds[0], feh_param.bounds[1]))

            result = self._session_run_moog(state, session)
            if result is None or state.n_failed > 0:
                state.advance_parameter()
                feh_param.value = feh_antes
                break

            ab = result.fe_i_abundance - self.config.fe_solar
            state.record_moog_output(ab, result.ep_slope, result.fe_i_fe_ii_diff, result.rw_slope, result.n_failed)
            state._last_result = result
            c += 1

        # After metallicity converges, reset ranges for other params
        state.params["metallicity"].ranges = [-999.0, -999.0, state.moog[0]]
        state.params["temperature"].ranges = [-999.0, -999.0]
        state.params["gravity"].ranges = [-999.0, -999.0]
        state.params["velocity"].ranges = [-999.0, -999.0]
        return 0

    def _converge_temperature(self, state: _ConvergenceState, session, met_idx: int) -> int:
        """Run temperature to convergence: adjust Teff until EP slope ~ 0.

        Ported from Atmos.py Temperature().
        """
        for _ in range(100_000):
            state.n_it_total += 1
            ep = state.moog[1]

            if abs(ep) <= state.tol.ep:
                ab = state.moog[0]
                if np.isclose(ab, state.params["metallicity"].value, atol=2 * state.tol.ab):
                    state.advance_parameter()  # T → pressure
                else:
                    state.change = "metallicity"  # ab drifted, re-converge [Fe/H]
                break

            t_param = state.params["temperature"]
            t_antes = t_param.value
            _bisect_parameter(t_param, ep, step=250.0, decimals=0)

            if t_param.value < t_param.bounds[0] or t_param.value > t_param.bounds[1]:
                state.advance_parameter()
                t_param.value = t_antes
                break

            result = self._session_run_moog(state, session)
            if result is None or state.n_failed > 0:
                state.advance_parameter()
                break

            ab = result.fe_i_abundance - self.config.fe_solar
            state.record_moog_output(ab, result.ep_slope, result.fe_i_fe_ii_diff, result.rw_slope, result.n_failed)
            state._last_result = result

            # Track metallicity convergence
            state.params["metallicity"].ranges[met_idx] = ab
            met_idx = (met_idx + 1) % 3

            # Check if metallicity has cycled (same value 3 times)
            if self._check_met_cycle(state):
                state.change = "metallicity"  # new_change('velocity') = metallicity
                break

        return met_idx

    def _converge_pressure(self, state: _ConvergenceState, session, met_idx: int) -> int:
        """Run logg to convergence: adjust logg until Fe I-II difference ~ 0.

        Ported from Atmos.py Pressure().
        """
        for _ in range(100_000):
            state.n_it_total += 1
            dif = state.moog[2]

            if abs(dif) <= state.tol.dif:
                ab = state.moog[0]
                if np.isclose(ab, state.params["metallicity"].value, atol=2 * state.tol.ab):
                    state.advance_parameter()  # P → velocity
                else:
                    state.change = "metallicity"  # ab drifted, re-converge [Fe/H]
                break

            g_param = state.params["gravity"]
            g_antes = g_param.value
            _bisect_parameter(g_param, dif, step=0.25, decimals=5)

            if g_param.value < g_param.bounds[0] or g_param.value > g_param.bounds[1]:
                state.advance_parameter()
                g_param.value = g_antes
                break

            result = self._session_run_moog(state, session)
            if result is None or state.n_failed > 0:
                state.advance_parameter()
                break

            ab = result.fe_i_abundance - self.config.fe_solar
            state.record_moog_output(ab, result.ep_slope, result.fe_i_fe_ii_diff, result.rw_slope, result.n_failed)
            state._last_result = result

            state.params["metallicity"].ranges[met_idx] = ab
            met_idx = (met_idx + 1) % 3

            if self._check_met_cycle(state):
                state.change = "metallicity"  # new_change('velocity') = metallicity
                break

        return met_idx

    def _converge_velocity(self, state: _ConvergenceState, session, met_idx: int) -> int:
        """Run vt to convergence: adjust vt until RW slope ~ 0.

        Ported from Atmos.py Velocity(). Hard limit of 50 iterations.
        """
        for v in range(50):
            state.n_it_total += 1
            rw = state.moog[3]

            if abs(rw) <= state.tol.rw:
                ab = state.moog[0]
                if np.isclose(ab, state.params["metallicity"].value, atol=state.tol.ab):
                    state.change = "temperature"  # next_change('metallicity') = temperature
                else:
                    state.change = "metallicity"  # ab drifted, re-converge [Fe/H]
                break

            vt_param = state.params["velocity"]
            vt_antes = vt_param.value
            _bisect_parameter(vt_param, rw, step=0.25, decimals=5)

            if vt_param.value < vt_param.bounds[0] or vt_param.value > vt_param.bounds[1]:
                state.advance_parameter()
                vt_param.value = vt_antes
                break

            result = self._session_run_moog(state, session)
            if result is None or state.n_failed > 0:
                state.advance_parameter()
                break

            ab = result.fe_i_abundance - self.config.fe_solar
            state.record_moog_output(ab, result.ep_slope, result.fe_i_fe_ii_diff, result.rw_slope, result.n_failed)
            state._last_result = result

            state.params["metallicity"].ranges[met_idx] = ab
            met_idx = (met_idx + 1) % 3

            if self._check_met_cycle(state):
                state.change = "metallicity"  # new_change('velocity') = metallicity
                break
        else:
            # Hit 50-iteration limit
            state.advance_parameter()

        return met_idx

    # ------------------------------------------------------------------
    # Shared helpers for the convergence loops
    # ------------------------------------------------------------------

    def _session_run_moog(self, state: _ConvergenceState, session) -> AbfindResult | None:
        """Run MOOG via session, handling grid interpolation."""
        feh, teff, logg, vt = state.values
        ok = self.grid.interpolate_and_write(
            teff, logg, feh, vt, session.model_path,
            fe_solar=self.config.fe_solar,
        )
        if not ok:
            state.n_break += 1
            return None
        try:
            result = session.run_abfind(read_mode=self.config.read_mode)
            logger.debug(
                "MOOG [%s]: feh=%.2f T=%.0f logg=%.2f vt=%.2f → ab=%.3f ep=%.3f dif=%.3f rw=%.3f",
                state.change, feh, teff, logg, vt,
                result.fe_i_abundance - self.config.fe_solar,
                result.ep_slope, result.fe_i_fe_ii_diff, result.rw_slope,
            )
            return result
        except MOOGError:
            state.n_break += 1
            return None

    @staticmethod
    def _check_met_cycle(state: _ConvergenceState) -> bool:
        """Check if metallicity ranges have cycled (same value 3 times).

        Uses np.isclose instead of exact == to handle float64 precision
        differences between MOOG runs (the original Python 2 code used
        Python floats where exact equality was more likely).
        """
        r = state.params["metallicity"].ranges
        if r[0] == -999.0 or r[1] == -999.0 or r[2] == -999.0:
            return False
        atol = state.tol.ab
        return (
            np.isclose(r[0], r[1], atol=atol)
            and np.isclose(r[1], r[2], atol=atol)
            and not np.isclose(r[0], state.params["metallicity"].value, atol=atol)
        )

    @staticmethod
    def _failed_result(state: _ConvergenceState) -> AtmosphericParameters:
        feh, teff, logg, vt = state.values
        logger.warning("Failed to converge after %d iterations", state.n_it_total)
        return AtmosphericParameters(
            teff=teff, logg=logg, feh=feh, vt=vt,
            converged=False, method="per_parameter",
        )

    # ------------------------------------------------------------------
    # Broyden's method (quasi-Newton, path-independent convergence)
    # ------------------------------------------------------------------

    # Perturbation step sizes for finite-difference Jacobian
    _FD_STEPS = np.array([100.0, 0.2, 0.1, 0.2])

    # Max step per iteration (trust region)
    _MAX_STEPS = np.array([500.0, 0.5, 0.3, 0.5])

    # Parameter bounds.
    #
    # NOTE these disagree with _ConvergenceState.DEFAULT_BOUNDS, which the
    # per-parameter method uses: [Fe/H] is (-4.0, 0.6) here against (-3.0, 1.0)
    # there, and vt (0.1, 5.0) against (0.0, 5.0). Two methods on the same class
    # therefore search different spaces. Left as-is rather than unified in this
    # pass, because changing a bound moves published numbers and that deserves
    # its own change with its own validation.
    _BOUNDS = np.array([
        [3500.0, 9000.0],  # Teff
        [0.5, 4.9],        # logg
        [-4.0, 0.6],       # [Fe/H]
        [0.1, 5.0],        # vt
    ])

    # The solver's parameter vector is [Teff, logg, [Fe/H], vt] -- note this is
    # NOT the order of `initial`, which is (feh, teff, logg, vt). Named here so
    # `hold` can be mapped onto vector positions without re-deriving it.
    _SOLVER_PARAM_ORDER = ("temperature", "gravity", "metallicity", "velocity")

    def _fit_broyden(
        self,
        linelist_path: Path,
        initial: tuple[float, float, float, float],
        hold: list[str],
        boundaries: dict[str, tuple[float, float]] | None,
    ) -> AtmosphericParameters:
        """Broyden's first method for atmospheric parameter determination.

        Solves the 4x4 nonlinear system:
            f1 = EP_slope → 0          (excitation balance)
            f2 = FeI - FeII → 0        (ionization balance)
            f3 = derived_feh - input_feh → 0  (abundance consistency)
            f4 = RW_slope → 0          (microturbulence balance)

        Typically converges in 10-20 MOOG calls vs 100-200 for bisection.
        Uses finite-difference initial Jacobian + rank-1 Broyden updates.
        """
        feh0, teff0, logg0, vt0 = initial
        params = np.array([teff0, logg0, feh0, vt0], dtype=float)
        params = np.clip(params, self._BOUNDS[:, 0], self._BOUNDS[:, 1])
        n_calls = 0
        last_result: AbfindResult | None = None

        # `hold` was accepted here and never used, while compute_errors DID
        # honour it -- so hold=["gravity"] returned a freely fitted log g
        # wearing the uncertainty of a held one. A fabricated error bar in an
        # output table is worse than a missing feature.
        #
        # Each residual pairs with exactly one parameter (EP slope <-> Teff,
        # Fe I - Fe II <-> log g, abundance consistency <-> [Fe/H], RW slope
        # <-> vt), so holding a parameter drops its residual too and the system
        # stays square.
        free = np.array([name not in hold for name in self._SOLVER_PARAM_ORDER])
        idx = np.where(free)[0]
        if idx.size == 0:
            raise ValueError("every parameter is held; there is nothing to solve")
        if idx.size < 4:
            logger.info("Broyden: holding %s", ", ".join(sorted(hold)))

        with self.moog.open_session(linelist_path) as session:
            def moog_func(p: np.ndarray) -> tuple[np.ndarray, AbfindResult | None]:
                """Evaluate the 4 diagnostics at parameter vector p."""
                teff, logg, feh, vt = p
                ok = self.grid.interpolate_and_write(
                    teff, logg, feh, vt, session.model_path,
                    fe_solar=self.config.fe_solar,
                )
                if not ok:
                    return np.array([1e10, 1e10, 1e10, 1e10]), None
                try:
                    r = session.run_abfind(read_mode=self.config.read_mode)
                except MOOGError:
                    return np.array([1e10, 1e10, 1e10, 1e10]), None

                ab = r.fe_i_abundance - self.config.fe_solar
                return np.array([r.ep_slope, r.fe_i_fe_ii_diff, ab - feh, r.rw_slope]), r

            # Initial evaluation
            f0, last_result = moog_func(params)
            n_calls += 1
            quadr = np.linalg.norm(f0[idx])

            # Convergence threshold: L2 norm of 4 residuals. Each residual
            # should be < tol (0.001), so ||f|| < sqrt(4)*tol ≈ 0.002. Use 0.005
            # for robustness (individual components may be slightly above tol).
            convergence_tol = 5.0 * self.config.tolerance.ab

            if quadr < convergence_tol:
                return self._broyden_result(params, last_result, True, n_calls, "broyden")

            # Compute initial Jacobian via finite differences (4 extra calls)
            J = np.eye(4)
            for j in idx:
                p_pert = params.copy()
                p_pert[j] += self._FD_STEPS[j]
                p_pert = np.clip(p_pert, self._BOUNDS[:, 0], self._BOUNDS[:, 1])
                step = p_pert[j] - params[j]
                if abs(step) < 1e-10:
                    p_pert[j] = params[j] - self._FD_STEPS[j]
                    p_pert = np.clip(p_pert, self._BOUNDS[:, 0], self._BOUNDS[:, 1])
                    step = p_pert[j] - params[j]
                f_pert, _ = moog_func(p_pert)
                n_calls += 1
                J[:, j] = (f_pert - f0) / step

            logger.info(
                "Broyden init: ||f||=%.4f at T=%.0f logg=%.2f feh=%.3f vt=%.3f (%d calls)",
                quadr, *params, n_calls,
            )

            # Main Broyden iteration loop
            stagnation = 0
            for iteration in range(1, 30):
                # Solve J @ delta = -f
                try:
                    delta = np.zeros(4)
                    delta[idx] = -np.linalg.solve(J[np.ix_(idx, idx)], f0[idx])
                except np.linalg.LinAlgError:
                    logger.warning("Singular Jacobian at iteration %d", iteration)
                    break

                # Trust region: clamp step size
                scale = 1.0
                for i in range(4):
                    if abs(delta[i]) > self._MAX_STEPS[i]:
                        scale = min(scale, self._MAX_STEPS[i] / abs(delta[i]))
                if scale < 1.0:
                    delta *= scale

                # Take step with box constraint projection
                params_new = np.clip(params + delta, self._BOUNDS[:, 0], self._BOUNDS[:, 1])
                f_new, result_new = moog_func(params_new)
                n_calls += 1

                if result_new is not None:
                    last_result = result_new

                new_quadr = np.linalg.norm(f_new[idx])
                logger.info(
                    "Broyden %2d: ||f||=%.4f T=%.0f logg=%.3f feh=%.4f vt=%.4f (%d calls)",
                    iteration, new_quadr, *params_new, n_calls,
                )

                # Broyden rank-1 update
                actual_dx = (params_new - params)[idx]
                actual_df = (f_new - f0)[idx]
                denom = actual_dx @ actual_dx
                if denom > 1e-30:
                    Jff = J[np.ix_(idx, idx)]
                    J[np.ix_(idx, idx)] = Jff + np.outer(
                        actual_df - Jff @ actual_dx, actual_dx
                    ) / denom

                params = params_new
                prev_quadr = quadr
                quadr = new_quadr
                f0 = f_new

                # Convergence check
                if quadr < convergence_tol:
                    logger.info(
                        "Broyden converged in %d iterations (%d MOOG calls)",
                        iteration, n_calls,
                    )
                    return self._broyden_result(params, last_result, True, n_calls, "broyden")

                # Stagnation detection
                if abs(prev_quadr - quadr) < 1e-7:
                    stagnation += 1
                    if stagnation >= 3:
                        logger.warning("Broyden stagnated, resetting Jacobian")
                        for j in idx:
                            p_pert = params.copy()
                            p_pert[j] += self._FD_STEPS[j]
                            p_pert = np.clip(p_pert, self._BOUNDS[:, 0], self._BOUNDS[:, 1])
                            step = p_pert[j] - params[j]
                            f_pert, _ = moog_func(p_pert)
                            n_calls += 1
                            J[:, j] = (f_pert - f0) / step
                        stagnation = 0
                else:
                    stagnation = 0

        logger.warning("Broyden did not converge after %d calls", n_calls)
        return self._broyden_result(params, last_result, False, n_calls, "broyden")

    def _broyden_result(
        self,
        params: np.ndarray,
        result: AbfindResult | None,
        converged: bool,
        n_calls: int,
        method: str,
    ) -> AtmosphericParameters:
        teff, logg, feh, vt = params
        return AtmosphericParameters(
            teff=float(teff),
            logg=float(logg),
            feh=float(feh),
            vt=float(vt),
            n_fe_i=result.n_fe_i if result else 0,
            n_fe_ii=result.n_fe_ii if result else 0,
            fe_i_abundance=result.fe_i_abundance if result else 0.0,
            fe_ii_abundance=result.fe_ii_abundance if result else 0.0,
            converged=converged,
            method=method,
        )

    # ------------------------------------------------------------------
    # Nelder-Mead simplex method
    # ------------------------------------------------------------------

    def _fit_simplex(
        self,
        linelist_path: Path,
        initial: tuple[float, float, float, float],
        hold: list[str],
        boundaries: dict[str, tuple[float, float]] | None,
    ) -> AtmosphericParameters:
        """Nelder-Mead simplex optimization over (T, logg, vt) with [Fe/H] set from MOOG."""
        from scipy.optimize import minimize

        feh0, teff0, logg0, vt0 = initial
        bounds = dict(_ConvergenceState.DEFAULT_BOUNDS)
        if boundaries:
            bounds.update(boundaries)
        feh_lo, feh_hi = bounds["metallicity"]

        # The simplex searches (Teff, log g, vt); [Fe/H] is not a free direction,
        # it is a FIXED POINT of the model at each trial point -- the classical
        # condition that the mean Fe I abundance equal the model metallicity.
        #
        # This method used to read `feh = feh0  # Updated below` and never update
        # it, so every MOOG call ran at the caller's starting metallicity (0.0 by
        # default) and the abundance residual `ab` was computed and then dropped
        # from the objective entirely. Three of the four classical conditions
        # were solved at a metallicity that was usually wrong, and the [Fe/H]
        # finally reported came from a single post-hoc run that was never fed
        # back. For a metal-poor star the error propagates into log g through the
        # ionisation balance, and it does so in the shape -- near zero at solar,
        # growing with |[Fe/H]| -- that is otherwise the signature of real
        # physics.
        def solve_feh(teff, logg, vt, feh_start):
            """Iterate [Fe/H] to its fixed point at fixed (Teff, log g, vt)."""
            feh = float(np.clip(feh_start, feh_lo, feh_hi))
            result = None
            for _ in range(50):
                result = self._run_moog(teff, logg, feh, vt, linelist_path)
                if result is None:
                    return None, feh
                ab = result.fe_i_abundance - self.config.fe_solar
                if abs(ab - feh) <= self.config.tolerance.ab:
                    return result, feh
                feh = float(np.clip(ab, feh_lo, feh_hi))
            return result, feh

        # Carried between objective evaluations so each trial starts from the
        # previous fixed point rather than from feh0; the simplex takes small
        # steps, so this converges in one or two MOOG calls after the first.
        feh_state = {"value": feh0}

        def objective(x: np.ndarray) -> float:
            teff, logg, vt = x
            result, feh = solve_feh(teff, logg, vt, feh_state["value"])
            if result is None:
                return 1e10
            feh_state["value"] = feh

            ep = result.ep_slope
            dif = result.fe_i_fe_ii_diff
            rw = result.rw_slope

            # Weights from the original: 5*((3.5*ep)^2 + (1.3*rw)^2) + 2*(dif)^2.
            # The abundance condition is absent on purpose -- solve_feh has
            # already driven it to zero, so including it would double-count.
            return 5.0 * ((3.5 * ep) ** 2 + (1.3 * rw) ** 2) + 2.0 * dif ** 2

        x0 = np.array([teff0, logg0, vt0])
        res = minimize(objective, x0, method="Nelder-Mead",
                       options={"maxiter": 10000, "xatol": 1.0, "fatol": 1e-6})

        teff, logg, vt = res.x
        # Final run at the converged point, with [Fe/H] again a fixed point.
        result, feh = solve_feh(teff, logg, vt, feh_state["value"])

        return AtmosphericParameters(
            teff=float(teff), logg=float(logg), feh=float(feh), vt=float(vt),
            n_fe_i=result.n_fe_i if result else 0,
            n_fe_ii=result.n_fe_ii if result else 0,
            fe_i_abundance=result.fe_i_abundance if result else 0,
            fe_ii_abundance=result.fe_ii_abundance if result else 0,
            converged=res.success, method="downhill_simplex",
        )

    # ------------------------------------------------------------------
    # MOOG execution
    # ------------------------------------------------------------------

    def _run_moog(
        self,
        teff: float,
        logg: float,
        feh: float,
        vt: float,
        linelist_path: Path,
    ) -> AbfindResult | None:
        """Run a single MOOG abfind iteration.

        Returns None if the atmosphere model can't be interpolated or MOOG fails.
        """
        try:
            with tempfile.TemporaryDirectory(prefix="species_atm_") as tmpdir:
                model_path = Path(tmpdir) / "model.atm"
                ok = self.grid.interpolate_and_write(
                    teff, logg, feh, vt, model_path, fe_solar=self.config.fe_solar,
                )
                if not ok:
                    logger.debug("Grid interpolation failed for T=%.0f logg=%.2f feh=%.2f", teff, logg, feh)
                    return None

                return self.moog.run_abfind(
                    model_path, linelist_path, read_mode=self.config.read_mode,
                )
        except MOOGError as e:
            logger.debug("MOOG error: %s", e)
            return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bisect_parameter(p: AtmosQuantity, moog_val: float, step: float, decimals: int) -> None:
    """Adjust a parameter via bisection based on the MOOG diagnostic value."""
    if moog_val > p.tol:
        p.ranges[0] = p.value
        if p.ranges[1] != -999.0 and p.value < p.ranges[1]:
            p.value = round(np.mean([p.ranges[0], p.ranges[1]]), decimals)
        else:
            p.value = round(_mult(p.value, step, "upper"), decimals)
    else:
        p.ranges[1] = p.value
        if p.ranges[0] != -999.0 and p.value > p.ranges[0]:
            p.value = round(np.mean([p.ranges[0], p.ranges[1]]), decimals)
        else:
            p.value = round(_mult(p.value, step, "floor"), decimals)

    # Clamp to bounds
    p.value = float(np.clip(p.value, p.bounds[0], p.bounds[1]))


def _mult(x: float, base: float, level: str) -> float:
    """Find the nearest multiple of ``base`` above or below ``x``.

    Ported from Atmos.py ``mult()``.
    """
    num = math.floor(x / base)
    if level == "upper":
        return (num + 1) * base
    result = num * base
    if result == x:
        result -= base
    return result
