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

        if method == "downhill_simplex":
            return self._fit_simplex(linelist_path, initial_params, hold, boundaries)
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
        """Iterative per-parameter convergence loop."""
        state = _ConvergenceState(
            initial, hold, self.config.tolerance, boundaries,
        )

        while True:
            state.new_iteration()
            state.check_repeat()

            if state.is_failed():
                break

            # Run MOOG
            feh, teff, logg, vt = state.values
            result = self._run_moog(teff, logg, feh, vt, linelist_path)

            if result is None:
                state.n_break += 1
                state.perturb_on_failure()
                continue

            ab = result.fe_i_abundance - self.config.fe_solar
            state.record_moog_output(ab, result.ep_slope, result.fe_i_fe_ii_diff, result.rw_slope, result.n_failed)

            logger.debug(
                "Iter %d [%s]: feh=%.2f T=%.0f logg=%.2f vt=%.2f → ab=%.3f ep=%.3f dif=%.3f rw=%.3f",
                state.n_it_total, state.change, feh, teff, logg, vt,
                ab, result.ep_slope, result.fe_i_fe_ii_diff, result.rw_slope,
            )

            if state.n_failed > 0:
                state.perturb_on_failure()
                state.n_break += 1
                state.advance_parameter()
                continue

            if state.is_converged():
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

            # Adjust the current parameter
            self._adjust_parameter(state)

        # Failed to converge
        feh, teff, logg, vt = state.values
        logger.warning("Failed to converge after %d iterations", state.n_it_total)
        return AtmosphericParameters(
            teff=teff, logg=logg, feh=feh, vt=vt,
            converged=False, method="per_parameter",
        )

    def _adjust_parameter(self, state: _ConvergenceState) -> None:
        """Adjust one parameter based on current MOOG output and advance."""
        change = state.change
        moog = state.moog  # [ab, ep, dif, rw]

        if change == "metallicity":
            p = state.params["metallicity"]
            # Metallicity is adjusted to match the derived abundance
            ab = moog[0]
            if abs(ab - p.value) > p.tol:
                p.ranges[2] = p.value
                p.value = float(ab)  # Set [Fe/H] = derived Fe abundance
                p.value = np.clip(p.value, p.bounds[0], p.bounds[1])
            state.advance_parameter()

        elif change == "temperature":
            p = state.params["temperature"]
            ep = moog[1]  # EP slope
            _bisect_parameter(p, ep, step=250.0, decimals=0)
            state.advance_parameter()

        elif change == "pressure":
            p = state.params["gravity"]
            dif = moog[2]  # Fe I - Fe II difference
            _bisect_parameter(p, dif, step=0.5, decimals=2)
            state.advance_parameter()

        elif change == "velocity":
            p = state.params["velocity"]
            rw = moog[3]  # RW slope
            _bisect_parameter(p, rw, step=0.5, decimals=2)
            state.advance_parameter()

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

        def objective(x: np.ndarray) -> float:
            teff, logg, vt = x
            feh = feh0  # Updated below

            result = self._run_moog(teff, logg, feh, vt, linelist_path)
            if result is None:
                return 1e10

            ab = result.fe_i_abundance - self.config.fe_solar
            ep = result.ep_slope
            dif = result.fe_i_fe_ii_diff
            rw = result.rw_slope

            # Objective: minimize weighted sum of squared residuals
            # Weights from original: 5*((3.5*ep)^2 + (1.3*rw)^2) + 2*(dif)^2
            return 5.0 * ((3.5 * ep) ** 2 + (1.3 * rw) ** 2) + 2.0 * dif ** 2

        x0 = np.array([teff0, logg0, vt0])
        res = minimize(objective, x0, method="Nelder-Mead",
                       options={"maxiter": 10000, "xatol": 1.0, "fatol": 1e-6})

        teff, logg, vt = res.x
        # Final MOOG run to get the actual parameters
        result = self._run_moog(teff, logg, feh0, vt, linelist_path)
        feh = (result.fe_i_abundance - self.config.fe_solar) if result else feh0

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
