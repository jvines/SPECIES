"""Korg + MARCS as a second atmospheric-parameter engine.

SPECIES's primary engine is MOOG + ATLAS9. This one solves the same four
classical excitation/ionisation conditions with Korg.jl and MARCS atmospheres,
behind the same interface as :class:`species.atmosphere.AtmosphereFitter`, and
consuming the *same* equivalent widths — the MOOG-format line list SPECIES has
already written. Any difference between the two engines is therefore radiative
transfer and atmosphere grid, not measurement.

It also returns something the MOOG engine does not: the full 4x4 parameter
covariance. Korg computes the Jacobian of the residual system at the solution
and discards everything but four magnitudes; KLOTHO recovers the joint. The
four classical parameters are strongly correlated — rho(Teff, [Fe/H]) = 0.87 on
the Sun — and downstream isochrone and SED fitting routinely consumes them as
independent.

**Julia is optional.** Without it SPECIES behaves exactly as it always has;
this engine reports itself unavailable and the caller carries on with MOOG.

Why a subprocess rather than juliacall: juliacall embeds a Julia runtime in the
CPython process and holds the GIL for the duration of each call, which does not
compose with a workload that is one independent star per core. MOOG is already
driven as a subprocess, so this is the established shape for this codebase.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from species.atmosphere import AtmosphericParameters

logger = logging.getLogger(__name__)

_DRIVER = Path(__file__).with_name("solve_ews.jl")

# SPECIES calls them this; Korg's parameter vector is [Teff, logg, vmic, m/H].
_HOLD_NAMES = ("temperature", "gravity", "metallicity", "velocity")


class KorgUnavailable(RuntimeError):
    """Raised when the Julia side cannot be reached."""


@dataclass
class KorgResult:
    """What the Korg engine returns, beyond the shared parameter object.

    ``covariance`` is the reason this engine exists. ``retcode`` distinguishes a
    measurement from a fit that merely stopped: ``railed`` means a parameter
    parked on a MARCS grid boundary, which converges and means nothing.
    """

    params: AtmosphericParameters
    retcode: str = "unknown"
    railed: list[str] = field(default_factory=list)
    covariance: dict | None = None
    n_lines: int = 0
    raw: dict = field(default_factory=dict)

    @property
    def is_measurement(self) -> bool:
        return self.retcode == "ok"

    def correlation(self, a: str, b: str) -> float:
        """Correlation between two named parameters, e.g. ``("teff", "feh")``."""
        if not self.covariance:
            raise ValueError("no covariance available")
        order = self.covariance["order"]
        C = self.covariance["correlation"]
        return float(C[order.index(a)][order.index(b)])


class KorgFitter:
    """Drop-in alternative to :class:`AtmosphereFitter`, backed by Korg.jl.

    Parameters
    ----------
    config
        SPECIES settings. Reads ``korg_project`` and ``julia_binary``.
    timeout
        Seconds to allow one star. Korg's first call in a process pays JIT
        compilation, so this is generous by default.
    """

    def __init__(self, config=None, timeout: float = 3600.0) -> None:
        self.config = config
        self.timeout = timeout
        self.last_result: KorgResult | None = None

    # -- availability -------------------------------------------------------

    @property
    def julia_binary(self) -> str:
        return (
            getattr(self.config, "julia_binary", None)
            or os.environ.get("SPECIES_JULIA_BINARY")
            or "julia"
        )

    @property
    def project(self) -> Path | None:
        p = (
            getattr(self.config, "korg_project", None)
            or os.environ.get("SPECIES_KORG_PROJECT")
        )
        return Path(p) if p else None

    def available(self) -> tuple[bool, str]:
        """Whether this engine can run, and why not if it cannot."""
        if shutil.which(self.julia_binary) is None:
            return False, f"julia not found on PATH (looked for {self.julia_binary!r})"
        if self.project is None:
            return False, (
                "no Korg project configured; set SPECIES_KORG_PROJECT to a Julia "
                "project with KLOTHO installed"
            )
        if not (self.project / "Project.toml").exists():
            return False, f"{self.project} is not a Julia project (no Project.toml)"
        if not _DRIVER.exists():
            return False, f"driver script missing at {_DRIVER}"
        return True, "ok"

    # -- the interface AtmosphereFitter presents ----------------------------

    def fit(
        self,
        linelist_path: Path,
        initial_params: tuple[float, float, float, float] | None = None,
        hold: list[str] | None = None,
        boundaries: dict[str, tuple[float, float]] | None = None,
        method: str | None = None,
    ) -> AtmosphericParameters:
        """Solve for atmospheric parameters. Signature matches AtmosphereFitter.

        ``initial_params`` is ``(feh, teff, logg, vt)``, as elsewhere in SPECIES
        — note this is *not* Korg's ordering, which the driver converts.

        ``boundaries`` and ``method`` are accepted for interface compatibility
        and ignored: Korg clamps to the MARCS grid ranges, and the solve is
        always clipped Newton-Raphson. Passing them logs a warning rather than
        failing silently, because silently ignoring a constraint is how a held
        parameter ends up free.
        """
        ok, why = self.available()
        if not ok:
            raise KorgUnavailable(why)

        if boundaries:
            logger.warning(
                "KorgFitter ignores `boundaries`; Korg clamps to the MARCS grid "
                "ranges. Requested: %s", boundaries,
            )
        if method not in (None, "korg"):
            logger.warning("KorgFitter ignores `method=%r`", method)

        feh0, teff0, logg0, vt0 = initial_params or (0.0, 5500.0, 4.36, 1.23)
        hold = list(hold or [])
        for name in hold:
            if name not in _HOLD_NAMES:
                raise ValueError(f"unknown hold parameter {name!r}; expected {_HOLD_NAMES}")

        with tempfile.TemporaryDirectory(prefix="species_korg_") as tmp:
            out = Path(tmp) / "korg_result.json"
            cmd = [
                self.julia_binary, f"--project={self.project}", str(_DRIVER),
                str(linelist_path), str(out),
                f"teff0={teff0}", f"logg0={logg0}", f"feh0={feh0}", f"vt0={vt0}",
            ]
            if hold:
                cmd.append("hold=" + ",".join(hold))

            logger.info("Korg engine: %s", " ".join(cmd[:3]))
            try:
                # check=False: a non-zero exit still leaves a JSON with a
                # retcode, which is more useful to the caller than a
                # CalledProcessError carrying only a return code.
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=self.timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as e:
                raise KorgUnavailable(f"Korg solve timed out after {self.timeout}s") from e

            if not out.exists():
                raise KorgUnavailable(
                    "Korg driver produced no output"
                    + (f": {proc.stderr[-400:]}" if proc.stderr else "")
                )
            payload = json.loads(out.read_text())

        self.last_result = _to_result(payload)
        if payload.get("status") == "error":
            logger.error("Korg engine failed: %s", payload.get("error"))
        return self.last_result.params


def _to_result(payload: dict) -> KorgResult:
    if payload.get("status") != "ok":
        return KorgResult(params=AtmosphericParameters(), retcode="solver_error",
                          raw=payload)
    params = AtmosphericParameters(
        teff=float(payload["teff"]),
        logg=float(payload["logg"]),
        feh=float(payload["feh"]),
        vt=float(payload["vt"]),
        n_fe_i=int(payload.get("n_fe_i", 0)),
        n_fe_ii=int(payload.get("n_fe_ii", 0)),
        # `converged` here means Korg met its residual tolerances AND nothing
        # parked on a grid boundary. A railed fit converges and is meaningless,
        # so it must not be reported as converged to the rest of the pipeline.
        converged=payload.get("retcode") == "ok",
        method="korg",
    )
    return KorgResult(
        params=params,
        retcode=str(payload.get("retcode", "unknown")),
        railed=list(payload.get("railed", [])),
        covariance=payload.get("covariance"),
        n_lines=int(payload.get("n_lines", 0)),
        raw=payload,
    )
