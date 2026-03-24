"""Configuration management for SPECIES.

Replaces the 80+ global variables from the original SPECIES.py with a typed,
validated Pydantic Settings model. All paths are configurable via environment
variables or constructor arguments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Science dataclasses (ported from Atmos.py)
# ---------------------------------------------------------------------------

@dataclass
class Tolerance:
    """Convergence tolerances for the iterative atmospheric parameter fitting.

    ab:  |derived [Fe/H] - input [Fe/H]|
    ep:  |slope of abundance vs excitation potential|
    dif: |mean Fe I - mean Fe II|
    rw:  |slope of abundance vs reduced equivalent width|
    """

    ab: float = 0.001
    ep: float = 0.001
    dif: float = 0.001
    rw: float = 0.001


@dataclass
class AtmosQuantity:
    """A single atmospheric parameter being fitted."""

    name: str
    value: float
    hold: bool = False
    ranges: list[float] = field(default_factory=list)
    bounds: tuple[float, float] = (0.0, 0.0)
    width: float = 0.0
    tol: float = 0.001
    change: float = 0.0


# ---------------------------------------------------------------------------
# Package data paths
# ---------------------------------------------------------------------------

def _pkg_data() -> Path:
    """Return the path to the bundled ``data/`` directory."""
    return Path(str(files("species").joinpath("data")))


# ---------------------------------------------------------------------------
# Main settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """SPECIES configuration.

    Values are loaded from environment variables prefixed with ``SPECIES_``
    (e.g. ``SPECIES_MOOG_BINARY``), or passed directly to the constructor.
    """

    model_config = {"env_prefix": "SPECIES_", "extra": "ignore"}

    # -- Paths ---------------------------------------------------------------
    atlas9_dir: Path = Field(
        default=Path("./atm_models/atlas9"),
        description="Directory containing ATLAS9 model atmosphere grids.",
    )
    moog_binary: str = Field(
        default="MOOGSILENT",
        description="Name or full path of the MOOGSILENT binary.",
    )
    moog_data_dir: Path = Field(
        default=Path("./MOOGFEB2017"),
        description="Directory containing MOOG support files (Barklem, etc.).",
    )
    output_dir: Path = Field(
        default=Path("./output"),
        description="Base directory for analysis output.",
    )

    # Bundled data — resolved from package if not overridden
    linelist_path: Path | None = Field(default=None)
    linelist_ab_path: Path | None = Field(default=None)
    linelist_vsini_path: Path | None = Field(default=None)
    mask_path: Path | None = Field(default=None)

    # -- Science defaults ----------------------------------------------------
    fe_solar: float = Field(default=7.50, description="Solar iron abundance (Asplund 2009).")
    tolerance: Tolerance = Field(default_factory=Tolerance)
    minimization: Literal["broyden", "per_parameter", "downhill_simplex"] = "broyden"
    read_mode: Literal["linearregression", "odr", "moog"] = "linearregression"

    # -- Pipeline toggles ----------------------------------------------------
    compute_errors: bool = True
    compute_abundances: bool = True
    compute_broadening: bool = True
    compute_rest_frame: bool = True
    is_giant: bool = False

    # -- MOOG tuning ---------------------------------------------------------
    moog_timeout: int = Field(default=300, description="MOOG subprocess timeout in seconds.")

    # -- Parallelization -----------------------------------------------------
    n_cores: int = 1

    @model_validator(mode="after")
    def _resolve_package_data(self) -> Settings:
        """Fill in bundled data paths for any that were not explicitly set."""
        pkg = _pkg_data()
        defaults = {
            "linelist_path": pkg / "linelists" / "linelist.dat",
            "linelist_ab_path": pkg / "linelists" / "lines_ab.dat",
            "linelist_vsini_path": pkg / "linelists" / "linelist_vsini.dat",
            "mask_path": pkg / "masks" / "G2.mas",
        }
        for attr, default_path in defaults.items():
            if getattr(self, attr) is None:
                object.__setattr__(self, attr, default_path)
        return self
