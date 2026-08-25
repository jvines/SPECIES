"""SPECIES — SPECtroscopic Inference of stEllar parameterS.

A modern stellar spectroscopy pipeline for determining atmospheric parameters,
chemical abundances, and broadening from high-resolution spectra.
"""

# Single-sourced from the installed distribution metadata, which comes from
# pyproject. This used to be a literal, and had drifted three ways: pyproject
# said 4.0.4, this said 4.0.0, and tests/test_smoke.py asserted 4.0.0a1 -- so
# the first test in the suite was red and nothing was running it.
from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("astro-species")
except PackageNotFoundError:  # source tree, not installed
    __version__ = "0.0.0.dev0"

# Public API — lazy imports to avoid pulling in heavy deps at import time
def __getattr__(name: str):
    if name == "Spectrum":
        from species.spectrum import Spectrum
        return Spectrum
    if name == "Analyzer":
        from species.analyzer import Analyzer
        return Analyzer
    if name == "Settings":
        from species.config import Settings
        return Settings
    raise AttributeError(f"module 'species' has no attribute {name!r}")

__all__ = ["Spectrum", "Analyzer", "Settings", "__version__"]
