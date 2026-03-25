"""SPECIES — SPECtroscopic Inference of stEllar parameterS.

A modern stellar spectroscopy pipeline for determining atmospheric parameters,
chemical abundances, and broadening from high-resolution spectra.
"""

__version__ = "4.0.0"

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
