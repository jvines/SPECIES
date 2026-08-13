"""The parallel batch path, which was broken and untested.

`Analyzer.batch(n_cores>1)` submitted a function nested inside `batch` to a
ProcessPoolExecutor. Nested functions cannot be pickled, so every submit raised,
every star was recorded as a failure, and the CLI wrote a full set of
zero-filled result files. Nothing caught it because nothing ran it.

These tests deliberately avoid MOOG: they check the plumbing that was broken --
that the worker is picklable, and that a Spectrum survives the process boundary
with the fields the old tuple dropped.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from species.analyzer import _run_one
from species.spectrum import Spectrum


def _toy_spectrum(name="star", rv=12.5):
    wave = np.linspace(6000.0, 6100.0, 256)
    flux = np.ones_like(wave)
    return Spectrum(
        wavelength=wave, flux=flux, snr=120.0, instrument="HARPS",
        star_name=name, rv=rv, header={"OBJECT": name},
    )


def test_worker_is_picklable():
    """The actual defect: a nested worker cannot cross a process boundary."""
    assert pickle.loads(pickle.dumps(_run_one)) is _run_one


def test_spectrum_survives_pickling_with_all_fields():
    """The tuple the old code sent carried five fields and dropped three."""
    spec = _toy_spectrum()
    back = pickle.loads(pickle.dumps(spec))

    assert np.array_equal(back.wavelength, spec.wavelength)
    assert np.array_equal(back.flux, spec.flux)
    assert back.instrument == spec.instrument
    assert back.star_name == spec.star_name
    # These three were lost by the tuple. rv especially: a worker rebuilding
    # from arrays alone would re-run the CCF on an already-corrected spectrum
    # and report rv = 0.
    assert back.rv == spec.rv
    assert back.header == spec.header
    assert back.ccf_result == spec.ccf_result


def test_task_tuple_is_picklable(tmp_path):
    """The whole payload, as `batch` builds it."""
    from species.config import Settings

    config_dict = {
        k: str(v) if hasattr(v, "__fspath__") else v
        for k, v in Settings().model_dump().items()
        if v is not None
    }
    task = (_toy_spectrum(), str(tmp_path / "star"), config_dict, {})
    spec, _star_dir, cfg, _kwargs = pickle.loads(pickle.dumps(task))
    assert spec.rv == 12.5
    assert Settings(**cfg) is not None


def test_resolving_power_from_instrument():
    """vsini is meaningless without R, so the lookup has to work."""
    assert _toy_spectrum().resolving_power == pytest.approx(115_000.0)
    unknown = Spectrum(
        wavelength=np.linspace(6000, 6100, 8), flux=np.ones(8), snr=10.0,
        instrument="NOT-A-SPECTROGRAPH",
    )
    assert unknown.resolving_power == 0.0
    explicit = Spectrum(
        wavelength=np.linspace(6000, 6100, 8), flux=np.ones(8), snr=10.0,
        instrument="NOT-A-SPECTROGRAPH", resolution=77_000.0,
    )
    assert explicit.resolving_power == pytest.approx(77_000.0)
