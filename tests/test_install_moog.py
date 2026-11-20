"""install-moog must compile the install directory into MOOG (SPEC-5)."""
import shutil
import tempfile
from pathlib import Path

import pytest

from species.cli import _patch_moog_path

SRC = """      write (moogpath,1001)
      moogpath = 
     .  '/moog17scat'
"""


def _tree(d):
    for n in ("Moog.f", "Moogsilent.f"):
        (d / n).write_text(SRC)
    return d


def test_patch_points_at_install_dir():
    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        _check(Path(d))


def _check(tmp_path):
    _tree(tmp_path)
    assert _patch_moog_path(tmp_path) == ["Moog.f", "Moogsilent.f"]
    for n in ("Moog.f", "Moogsilent.f"):
        t = (tmp_path / n).read_text()
        assert "/moog17scat'" not in t
        assert f"'{tmp_path.resolve()}'" in t


def test_patch_rejects_long_prefix(tmp_path):
    long = tmp_path / ("x" * 60)
    long.mkdir()
    _tree(long)
    with pytest.raises(ValueError):
        _patch_moog_path(long)


def test_patch_fails_loudly_when_source_changed(tmp_path):
    (tmp_path / "Moog.f").write_text("nothing here")
    (tmp_path / "Moogsilent.f").write_text("nothing here")
    with pytest.raises(ValueError):
        _patch_moog_path(tmp_path)


@pytest.mark.skipif(not (shutil.which("gfortran") and shutil.which("git")), reason="needs gfortran+git")
def test_built_binary_has_no_moog17scat_path():
    """Network: clone, patch, compile, check the embedded path."""
    from species.cli import main
    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        prefix = Path(d) / "moog"
        try:
            main(["install-moog", "--prefix", str(prefix)])
        except SystemExit:
            pass
        if not (prefix / "MOOGSILENT").exists():
            pytest.skip("could not clone/build (offline?)")
        data = (prefix / "MOOGSILENT").read_bytes()
        assert str(prefix.resolve()).encode() in data
        assert b"/moog17scat" not in data
