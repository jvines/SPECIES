"""The ATLAS9 grid is a data product; its storage format is a contract.

These tests exist because the grid used to ship as a pickle, and a pickle has no
version, no schema and no way to be read from anything but a compatible Python.
Swapping the format is only safe if the numbers survive exactly -- the grid
feeds MOOG, and a silently transposed or truncated model atmosphere would move
every abundance without failing anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from species.moog.grid_io import (
    COLUMN_NAMES,
    GRID_FORMAT_VERSION,
    read_grid_netcdf,
    write_grid_netcdf,
)


def _toy_grid(n_teff=3, n_logg=2, n_feh=4, n_layers=5, n_cols=7):
    """A small grid with the same structure as the real one, holes included."""
    teff = np.linspace(4000, 6000, n_teff)
    logg = np.linspace(1.0, 4.5, n_logg)
    feh = np.linspace(-2.0, 0.5, n_feh)
    tt, gg, mm = np.meshgrid(teff, logg, feh, indexing="ij")
    n = tt.size
    rng = np.random.default_rng(0)
    grid = {"tgrid": tt.ravel(), "ggrid": gg.ravel(), "mgrid": mm.ravel()}
    for i in range(n_cols):
        col = rng.normal(size=(n, n_layers)) * 10 ** i
        col[::3] = np.nan          # holes, as in the real grid
        grid[f"col{i}"] = col
    return grid


def test_round_trip_is_exact(tmp_path):
    grid = _toy_grid()
    path = write_grid_netcdf(grid, tmp_path / "g.nc")
    back = read_grid_netcdf(path)

    assert sorted(back) == sorted(grid)
    for key in grid:
        assert np.array_equal(
            np.asarray(grid[key]), np.asarray(back[key]), equal_nan=True
        ), f"{key} changed across the round trip"


def test_point_order_is_preserved(tmp_path):
    """The interpolator indexes col arrays by position in tgrid/ggrid/mgrid."""
    grid = _toy_grid()
    back = read_grid_netcdf(write_grid_netcdf(grid, tmp_path / "g.nc"))
    for i in (0, 1, 7, 13):
        assert back["tgrid"][i] == grid["tgrid"][i]
        assert back["ggrid"][i] == grid["ggrid"][i]
        assert back["mgrid"][i] == grid["mgrid"][i]
        assert np.array_equal(back["col3"][i], grid["col3"][i], equal_nan=True)


def test_holes_survive_as_nan(tmp_path):
    """Missing models must stay NaN, not become a fill value or a zero."""
    grid = _toy_grid()
    back = read_grid_netcdf(write_grid_netcdf(grid, tmp_path / "g.nc"))
    assert np.isnan(back["col0"][0]).all()
    assert np.isnan(np.asarray(back["col0"])).sum() == np.isnan(np.asarray(grid["col0"])).sum()


def test_scrambled_point_order_is_refused(tmp_path):
    """A grid not in C-order would be silently transposed by the reshape."""
    grid = _toy_grid()
    order = np.arange(grid["tgrid"].size)[::-1]
    scrambled = {k: (v[order] if v.ndim >= 1 else v) for k, v in grid.items()}
    with pytest.raises(ValueError, match="C-order"):
        write_grid_netcdf(scrambled, tmp_path / "bad.nc")


def test_file_is_self_describing(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    path = write_grid_netcdf(_toy_grid(), tmp_path / "g.nc")
    with netCDF4.Dataset(path) as ds:
        assert ds.grid_format_version == GRID_FORMAT_VERSION
        assert ds.column_names.split(",") == list(COLUMN_NAMES)
        assert set(ds.dimensions) == {"teff", "logg", "feh", "layer", "column"}
        assert ds.variables["teff"].units == "K"


def test_future_format_version_is_refused(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    path = write_grid_netcdf(_toy_grid(), tmp_path / "g.nc")
    with netCDF4.Dataset(path, "a") as ds:
        ds.grid_format_version = GRID_FORMAT_VERSION + 1
    with pytest.raises(ValueError, match="grid format"):
        read_grid_netcdf(path)


@pytest.mark.slow
def test_bundled_grid_loads_and_has_the_expected_shape():
    """The real bundled grid: 41 Teff x 11 logg x 19 [Fe/H], 72 layers."""
    from importlib.resources import files
    from pathlib import Path

    path = Path(str(files("species").joinpath("data", "ATLAS9_grid.nc")))
    if not path.exists():
        pytest.skip("bundled grid not present")
    grid = read_grid_netcdf(path)
    assert grid["tgrid"].size == 41 * 11 * 19
    assert grid["col0"].shape == (41 * 11 * 19, 72)
    assert len(np.unique(grid["tgrid"])) == 41
    # Slightly over half the cube has no model; that is expected, not a fault.
    empty = np.isnan(grid["col0"][:, 0]).sum()
    assert 0.4 < empty / grid["tgrid"].size < 0.6
