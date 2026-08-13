"""NetCDF storage for the ATLAS9 atmosphere grid.

Replaces the pickle the grid used to ship as. Pickle is a bad archive format for
scientific data: it is unversioned, it executes arbitrary code on load, it is
readable only from Python (and only from a compatible Python), and it stores a
flattened point list that throws away the fact that the grid is rectilinear.

The NetCDF form keeps the structure the data actually has --
``(teff, logg, feh, layer, column)`` -- which is self-describing, readable from
any language, and compresses the holes. Slightly over half the grid is NaN,
because combinations like Teff = 15000 K at log g = 0 do not exist, and the
pickle stored every one of those at full width.

The on-disk layout is deliberately *not* the in-memory layout: the loader
reconstructs the flat ``tgrid``/``ggrid``/``mgrid``/``col0..col6`` dictionary
the interpolator has always consumed, in exactly the original point order, so
nothing downstream changes and results stay bit-identical.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Column meanings in the Kurucz/ATLAS9 model tables, in file order. Recorded in
# the NetCDF attributes so the file explains itself; the pickle never did.
COLUMN_NAMES = (
    "rhox",      # mass column density        [g cm^-2]
    "temp",      # temperature                [K]
    "pgas",      # gas pressure               [dyn cm^-2]
    "xne",       # electron number density    [cm^-3]
    "abross",    # Rosseland mean absorption  [cm^2 g^-1]
    "accrad",    # radiative acceleration     [cm s^-2]
    "vturb",     # turbulent velocity         [cm s^-1]
)
N_COLUMNS = len(COLUMN_NAMES)
GRID_FORMAT_VERSION = 1


def write_grid_netcdf(grid: dict, path: Path | str, *, complevel: int = 4) -> Path:
    """Write the flat grid dictionary to NetCDF, keeping its rectilinear shape.

    `grid` is the historical in-memory form: `tgrid`, `ggrid`, `mgrid` (each of
    length N) and `col0`..`col6` (each N x n_layers). The axes are recovered by
    uniquing the coordinate arrays, and the point order is asserted to be
    C-order over (teff, logg, feh) rather than assumed -- if a future grid is
    built in a different order this raises instead of silently transposing the
    atmosphere models.
    """
    from netCDF4 import Dataset

    path = Path(path)
    t, g, m = (np.asarray(grid[k], dtype=float) for k in ("tgrid", "ggrid", "mgrid"))
    teff, logg, feh = (np.unique(x) for x in (t, g, m))
    n_layers = np.asarray(grid["col0"]).shape[1]

    expected = np.meshgrid(teff, logg, feh, indexing="ij")
    if not all(np.array_equal(a, b.ravel()) for a, b in zip((t, g, m), expected)):
        raise ValueError(
            "grid points are not in C-order over (teff, logg, feh); refusing to "
            "write, because the reshape below would silently scramble models"
        )

    shape = (teff.size, logg.size, feh.size, n_layers, N_COLUMNS)
    cube = np.empty(shape, dtype="f8")
    for i in range(N_COLUMNS):
        cube[..., i] = np.asarray(grid[f"col{i}"], dtype=float).reshape(
            teff.size, logg.size, feh.size, n_layers
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.title = "ATLAS9 (Kurucz) model atmosphere grid"
        ds.source = "SPECIES; converted from ATLAS9_grid.pickle"
        ds.grid_format_version = GRID_FORMAT_VERSION
        ds.column_names = ",".join(COLUMN_NAMES)
        ds.comment = (
            "NaN marks a (teff, logg, feh) combination with no ATLAS9 model. "
            "Roughly half the cube is NaN by construction."
        )

        ds.createDimension("teff", teff.size)
        ds.createDimension("logg", logg.size)
        ds.createDimension("feh", feh.size)
        ds.createDimension("layer", n_layers)
        ds.createDimension("column", N_COLUMNS)

        for name, values, unit in (
            ("teff", teff, "K"), ("logg", logg, "dex(cm s-2)"), ("feh", feh, "dex"),
        ):
            v = ds.createVariable(name, "f8", (name,))
            v[:] = values
            v.units = unit
        lay = ds.createVariable("layer", "i4", ("layer",))
        lay[:] = np.arange(n_layers, dtype="i4")
        lay.long_name = "atmosphere layer index, top down"

        # Chunked along the model axes so the all-NaN slabs compress as blocks.
        var = ds.createVariable(
            "atmosphere", "f8", ("teff", "logg", "feh", "layer", "column"),
            zlib=True, complevel=complevel, shuffle=True,
            chunksizes=(1, logg.size, feh.size, n_layers, N_COLUMNS),
            fill_value=np.nan,
        )
        var[:] = cube
        var.column_names = ",".join(COLUMN_NAMES)

    logger.info("wrote ATLAS9 grid to %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return path


def read_grid_netcdf(path: Path | str) -> dict:
    """Read the NetCDF grid back into the flat dictionary the interpolator uses.

    Returns the same keys, dtypes, shapes and point order as the pickle did.
    """
    from netCDF4 import Dataset

    with Dataset(Path(path), "r") as ds:
        version = getattr(ds, "grid_format_version", 0)
        if version > GRID_FORMAT_VERSION:
            raise ValueError(
                f"{path} is grid format v{version}, this SPECIES understands "
                f"v{GRID_FORMAT_VERSION}; upgrade astro-species"
            )
        teff = np.asarray(ds.variables["teff"][:], dtype=float)
        logg = np.asarray(ds.variables["logg"][:], dtype=float)
        feh = np.asarray(ds.variables["feh"][:], dtype=float)
        var = ds.variables["atmosphere"]
        # NetCDF hands back a masked array; the interpolator has always seen
        # plain NaN for missing models, so keep that contract.
        cube = np.ma.filled(var[:].astype("f8"), np.nan)

    tt, gg, mm = np.meshgrid(teff, logg, feh, indexing="ij")
    n_layers = cube.shape[3]
    out: dict = {"tgrid": tt.ravel(), "ggrid": gg.ravel(), "mgrid": mm.ravel()}
    flat = cube.reshape(teff.size * logg.size * feh.size, n_layers, cube.shape[4])
    for i in range(cube.shape[4]):
        out[f"col{i}"] = np.ascontiguousarray(flat[:, :, i])
    return out
