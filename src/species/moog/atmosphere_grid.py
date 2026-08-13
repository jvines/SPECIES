"""ATLAS9 atmosphere model grid interpolation.

Ported from AtmosGrid.py and AtmosInterpol.py. Provides lazy-loaded singleton
access to the pre-computed grid, and 3D interpolation to generate MOOG-format
atmosphere model files for arbitrary (Teff, logg, [Fe/H]).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import ClassVar

import numpy as np
from scipy.interpolate import griddata, LinearNDInterpolator

from species.moog.grid_io import read_grid_netcdf, write_grid_netcdf

logger = logging.getLogger(__name__)

# Number of layers and columns in ATLAS9 Kurucz models
_N_LAYERS = 72
_N_COLUMNS = 9

# MOOG molecular equilibrium species
_MOOG_MOLECULES = (
    "      606.0    106.0    607.0    608.0    107.0    108.0    112.0    707.0\n"
    "      708.0    808.0     12.1  60808.0  10108.0    101.0      6.1      7.1\n"
    "        8.1    822.0     22.1\n"
)


class AtmosphereGrid:
    """Lazy-loaded ATLAS9 atmosphere model grid.

    The grid is loaded from a NetCDF file on first access. If it does not
    exist, it is built from the raw ATLAS9 grid files.

    Usage::

        grid = AtmosphereGrid.get(atlas9_dir=Path("./atm_models/atlas9"))
        model_path = grid.interpolate_and_write(
            teff=5777, logg=4.44, feh=0.0, micro=1.0,
            output_path=Path("/tmp/model.atm"),
        )
    """

    _instance: ClassVar[AtmosphereGrid | None] = None
    _instance_dir: ClassVar[Path | None] = None

    def __init__(self, atlas9_dir: Path | str) -> None:
        self.atlas9_dir = Path(atlas9_dir)
        self._grid: dict | None = None

    @classmethod
    def get(cls, atlas9_dir: Path) -> AtmosphereGrid:
        """Return a singleton instance for the given grid directory."""
        if cls._instance is None or cls._instance_dir != atlas9_dir:
            cls._instance = cls(atlas9_dir)
            cls._instance_dir = atlas9_dir
        return cls._instance

    @property
    def grid(self) -> dict:
        """Lazy-load the grid on first access."""
        if self._grid is None:
            self._load_grid()
        return self._grid

    def interpolate_and_write(
        self,
        teff: float,
        logg: float,
        feh: float,
        micro: float,
        output_path: Path,
        fe_solar: float = 7.50,
    ) -> bool:
        """Interpolate the grid and write a MOOG-format ``.atm`` file.

        Parameters
        ----------
        teff, logg, feh
            Stellar parameters to interpolate to.
        micro
            Microturbulence velocity (km/s).
        output_path
            Where to write the ``.atm`` file.
        fe_solar
            Solar iron abundance.

        Returns
        -------
        bool
            ``True`` if interpolation succeeded, ``False`` if the point
            is outside the grid or interpolation failed.
        """
        g = self.grid
        tgrid = g["tgrid"]
        ggrid = g["ggrid"]
        mgrid = g["mgrid"]

        # Select nearby grid points (within 301K, 0.501 dex)
        mask = (
            (np.abs(tgrid - teff) <= 301.0)
            & (np.abs(mgrid - feh) <= 0.501)
            & (np.abs(ggrid - logg) <= 0.501)
        )

        if np.sum(mask) < 2:
            return False

        # Verify the point is within the convex hull of nearby points
        t_sel, g_sel, m_sel = tgrid[mask], ggrid[mask], mgrid[mask]
        dt = t_sel.max() - t_sel.min()
        dm = m_sel.max() - m_sel.min()
        dg = g_sel.max() - g_sel.min()

        inside = (
            t_sel.min() <= teff <= t_sel.max()
            and m_sel.min() <= feh <= m_sel.max()
            and g_sel.min() <= logg <= g_sel.max()
            and dt > 0
            and dm > 0
            and dg > 0
            and 0.0 <= micro <= 5.0
        )

        if not inside:
            return False

        # 3D linear interpolation for all 7 columns at once
        # Use cached interpolator when the same grid neighborhood is reused
        # (common during iterative convergence where parameters change slightly)
        points = np.column_stack([t_sel, g_sel, m_sel])
        xi = np.array([[teff, logg, feh]])

        columns: list[np.ndarray] = []
        for col_key in ("col0", "col1", "col2", "col3", "col4", "col5", "col6"):
            interp = self._get_interpolator(
                tuple(map(tuple, points)), col_key, g[col_key][mask], points,
            )
            col = interp(xi).ravel()
            if not np.all(np.isfinite(col)):
                return False
            columns.append(col)

        n_layers = len(columns[0])

        # Write MOOG-format Kurucz model
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("KURUCZ\n")
            f.write(f"          Teff = {teff:5.0f} log g = {logg:4.2f}\n")
            f.write(f"NTAU        {n_layers:3.0f}\n")
            for i in range(n_layers):
                f.write(
                    f" {columns[0][i]:14.8E}  {columns[1][i]:7.1f}"
                    f" {columns[2][i]:9.3E} {columns[3][i]:9.3E}"
                    f" {columns[4][i]:9.3E} {columns[5][i]:9.3E}"
                    f" {columns[6][i]:9.3E}\n"
                )
            f.write(f"    {micro:5.3f}e+05\n")
            f.write(f"NATOMS     1  {feh:5.2f}\n")
            f.write(f"      26.0   {feh + fe_solar:4.2f}\n")
            f.write("NMOL      19\n")
            f.write(_MOOG_MOLECULES)

        return True

    # Manual cache for interpolators (keyed by grid neighborhood hash + column)
    _interp_cache: ClassVar[dict[tuple, LinearNDInterpolator]] = {}
    _interp_cache_maxsize: ClassVar[int] = 56  # 7 columns × 8 neighborhoods

    def _get_interpolator(
        self,
        points_key: tuple,
        col_key: str,
        values: np.ndarray,
        points: np.ndarray,
    ) -> LinearNDInterpolator:
        """Get or create a cached LinearNDInterpolator for a grid neighborhood.

        The cache key is a hash of the grid vertices + column name. When
        consecutive iterations select the same nearby grid points (common
        during convergence), the triangulation is reused.
        """
        # Use hash of points_key for speed (tuple of tuples is hashable)
        cache_key = (hash(points_key), col_key)
        if cache_key in self._interp_cache:
            return self._interp_cache[cache_key]

        # Evict oldest if cache is full
        if len(self._interp_cache) >= self._interp_cache_maxsize:
            oldest = next(iter(self._interp_cache))
            del self._interp_cache[oldest]

        interp = LinearNDInterpolator(points, values, rescale=True)
        self._interp_cache[cache_key] = interp
        return interp

    # ------------------------------------------------------------------
    # Grid loading
    # ------------------------------------------------------------------

    GRID_FILENAME = "ATLAS9_grid.nc"

    def _load_grid(self) -> None:
        """Load the grid from NetCDF.

        Search order:
        1. ``atlas9_dir / ATLAS9_grid.nc`` (user-provided or env var)
        2. Bundled package data (ships with pip install)
        3. Build from the raw ATLAS9 grid files (legacy, needs the 3.3 GB grids)

        The grid used to ship as a pickle and no longer does; a pickle is not
        read at all any more. An unversioned format that executes arbitrary code
        on load has no business being the distribution channel for a 34 MB data
        product. The NetCDF file holds identical numbers in 13 MB, because
        slightly over half the cube is NaN -- combinations such as Teff = 15000 K
        at log g = 0 have no ATLAS9 model -- and those blocks compress away.
        """
        grid_path = self.atlas9_dir / self.GRID_FILENAME

        if not grid_path.exists():
            try:
                from importlib.resources import files
                bundled = Path(str(files("species").joinpath("data", self.GRID_FILENAME)))
                if bundled.exists():
                    grid_path = bundled
                    logger.info("Using bundled ATLAS9 grid")
            except Exception:
                pass

        if not grid_path.exists():
            grid_path = self.atlas9_dir / self.GRID_FILENAME
            logger.info("Building ATLAS9 grid from raw models (one-time operation)...")
            self._create_grid(grid_path)

        self._grid = read_grid_netcdf(grid_path)
        n_models = int(np.isfinite(self._grid["col0"][:, 0]).sum())
        logger.info(
            "Loaded ATLAS9 grid: %d nodes, %d with models, %d empty",
            len(self._grid["tgrid"]), n_models, len(self._grid["tgrid"]) - n_models,
        )

    def _create_grid(self, grid_path: Path) -> None:
        """Build the grid from raw ATLAS9 model files and write it as NetCDF.

        Ported from AtmosGrid.create_grid_file().
        """
        teff_values = np.array([
            3500, 3750, 4000, 4250, 4500, 4750, 5000, 5250, 5500, 5750,
            6000, 6250, 6500, 6750, 7000, 7250, 7500, 7750, 8000, 8250,
            8500, 8750, 9000, 9250, 9500, 9750, 10000, 10250, 10500, 10750,
            11000, 11250, 11500, 11750, 12000, 12250, 12500, 12750, 13000,
            14000, 15000,
        ], dtype=float)

        logg_values = np.linspace(0.0, 5.0, 11)

        feh_values = np.array([
            -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5,
            -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 1.0,
        ])

        # Build flat coordinate arrays
        tgrid, ggrid, mgrid = [], [], []
        for t in teff_values:
            for g in logg_values:
                for m in feh_values:
                    tgrid.append(t)
                    ggrid.append(g)
                    mgrid.append(m)

        # Read atmosphere models
        cols = {f"col{i}": [] for i in range(7)}

        for t, g, feh in zip(tgrid, ggrid, mgrid):
            sign = "m" if feh < 0 else "p"
            avar = str(int(abs(feh * 10)))
            ametal = f"{sign}0{avar}" if abs(feh * 10) <= 5 else f"{sign}{avar}"

            grid_dir = self.atlas9_dir / "grids" / f"grid{ametal}"
            odfnew_dir = grid_dir.parent / f"grid{ametal}odfnew"

            path_file = odfnew_dir if odfnew_dir.is_dir() else grid_dir

            # Split concatenated model files if individual files don't exist
            if path_file.is_dir() and not list(path_file.glob(f"{ametal}_*.dat")):
                for concat_file in path_file.glob(f"a{ametal}*.dat"):
                    _split_model_file(concat_file, ametal, path_file)
                    break

            model_file = path_file / f"{ametal}_{int(t)}_{g:.1f}.dat"
            tab = _read_model_file(model_file, _N_LAYERS, _N_COLUMNS)

            for i in range(7):
                cols[f"col{i}"].append(tab.T[i])

        grid_data = {
            "tgrid": np.array(tgrid),
            "ggrid": np.array(ggrid),
            "mgrid": np.array(mgrid),
        }
        for key, values in cols.items():
            grid_data[key] = np.array(values)

        write_grid_netcdf(grid_data, grid_path)
        logger.info("Created ATLAS9 grid at %s", grid_path)


# ---------------------------------------------------------------------------
# File I/O helpers (ported from AtmosGrid.py)
# ---------------------------------------------------------------------------

def _split_model_file(file_path: Path, ametal: str, output_dir: Path) -> None:
    """Split a concatenated ATLAS9 grid file into individual models."""
    current_file = None
    with open(file_path) as f:
        for line in f:
            stripped = line.strip()
            m1 = re.search(r"TEFF\s+(\d+\.)\s+GRAVITY\s+(\d+\.\d+)\s+LTE", stripped)
            m2 = re.search(r"EFF\s+(\d+\.)\s+GRAVITY\s+(\d+\.\d+)\s+LTE", stripped)
            match = m1 or m2
            if match:
                if current_file is not None:
                    current_file.close()
                teff = float(match.group(1))
                logg = float(match.group(2))
                out_name = f"{ametal}_{int(teff)}_{logg}.dat"
                current_file = open(output_dir / out_name, "w")
            if current_file is not None:
                current_file.write(line)

    if current_file is not None:
        current_file.close()


def _read_model_file(file_path: Path, n_layers: int, n_cols: int) -> np.ndarray:
    """Read a single ATLAS9 model atmosphere file into a 2D array."""
    table = np.full((n_layers, n_cols), np.nan)
    if not file_path.exists():
        return table

    i = 0
    with open(file_path) as f:
        for line in f:
            if re.match(r"\A[0-9]\.[0-9]*", line):
                cols = line.split()
                for j in range(min(n_cols, len(cols))):
                    table[i][j] = float(cols[j])
                i += 1
                if i >= n_layers:
                    break

    return table
