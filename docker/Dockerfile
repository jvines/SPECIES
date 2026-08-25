FROM python:3.12-slim-bookworm

# gfortran is for MOOG (a Fortran binary the pipeline shells out to); libnetcdf
# backs the atmosphere grid now that it is NetCDF rather than a pickle; git is
# needed by setuptools-scm and by Documenter-style tooling.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gfortran build-essential git ca-certificates curl \
        libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*

# The bind-mounted checkout is owned by the host user, not root, so git refuses
# to operate on it ("dubious ownership") and every git call exits 128. Same trap
# as the sibling Julia project.
RUN git config --global --add safe.directory /work \
    && git config --global --add safe.directory '*'

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /work

# Dependencies are installed from pyproject at build time so the image is not
# rebuilt on every source edit; the package itself is installed editable at
# runtime against the bind mount.
COPY pyproject.toml README.md /work/
RUN python -m pip install --upgrade pip \
    && python -m pip install \
        "numpy>=1.26" "scipy>=1.12" "odrpack>=0.5" "astropy>=6.0" \
        "matplotlib>=3.8" "pydantic>=2.5" "pydantic-settings>=2.1" \
        "uncertainties>=3.1" "PyAstronomy>=0.21" \
        "astroquery>=0.4.7" "click>=8.1" \
        "pytest>=8.0" "pytest-cov" "ruff" \
        "netCDF4>=1.6"

CMD ["bash"]
