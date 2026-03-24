# ── Stage 0: base ──────────────────────────────────────────────────────
# System deps + MOOG compilation. Rebuilds only when MOOG source changes.
FROM python:3.12-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    gfortran \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Compile MOOG from the bundled source
COPY MOOGFEB2017/ /opt/moog/
WORKDIR /opt/moog
# Build MOOGSILENT if Makefile or source exists
RUN if [ -f Makefile ]; then make -f Makefile 2>/dev/null || true; fi
# If there's a pre-compiled binary, use it; otherwise check the build
RUN ls -la /opt/moog/MOOGSILENT 2>/dev/null || echo "MOOG binary not found — will need to be provided"
ENV PATH="/opt/moog:${PATH}"

# ── Stage 1: python deps ──────────────────────────────────────────────
FROM base AS deps

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY pyproject.toml /app/
RUN pip install --no-cache-dir -e ".[test]" 2>/dev/null || \
    pip install --no-cache-dir numpy scipy astropy matplotlib \
    pydantic pydantic-settings uncertainties PyAstronomy pytest

# ── Stage 2: app ──────────────────────────────────────────────────────
FROM deps AS app

# Copy the package source
COPY src/ /app/src/
COPY tests/ /app/tests/

# Copy ATLAS9 grids (mounted as volume in production, copied for testing)
# These are expected at /app/atm_models/atlas9/ or via SPECIES_ATLAS9_DIR env
COPY atm_models/ /app/atm_models/

# Copy sample spectra for testing
COPY Spectra/ /app/Spectra/
COPY EW/ /app/EW/
COPY binary_masks/ /app/binary_masks/
COPY MOOG_linelist/ /app/MOOG_linelist/

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

# Set environment for SPECIES
ENV SPECIES_ATLAS9_DIR=/app/atm_models/atlas9
ENV SPECIES_MOOG_BINARY=MOOGSILENT
ENV SPECIES_MOOG_DATA_DIR=/opt/moog

WORKDIR /app

CMD ["python", "-c", "from species import Spectrum, Analyzer, Settings; print('SPECIES v4 ready')"]
