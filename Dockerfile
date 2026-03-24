# ── Stage 0: base ──────────────────────────────────────────────────────
# System deps + MOOG compilation from moog17scat (headless, no X11/SM).
FROM python:3.12-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    gfortran \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone and compile moog17scat (headless MOOG variant, no X11/SM deps)
ARG MOOG_COMMIT=53d3f645f18c1acfb568c5ed7ea0c4e4551ef5e5
RUN git clone https://github.com/jvines/moog17scat.git /opt/moog17scat \
    && cd /opt/moog17scat \
    && git checkout $MOOG_COMMIT \
    && make \
    && ls -la MOOGSILENT
ENV PATH="/opt/moog17scat:${PATH}"
# MOOG hardcodes /moog17scat/ for Barklem data files
RUN ln -sf /opt/moog17scat /moog17scat

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

# Copy ATLAS9 grids
COPY atm_models/ /app/atm_models/

# Copy sample spectra and data for testing
COPY Spectra/ /app/Spectra/
COPY EW/ /app/EW/
COPY binary_masks/ /app/binary_masks/
COPY MOOG_linelist/ /app/MOOG_linelist/
# Copy MOOG support files (Barklem.dat etc) from the bundled MOOGFEB2017
COPY MOOGFEB2017/Barklem.dat MOOGFEB2017/BarklemUV.dat /app/MOOGFEB2017/
# Copy abfind.par template
COPY MOOGFEB2017/abfind.par /opt/moog17scat/abfind.par

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

# Set environment for SPECIES
ENV SPECIES_ATLAS9_DIR=/app/atm_models/atlas9
ENV SPECIES_MOOG_BINARY=MOOGSILENT
ENV SPECIES_MOOG_DATA_DIR=/opt/moog17scat

WORKDIR /app

CMD ["python", "-c", "from species import Spectrum, Analyzer, Settings; print('SPECIES v4 ready')"]
