"""Solar chemical abundances from Asplund et al. (2009).

Ported from Atmos.py lines 1118-1139.
"""

from __future__ import annotations

# Atomic numbers for elements used in abundance analysis
ELEMENTS: dict[str, float] = {
    "C": 6.0,
    "N": 7.0,
    "O": 8.0,
    "Na": 11.0,
    "Mg": 12.0,
    "Al": 13.0,
    "Si": 14.0,
    "S": 16.0,
    "Ca": 20.0,
    "Sc": 21.0,
    "Ti": 22.0,
    "V": 23.0,
    "Cr": 24.0,
    "Mn": 25.0,
    "Fe": 26.0,
    "Co": 27.0,
    "Ni": 28.0,
    "Cu": 29.0,
    "Zn": 30.0,
    "Y": 39.0,
    "Ba": 56.0,
}

# Solar abundances on the A(X) = log(N_X/N_H) + 12 scale
# Source: Asplund, Grevesse, Sauval & Scott (2009), ARA&A 47, 481
SOLAR_ABUNDANCE: dict[str, float] = {
    "C": 8.43,
    "N": 7.83,
    "O": 8.69,
    "Na": 6.24,
    "Mg": 7.60,
    "Al": 6.45,
    "Si": 7.51,
    "S": 7.12,
    "Ca": 6.34,
    "Sc": 3.15,
    "Ti": 4.95,
    "V": 3.93,
    "Cr": 5.64,
    "Mn": 5.43,
    "Fe": 7.50,
    "Co": 4.99,
    "Ni": 6.22,
    "Cu": 4.19,
    "Zn": 4.56,
    "Y": 2.21,
    "Ba": 2.18,
}
