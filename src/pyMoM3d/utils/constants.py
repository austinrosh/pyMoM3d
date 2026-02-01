"""Physical constants for electromagnetic computations.

All values in SI units.
"""

import numpy as np

#: Speed of light in vacuum (m/s)
c0: float = 299792458.0

#: Permeability of free space (H/m)
mu0: float = 4.0e-7 * np.pi

#: Permittivity of free space (F/m)
eps0: float = 1.0 / (mu0 * c0**2)

#: Intrinsic impedance of free space (Ohms)
eta0: float = mu0 * c0
