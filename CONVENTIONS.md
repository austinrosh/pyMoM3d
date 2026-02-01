# Conventions

## Time Convention
- `exp(-j*omega*t)` 

## RWG Basis Function Orientation
- **t_plus**: triangle where current flows AWAY from the free vertex (divergence positive: `+l_n / (2*A_plus)`)
- **t_minus**: triangle where current flows TOWARD the free vertex (divergence negative: `-l_n / (2*A_minus)`)
- Basis function vector: `f_n(r) = (l_n / (2*A_plus)) * rho_plus` on T+, `(l_n / (2*A_minus)) * rho_minus` on T-
- `rho_plus = r - r_free_plus` (points away from free vertex on T+)
- `rho_minus = r_free_minus - r` (points toward free vertex on T-)

## Green's Function
- Free-space scalar: `g(r,r') = exp(-jkR) / (4*pi*R)`, where `R = |r - r'|`
- Phase sign consistent with `exp(-j*omega*t)` time convention

## Far-Field
- Far-field uses `exp(+jk*r_hat.r')` — opposite sign from Green's function

## Units
- Length: meters
- Frequency: Hz
- Electric field: V/m
- Magnetic field: A/m
- Impedance: Ohms

## Numerical Types
- Coordinates / physical quantities: `np.float64`
- Index arrays: `np.int32`
- Complex fields / impedance: `np.complex128`
