"""
Example: Using the high-level Simulation driver.

Demonstrates the simplified API:
- SimulationConfig for setup
- Simulation.run() for single frequency
- Simulation.sweep() for frequency sweep
- SimulationResult for accessing outputs
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt

from pyMoM3d import (
    Sphere,
    Simulation,
    SimulationConfig,
    PlaneWaveExcitation,
    compute_far_field,
    compute_rcs,
    eta0,
    c0,
)


def main():
    print("=" * 60)
    print("Simulation Driver Demo")
    print("=" * 60)

    # --- Setup ---
    radius = 0.1
    frequency = 1e9

    exc = PlaneWaveExcitation(
        E0=np.array([1.0, 0.0, 0.0]),
        k_hat=np.array([0.0, 0.0, -1.0]),
    )

    config = SimulationConfig(
        frequency=frequency,
        excitation=exc,
        solver_type='direct',
        quad_order=4,
    )

    # --- Create simulation from geometry ---
    print("\n--- Building simulation from Sphere geometry ---")
    sim = Simulation(config, geometry=Sphere(radius=radius), subdivisions=2)

    print(f"Mesh: {sim.mesh.get_num_triangles()} triangles, "
          f"{sim.basis.num_basis} basis functions")

    # --- Single frequency solve ---
    print("\n--- Single frequency solve ---")
    result = sim.run()

    print(f"Frequency: {result.frequency/1e9:.2f} GHz")
    print(f"Condition number: {result.condition_number:.2e}")
    print(f"Max |I|: {np.max(np.abs(result.I_coefficients)):.4e}")

    # --- Compute RCS from result ---
    k = 2 * np.pi * frequency / c0
    theta = np.linspace(0.01, np.pi - 0.01, 91)
    phi = np.zeros_like(theta)
    E_theta, E_phi = compute_far_field(
        result.I_coefficients, sim.basis, sim.mesh, k, eta0, theta, phi
    )
    rcs = compute_rcs(E_theta, E_phi)
    print(f"Backscatter RCS: {rcs[-1]:.2f} dBsm")

    # --- Frequency sweep ---
    print("\n--- Frequency sweep ---")
    frequencies = [0.8e9, 1.0e9, 1.2e9]
    results = sim.sweep(frequencies)

    for r in results:
        k_f = 2 * np.pi * r.frequency / c0
        E_th, E_ph = compute_far_field(
            r.I_coefficients, sim.basis, sim.mesh, k_f, eta0,
            np.array([np.pi]), np.array([0.0])
        )
        rcs_back = compute_rcs(E_th, E_ph)
        print(f"  f={r.frequency/1e9:.1f} GHz: backscatter RCS = {rcs_back[0]:.2f} dBsm")

    # --- Save and reload ---
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        result.save(f.name)
        from pyMoM3d.simulation import SimulationResult
        loaded = SimulationResult.load(f.name)
        print(f"\nSaved and reloaded result: {loaded.frequency/1e9:.2f} GHz, "
              f"{len(loaded.I_coefficients)} coefficients")
        os.unlink(f.name)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    theta_deg = np.degrees(theta)
    ax.plot(theta_deg, rcs, 'b-', linewidth=1.5)
    ax.set_xlabel('Theta (deg)')
    ax.set_ylabel('RCS (dBsm)')
    ax.set_title(f'Sphere RCS via Simulation driver, f={frequency/1e9:.1f} GHz')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 180])

    plt.tight_layout()

    images_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
    os.makedirs(images_dir, exist_ok=True)
    output_file = os.path.join(images_dir, 'simulation_driver_demo.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {output_file}")

    plt.show()

    print("\n" + "=" * 60)
    print("Simulation driver demo complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
