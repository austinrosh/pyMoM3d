"""Verify feedline calibration with synthetic (analytical) S-parameters.

This proves the de-embedding math is correct by testing with known data:
a through-line of length L_total = L_dut + 2*L_ext, with known Z0 and eps_eff.

Generate: S_raw = cascade(S_feedline1, S_dut, S_feedline2)
De-embed: S_cal = deembed(S_raw) → should recover S_dut exactly.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from pyMoM3d.utils.constants import c0
from pyMoM3d.network.feedline_calibration import (
    FeedlineCalibration, _tl_abcd, _abcd_to_s, _s_to_abcd,
)

Z0_line = 50.0
eps_eff = 3.33
L_dut = 10e-3
L_ext = 15e-3

print("=== Synthetic feedline calibration test ===\n")

cal = FeedlineCalibration(Z0_line, eps_eff, L_ext, Z0_ref=50.0)

for freq in [1e9, 2e9, 3e9, 4e9]:
    gamma = cal.gamma(freq)
    beta = gamma.imag

    # DUT ABCD (just a through line of length L_dut)
    T_dut = _tl_abcd(Z0_line, gamma, L_dut)
    S_dut = _abcd_to_s(T_dut, 50.0)

    # Full structure ABCD
    T_ext1 = _tl_abcd(Z0_line, gamma, L_ext)
    T_ext2 = _tl_abcd(Z0_line, gamma, L_ext)
    T_raw = T_ext1 @ T_dut @ T_ext2
    S_raw = _abcd_to_s(T_raw, 50.0)

    # De-embed
    S_cal = cal.deembed(S_raw, freq)

    # Compare
    s11_dut = 20 * np.log10(max(abs(S_dut[0, 0]), 1e-30))
    s21_dut = 20 * np.log10(max(abs(S_dut[1, 0]), 1e-15))
    s11_raw = 20 * np.log10(max(abs(S_raw[0, 0]), 1e-30))
    s21_raw = 20 * np.log10(max(abs(S_raw[1, 0]), 1e-15))
    s11_cal = 20 * np.log10(max(abs(S_cal[0, 0]), 1e-30))
    s21_cal = 20 * np.log10(max(abs(S_cal[1, 0]), 1e-15))

    err_s11 = abs(S_cal[0, 0] - S_dut[0, 0])
    err_s21 = abs(S_cal[1, 0] - S_dut[1, 0])

    print(f"f = {freq/1e9:.0f} GHz (beta*L = {beta*L_dut*180/np.pi:.0f} deg):")
    print(f"  S_dut:  S11={s11_dut:>8.2f} dB, S21={s21_dut:>8.2f} dB")
    print(f"  S_raw:  S11={s11_raw:>8.2f} dB, S21={s21_raw:>8.2f} dB")
    print(f"  S_cal:  S11={s11_cal:>8.2f} dB, S21={s21_cal:>8.2f} dB")
    print(f"  Error:  |dS11|={err_s11:.2e}, |dS21|={err_s21:.2e}")
    print()

print("If S_cal matches S_dut exactly → de-embedding math is correct.")

# --- Now test with MoM-like data (S21 very small) ---
print("\n=== Simulating MoM-like weak S21 ===\n")

for freq in [1e9, 2e9, 3e9]:
    gamma = cal.gamma(freq)

    # MoM-like raw S-matrix: S11 ≈ -j (capacitive), S21 tiny
    # Simulate a port with large capacitive mismatch
    Z_in = -1j * 200  # purely capacitive port
    S11_mom = (Z_in - 50) / (Z_in + 50)  # S11 for capacitive load
    S21_mom = 1e-3 * np.exp(-1j * gamma.imag * (L_dut + 2*L_ext))  # tiny coupling

    S_raw_mom = np.array([
        [S11_mom, S21_mom],
        [S21_mom, S11_mom],  # reciprocal, symmetric
    ], dtype=np.complex128)

    S_cal_mom = cal.deembed(S_raw_mom, freq)

    print(f"f = {freq/1e9:.0f} GHz:")
    print(f"  S_raw: S11={20*np.log10(abs(S11_mom)):.1f} dB, S21={20*np.log10(abs(S21_mom)):.1f} dB")
    print(f"  S_cal: S11={20*np.log10(max(abs(S_cal_mom[0,0]),1e-30)):.1f} dB, "
          f"S21={20*np.log10(max(abs(S_cal_mom[1,0]),1e-15)):.1f} dB")

    # The ABCD from tiny S21 is almost singular
    T_raw_mom = _s_to_abcd(S_raw_mom, 50.0)
    print(f"  ABCD raw: |A|={abs(T_raw_mom[0,0]):.1f}, |D|={abs(T_raw_mom[1,1]):.1f}")
    print()
