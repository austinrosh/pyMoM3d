"""Inductor characterization plots for EDA-grade parameter visualization.

Provides publication-quality plots of L(f), Q(f), R(f), S-parameters,
and broadband model fitting results.  Follows the LaTeX plotting
conventions defined in ``plot_style.py``.

Typical usage
-------------
>>> from pyMoM3d.visualization import plot_inductor_characterization
>>> from pyMoM3d import configure_latex_style
>>> configure_latex_style()
>>> fig = plot_inductor_characterization(result, wheeler_L=14e-9)
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Label formatters (following plot_style.py convention)
# ---------------------------------------------------------------------------

def format_inductance_label(unit: str = 'nH') -> str:
    r"""Return LaTeX-formatted inductance axis label.

    Parameters
    ----------
    unit : str
        Unit string (default ``'nH'``).

    Returns
    -------
    str
        e.g. ``r'$L$ (nH)'``
    """
    return rf'$L$ ({unit})'


def format_quality_factor_label() -> str:
    r"""Return LaTeX-formatted quality factor axis label."""
    return r'$Q$'


def format_resistance_label() -> str:
    r"""Return LaTeX-formatted resistance axis label."""
    return r'$R$ ($\Omega$)'


# ---------------------------------------------------------------------------
# Unit scaling helpers
# ---------------------------------------------------------------------------

def _auto_l_scale(L_array: np.ndarray) -> tuple:
    """Choose inductance unit and scale factor automatically."""
    L_max = np.max(np.abs(L_array[np.isfinite(L_array)])) if np.any(np.isfinite(L_array)) else 1e-9
    if L_max >= 1e-6:
        return 1e6, r'$\mu$H'
    elif L_max >= 1e-9:
        return 1e9, 'nH'
    else:
        return 1e12, 'pH'


# ---------------------------------------------------------------------------
# Main characterization plot
# ---------------------------------------------------------------------------

def plot_inductor_characterization(
    result,
    wheeler_L: Optional[float] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    r"""Publication-quality 2x2 inductor characterization plot.

    Panels:
    - (0, 0): L(f) from Y-parameter extraction
    - (0, 1): Q(f) with peak annotation and SRF marker
    - (1, 0): R(f) series resistance
    - (1, 1): \|S11\| in dB

    Parameters
    ----------
    result : CharacterizationResult
        Output of ``InductorCharacterization.characterize()``.
    wheeler_L : float, optional
        Analytical Wheeler inductance (H) for reference line.
    title : str, optional
        Figure title.  Auto-generated if None.
    save_path : str, optional
        Save figure to this path.
    show : bool
        Call ``plt.show()`` (default True).

    Returns
    -------
    matplotlib.figure.Figure
    """
    freq_ghz = result.frequencies / 1e9

    # Auto-scale inductance units
    L_scale, L_unit = _auto_l_scale(result.L)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- (0,0): L(f) ---
    ax = axes[0, 0]
    ax.plot(freq_ghz, result.L * L_scale, 'b-o', ms=4, lw=1.5,
            label=r'$L$ (Y-param)')
    if wheeler_L is not None:
        ax.axhline(wheeler_L * L_scale, color='k', ls=':', lw=1.5,
                    label=rf'Wheeler ($L = {wheeler_L * L_scale:.2f}$ {L_unit})')
    if np.isfinite(result.srf):
        ax.axvline(result.srf / 1e9, color='gray', ls='--', alpha=0.5,
                   label=rf'SRF $= {result.srf / 1e9:.2f}$ GHz')
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(format_inductance_label(L_unit))
    ax.set_title(r'Inductance $L(f)$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- (0,1): Q(f) ---
    ax = axes[0, 1]
    ax.plot(freq_ghz, result.Q, 'r-s', ms=4, lw=1.5, label=r'$Q$ (Y-param)')
    if result.Q_peak > 0:
        ax.annotate(
            rf'$Q_{{\mathrm{{peak}}}} = {result.Q_peak:.1f}$',
            xy=(result.f_Q_peak / 1e9, result.Q_peak),
            xytext=(result.f_Q_peak / 1e9 + 0.5, result.Q_peak * 0.8),
            fontsize=8, ha='left',
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.8),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7),
        )
    if np.isfinite(result.srf):
        ax.axvline(result.srf / 1e9, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(format_quality_factor_label())
    ax.set_title(r'Quality Factor $Q(f)$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- (1,0): R(f) ---
    ax = axes[1, 0]
    ax.plot(freq_ghz, result.R, 'g-^', ms=4, lw=1.5)
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(format_resistance_label())
    ax.set_title(r'Series Resistance $R(f)$')
    ax.grid(True, alpha=0.3)

    # --- (1,1): |S11| (dB) ---
    ax = axes[1, 1]
    S11_dB = 20.0 * np.log10(np.abs(result.S11) + 1e-30)
    ax.plot(freq_ghz, S11_dB, 'm-D', ms=4, lw=1.5)
    ax.axhline(-10, color='k', ls='--', alpha=0.3, label=r'$-10$ dB')
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'$|S_{11}|$ (dB)')
    ax.set_title(r'Return Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=13)
    else:
        fig.suptitle(
            rf'Inductor Characterization '
            rf'({result.frequencies[0]/1e9:.1f}$-${result.frequencies[-1]/1e9:.1f} GHz)',
            fontsize=13,
        )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Model fit comparison plot
# ---------------------------------------------------------------------------

def plot_model_fit(
    result,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Compare measured vs fitted Y-parameters from pi-model.

    Two-panel plot showing Re(Y11) and Im(Y11) with measured data
    and model overlay.

    Parameters
    ----------
    result : CharacterizationResult
        Must have ``pi_model`` and ``pi_model_fit_info`` set.
    save_path : str, optional
        Save figure to this path.
    show : bool
        Call ``plt.show()`` (default True).

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If no pi-model has been fitted.
    """
    if result.pi_model is None or result.pi_model_fit_info is None:
        raise ValueError("No pi-model fitted. Call characterize(fit_model=True) first.")

    freq_ghz = result.frequencies / 1e9
    Y_meas = result.Y11
    Y_fit = result.pi_model_fit_info['Y_fitted']

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Re(Y11)
    ax = axes[0]
    ax.plot(freq_ghz, Y_meas.real * 1e3, 'bo', ms=4, label='MoM')
    ax.plot(freq_ghz, Y_fit.real * 1e3, 'r-', lw=1.5, label=r'$\pi$-model')
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'Re$(Y_{11})$ (mS)')
    ax.set_title(r'Real Part of $Y_{11}$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Im(Y11)
    ax = axes[1]
    ax.plot(freq_ghz, Y_meas.imag * 1e3, 'bo', ms=4, label='MoM')
    ax.plot(freq_ghz, Y_fit.imag * 1e3, 'r-', lw=1.5, label=r'$\pi$-model')
    ax.axhline(0, color='k', ls=':', alpha=0.3)
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(r'Im$(Y_{11})$ (mS)')
    ax.set_title(r'Imaginary Part of $Y_{11}$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        rf'Pi-Model Fit '
        rf'($L_s = {result.pi_model.L_s*1e9:.2f}$ nH, '
        rf'$R_s = {result.pi_model.R_s:.2f}$ $\Omega$)',
        fontsize=12,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Z vs Y comparison plot
# ---------------------------------------------------------------------------

def plot_z_vs_y_comparison(
    result,
    wheeler_L: Optional[float] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Side-by-side comparison of Z-based vs Y-based L(f) extraction.

    Demonstrates why Y-parameter extraction gives flat L(f) while
    Z-parameter extraction gives wildly varying results due to
    distributed capacitance.

    Parameters
    ----------
    result : CharacterizationResult
        Output of ``InductorCharacterization.characterize()``.
    wheeler_L : float, optional
        Analytical Wheeler inductance (H) for reference line.
    save_path : str, optional
        Save figure to this path.
    show : bool
        Call ``plt.show()`` (default True).

    Returns
    -------
    matplotlib.figure.Figure
    """
    freq_ghz = result.frequencies / 1e9

    L_scale, L_unit = _auto_l_scale(result.L)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # L from Z
    ax = axes[0]
    ax.plot(freq_ghz, result.L_z * L_scale, 'b-o', ms=4, lw=1.5)
    if wheeler_L is not None:
        ax.axhline(wheeler_L * L_scale, color='k', ls=':', lw=1.5,
                    label=rf'Wheeler ($L = {wheeler_L * L_scale:.2f}$ {L_unit})')
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(format_inductance_label(L_unit))
    ax.set_title(r'$L(f)$ from $Z$-parameter: $L = \mathrm{Im}(Z_{11}) / \omega$')
    if wheeler_L is not None:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # L from Y
    ax = axes[1]
    ax.plot(freq_ghz, result.L * L_scale, 'r-s', ms=4, lw=1.5)
    if wheeler_L is not None:
        ax.axhline(wheeler_L * L_scale, color='k', ls=':', lw=1.5,
                    label=rf'Wheeler ($L = {wheeler_L * L_scale:.2f}$ {L_unit})')
    if np.isfinite(result.srf):
        ax.axvline(result.srf / 1e9, color='gray', ls='--', alpha=0.5,
                   label=rf'SRF $= {result.srf / 1e9:.2f}$ GHz')
    ax.set_xlabel(r'Frequency $f$ (GHz)')
    ax.set_ylabel(format_inductance_label(L_unit))
    ax.set_title(r'$L(f)$ from $Y$-parameter: $L = -1 / (\omega \cdot \mathrm{Im}(Y_{11}))$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Match y-axis scales for fair comparison
    L_y_finite = result.L[np.isfinite(result.L)]
    if len(L_y_finite) > 0:
        L_min = np.min(L_y_finite) * L_scale * 0.5
        L_max = np.max(L_y_finite) * L_scale * 1.5
        if L_max > L_min > 0:
            axes[1].set_ylim(L_min, L_max)

    fig.suptitle(
        r'Z-parameter vs Y-parameter Inductance Extraction',
        fontsize=13,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig
