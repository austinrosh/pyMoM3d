"""Centralized plot style configuration for pyMoM3d.

This module provides consistent, publication-quality LaTeX-rendered text
for all matplotlib plots. It configures fonts, enables math rendering,
and provides helper functions for consistent scientific notation.

Usage
-----
Import and call `configure_latex_style()` at the start of any plotting script:

    from pyMoM3d.visualization.plot_style import configure_latex_style
    configure_latex_style()

Or use the context manager for temporary style changes:

    from pyMoM3d.visualization.plot_style import latex_style
    with latex_style():
        plt.plot(...)

Notation Conventions
--------------------
- Vectors: boldface (e.g., $\\mathbf{J}$, $\\mathbf{E}$)
- Scalars: italic (e.g., $f$, $k$, $\\lambda$)
- Units: roman font (e.g., $\\mathrm{A/m}$, $\\mathrm{Hz}$)
- Subscripts: roman for labels (e.g., $R_{\\mathrm{in}}$), italic for indices
"""

import warnings
from contextlib import contextmanager
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib as mpl

# Track whether LaTeX style has been configured
_latex_configured = False
_original_params = {}


def _check_latex_available() -> bool:
    """Check if a LaTeX installation is available for usetex mode.

    Returns
    -------
    available : bool
        True if LaTeX is available, False otherwise.
    """
    import shutil
    return shutil.which('latex') is not None


def configure_latex_style(
    use_tex: Optional[bool] = None,
    font_size: int = 11,
    font_family: str = 'serif',
    figure_dpi: int = 150,
    save_dpi: int = 300,
) -> bool:
    """Configure matplotlib for LaTeX-rendered text.

    This function sets up matplotlib to use LaTeX-style math rendering
    for publication-quality plots. If a full LaTeX installation is not
    available, it falls back to matplotlib's built-in mathtext renderer.

    Parameters
    ----------
    use_tex : bool, optional
        If True, use full LaTeX rendering (requires LaTeX installation).
        If False, use matplotlib's mathtext (no external dependencies).
        If None (default), auto-detect LaTeX availability.
    font_size : int, default 11
        Base font size for all text elements.
    font_family : str, default 'serif'
        Font family: 'serif', 'sans-serif', or 'monospace'.
    figure_dpi : int, default 150
        DPI for figure display.
    save_dpi : int, default 300
        DPI for saved figures.

    Returns
    -------
    using_tex : bool
        True if full LaTeX rendering is enabled, False if using mathtext.

    Examples
    --------
    >>> from pyMoM3d.visualization.plot_style import configure_latex_style
    >>> using_tex = configure_latex_style()
    >>> print(f"Using full LaTeX: {using_tex}")
    """
    global _latex_configured, _original_params

    # Store original params for restoration
    if not _latex_configured:
        _original_params = {
            'text.usetex': mpl.rcParams['text.usetex'],
            'font.family': mpl.rcParams['font.family'],
            'font.size': mpl.rcParams['font.size'],
            'figure.dpi': mpl.rcParams['figure.dpi'],
            'savefig.dpi': mpl.rcParams['savefig.dpi'],
            'axes.labelsize': mpl.rcParams['axes.labelsize'],
            'axes.titlesize': mpl.rcParams['axes.titlesize'],
            'legend.fontsize': mpl.rcParams['legend.fontsize'],
            'xtick.labelsize': mpl.rcParams['xtick.labelsize'],
            'ytick.labelsize': mpl.rcParams['ytick.labelsize'],
        }

    # Determine whether to use full LaTeX
    if use_tex is None:
        use_tex = _check_latex_available()

    if use_tex and not _check_latex_available():
        warnings.warn(
            "LaTeX installation not found. Falling back to matplotlib mathtext. "
            "Install LaTeX for full rendering support.",
            RuntimeWarning,
            stacklevel=2,
        )
        use_tex = False

    # Configure matplotlib
    plt.rcParams.update({
        # LaTeX rendering
        'text.usetex': use_tex,

        # Font configuration
        'font.family': font_family,
        'font.size': font_size,

        # Math text (fallback when usetex=False)
        'mathtext.fontset': 'cm',  # Computer Modern (LaTeX-like)
        'mathtext.rm': 'serif',

        # Figure settings
        'figure.dpi': figure_dpi,
        'savefig.dpi': save_dpi,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,

        # Axes settings
        'axes.labelsize': font_size,
        'axes.titlesize': font_size + 2,
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'axes.grid.which': 'major',

        # Tick settings
        'xtick.labelsize': font_size - 1,
        'ytick.labelsize': font_size - 1,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4,
        'ytick.major.size': 4,

        # Legend settings
        'legend.fontsize': font_size - 1,
        'legend.framealpha': 0.9,
        'legend.edgecolor': 'gray',

        # Grid settings
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,

        # Line settings
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
    })

    # Additional LaTeX-specific settings
    if use_tex:
        plt.rcParams.update({
            'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}',
        })

    _latex_configured = True
    return use_tex


def restore_default_style():
    """Restore matplotlib to default settings.

    Reverts any changes made by `configure_latex_style()`.
    """
    global _latex_configured, _original_params

    if _original_params:
        for key, value in _original_params.items():
            try:
                plt.rcParams[key] = value
            except (KeyError, ValueError):
                pass
        _original_params = {}

    _latex_configured = False


@contextmanager
def latex_style(**kwargs):
    """Context manager for temporary LaTeX style configuration.

    Parameters
    ----------
    **kwargs
        Arguments passed to `configure_latex_style()`.

    Examples
    --------
    >>> with latex_style(font_size=12):
    ...     plt.plot([1, 2, 3])
    ...     plt.xlabel(r'$x$ (m)')
    ...     plt.show()
    """
    # Save current state
    saved_params = {k: mpl.rcParams[k] for k in mpl.rcParams}

    try:
        configure_latex_style(**kwargs)
        yield
    finally:
        # Restore previous state
        for key in list(mpl.rcParams.keys()):
            if key in saved_params:
                try:
                    mpl.rcParams[key] = saved_params[key]
                except (KeyError, ValueError):
                    pass


# =============================================================================
# Standard Label Formatters
# =============================================================================

def format_frequency_label(with_unit: bool = True) -> str:
    """Return LaTeX-formatted frequency axis label.

    Parameters
    ----------
    with_unit : bool, default True
        Include unit in parentheses.

    Returns
    -------
    label : str
        LaTeX-formatted label string.
    """
    if with_unit:
        return r'Frequency $f$ (GHz)'
    return r'Frequency $f$'


def format_angle_label(angle_type: str = 'theta', unit: str = 'deg') -> str:
    """Return LaTeX-formatted angle axis label.

    Parameters
    ----------
    angle_type : str, default 'theta'
        Angle type: 'theta', 'phi', or 'psi'.
    unit : str, default 'deg'
        Unit: 'deg' or 'rad'.

    Returns
    -------
    label : str
        LaTeX-formatted label string.
    """
    symbols = {'theta': r'\theta', 'phi': r'\phi', 'psi': r'\psi'}
    units = {'deg': r'deg', 'rad': r'rad'}

    sym = symbols.get(angle_type, angle_type)
    u = units.get(unit, unit)

    return rf'${sym}$ ({u})'


def format_rcs_label(unit: str = 'dBsm') -> str:
    """Return LaTeX-formatted RCS axis label.

    Parameters
    ----------
    unit : str, default 'dBsm'
        RCS unit: 'dBsm', 'm2', or 'normalized'.

    Returns
    -------
    label : str
        LaTeX-formatted label string.
    """
    if unit == 'dBsm':
        return r'RCS $\sigma$ (dBsm)'
    elif unit == 'm2':
        return r'RCS $\sigma$ (m$^2$)'
    else:
        return r'Normalized RCS $\sigma/\pi a^2$'


def format_impedance_label(component: str = 'real', unit: str = 'ohm') -> str:
    """Return LaTeX-formatted impedance axis label.

    Parameters
    ----------
    component : str, default 'real'
        Component: 'real', 'imag', or 'mag'.
    unit : str, default 'ohm'
        Always ohms, but controls symbol style.

    Returns
    -------
    label : str
        LaTeX-formatted label string.
    """
    if component == 'real':
        return r'$R_{\mathrm{in}}$ ($\Omega$)'
    elif component == 'imag':
        return r'$X_{\mathrm{in}}$ ($\Omega$)'
    else:
        return r'$|Z_{\mathrm{in}}|$ ($\Omega$)'


def format_current_label(log_scale: bool = False) -> str:
    """Return LaTeX-formatted surface current density label.

    Parameters
    ----------
    log_scale : bool, default False
        If True, format for dB scale.

    Returns
    -------
    label : str
        LaTeX-formatted label string.
    """
    if log_scale:
        return r'$|\mathbf{J}|$ (dB A/m)'
    return r'$|\mathbf{J}|$ (A/m)'


def format_directivity_label(unit: str = 'dBi') -> str:
    """Return LaTeX-formatted directivity axis label.

    Parameters
    ----------
    unit : str, default 'dBi'
        Unit: 'dBi', 'dBd', or 'linear'.

    Returns
    -------
    label : str
        LaTeX-formatted label string.
    """
    if unit == 'dBi':
        return r'Directivity $D$ (dBi)'
    elif unit == 'dBd':
        return r'Directivity $D$ (dBd)'
    else:
        return r'Directivity $D$'


def format_coordinate_label(axis: str, unit: str = 'm') -> str:
    """Return LaTeX-formatted coordinate axis label.

    Parameters
    ----------
    axis : str
        Axis name: 'x', 'y', or 'z'.
    unit : str, default 'm'
        Unit string.

    Returns
    -------
    label : str
        LaTeX-formatted label string.
    """
    return rf'${axis}$ ({unit})'


# =============================================================================
# Title Formatters
# =============================================================================

def format_rcs_title(
    geometry: str,
    frequency_ghz: Optional[float] = None,
    ka: Optional[float] = None,
) -> str:
    """Return LaTeX-formatted title for RCS plots.

    Parameters
    ----------
    geometry : str
        Geometry description (e.g., 'PEC Sphere').
    frequency_ghz : float, optional
        Frequency in GHz.
    ka : float, optional
        Electrical size parameter.

    Returns
    -------
    title : str
        LaTeX-formatted title string.
    """
    parts = [f'{geometry} RCS']
    if frequency_ghz is not None:
        parts.append(rf'$f = {frequency_ghz:.2f}$ GHz')
    if ka is not None:
        parts.append(rf'$ka = {ka:.2f}$')

    return ', '.join(parts)


def format_current_title(
    geometry: str,
    frequency_ghz: Optional[float] = None,
    num_basis: Optional[int] = None,
) -> str:
    """Return LaTeX-formatted title for surface current plots.

    Parameters
    ----------
    geometry : str
        Geometry description (e.g., 'PEC Sphere').
    frequency_ghz : float, optional
        Frequency in GHz.
    num_basis : int, optional
        Number of RWG basis functions.

    Returns
    -------
    title : str
        LaTeX-formatted title string.
    """
    parts = [rf'Surface Current $|\mathbf{{J}}|$ on {geometry}']
    if frequency_ghz is not None:
        parts.append(rf'$f = {frequency_ghz:.2f}$ GHz')
    if num_basis is not None:
        parts.append(rf'$N = {num_basis}$')

    return '\n'.join([parts[0], ', '.join(parts[1:])] if len(parts) > 1 else parts)


# =============================================================================
# Annotation Helpers
# =============================================================================

def format_resonance_annotation(
    frequency_ghz: float,
    resistance_ohm: float,
) -> str:
    """Return LaTeX-formatted annotation for resonance point.

    Parameters
    ----------
    frequency_ghz : float
        Resonant frequency in GHz.
    resistance_ohm : float
        Input resistance at resonance in Ohms.

    Returns
    -------
    annotation : str
        LaTeX-formatted annotation string.
    """
    return (
        rf'$f_{{\mathrm{{res}}}} = {frequency_ghz:.3f}$ GHz'
        '\n'
        rf'$R_{{\mathrm{{in}}}} = {resistance_ohm:.1f}$ $\Omega$'
    )


def format_plane_wave_annotation(
    E0_direction: str,
    k_direction: str,
) -> str:
    """Return LaTeX-formatted annotation for plane wave excitation.

    Parameters
    ----------
    E0_direction : str
        E-field polarization direction (e.g., 'x').
    k_direction : str
        Propagation direction (e.g., '-z').

    Returns
    -------
    annotation : str
        LaTeX-formatted annotation string.
    """
    return (
        rf'$\mathbf{{E}}_0 \parallel \hat{{{E0_direction}}}$, '
        rf'$\hat{{\mathbf{{k}}}} = {k_direction}$'
    )
