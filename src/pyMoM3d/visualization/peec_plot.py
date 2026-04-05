"""PEEC segment current visualization.

Visualizes current flow through PEEC trace segments as colored arrows
along the segment centerlines.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors


def plot_peec_currents(
    segments,
    I_segments: np.ndarray,
    ax: Optional[plt.Axes] = None,
    cmap: str = 'viridis',
    normalize: bool = False,
    title: Optional[str] = None,
    show_width: bool = True,
    alpha: float = 0.8,
    clim: Optional[tuple] = None,
) -> tuple:
    """Plot PEEC segment currents as colored arrows along trace centerlines.

    Parameters
    ----------
    segments : list of TraceSegment
        PEEC trace segments from ``TraceNetwork.all_segments``.
    I_segments : ndarray, shape (M,), complex
        Branch currents from MNA solve (one per segment).
        Typically ``result.I_solutions[:, port_idx]``.
    ax : matplotlib Axes, optional
        2D axes for plotting.  Created if not provided.
    cmap : str
        Colormap name for current magnitude.
    normalize : bool
        If True, normalize arrow lengths to unit length.
    title : str, optional
        Plot title.
    show_width : bool
        If True, draw trace outlines showing physical width.
    alpha : float
        Arrow transparency.
    clim : tuple of (vmin, vmax), optional
        Color limits for current magnitude.

    Returns
    -------
    fig : Figure
    ax : Axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    else:
        fig = ax.figure

    I_mag = np.abs(I_segments)
    vmin = clim[0] if clim else 0.0
    vmax = clim[1] if clim else (np.max(I_mag) if np.max(I_mag) > 0 else 1.0)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.get_cmap(cmap)

    # Draw trace outlines if requested
    if show_width:
        for seg in segments:
            s = seg.start[:2] * 1e3  # mm
            e = seg.end[:2] * 1e3
            d = e - s
            length = np.linalg.norm(d)
            if length < 1e-12:
                continue
            u = d / length
            n = np.array([-u[1], u[0]])  # normal
            hw = seg.width * 1e3 / 2.0

            corners = np.array([
                s - n * hw, s + n * hw,
                e + n * hw, e - n * hw,
                s - n * hw,  # close
            ])
            ax.plot(corners[:, 0], corners[:, 1],
                    color='lightgray', lw=0.5, alpha=0.5)

    # Draw current arrows
    for i, seg in enumerate(segments):
        s = seg.start[:2] * 1e3
        e = seg.end[:2] * 1e3
        mid = 0.5 * (s + e)
        d = e - s
        length = np.linalg.norm(d)
        if length < 1e-12:
            continue

        # Arrow direction: along segment if I > 0, reversed if I < 0
        I_real = I_segments[i].real
        direction = d / length
        if I_real < 0:
            direction = -direction

        color = colormap(norm(I_mag[i]))

        if normalize:
            arrow_len = length * 0.7
        else:
            arrow_len = length * 0.7 * I_mag[i] / vmax if vmax > 0 else 0

        ax.annotate('', xy=mid + direction * arrow_len / 2,
                    xytext=mid - direction * arrow_len / 2,
                    arrowprops=dict(arrowstyle='->', color=color,
                                   lw=1.5, alpha=alpha))

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label(r'$|I|$ (A)')

    ax.set_xlabel(r'$x$ (mm)')
    ax.set_ylabel(r'$y$ (mm)')
    if title:
        ax.set_title(title)
    ax.set_aspect('equal')

    return fig, ax
