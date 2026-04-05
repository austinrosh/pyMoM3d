"""Preset cross-section geometry factories.

Provides convenience functions for common transmission line geometries:
microstrip, stripline, CPW, coupled microstrip, and a bridge from
the existing LayerStack.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .geometry import CrossSection, Conductor, DielectricRegion


def microstrip_cross_section(
    W: float,
    h: float,
    eps_r: float,
    t: float = 0.0,
) -> CrossSection:
    """Standard microstrip: signal strip over PEC ground plane.

    Parameters
    ----------
    W : float
        Strip width (m).
    h : float
        Substrate height (m).
    eps_r : float
        Substrate relative permittivity.
    t : float
        Conductor thickness (m).  0 = zero-thickness.

    Returns
    -------
    CrossSection
    """
    conductors = []

    # Ground must span full computational domain
    ground_extent = 100.0 * max(W, h)
    conductors.append(Conductor(
        name='ground',
        x_min=-ground_extent,
        x_max=+ground_extent,
        y_min=0.0,
        y_max=0.0,
        voltage=0.0,
    ))

    # Signal strip at y = h
    if t > 0:
        conductors.append(Conductor(
            name='signal',
            x_min=-W / 2.0,
            x_max=+W / 2.0,
            y_min=h,
            y_max=h + t,
            voltage=1.0,
        ))
    else:
        conductors.append(Conductor(
            name='signal',
            x_min=-W / 2.0,
            x_max=+W / 2.0,
            y_min=h,
            y_max=h,
            voltage=1.0,
        ))

    # Substrate dielectric region
    dielectrics = [
        DielectricRegion(
            name='substrate',
            x_min=-ground_extent,
            x_max=+ground_extent,
            y_min=0.0,
            y_max=h,
            eps_r=eps_r,
        ),
    ]

    return CrossSection(conductors=conductors, dielectric_regions=dielectrics)


def stripline_cross_section(
    W: float,
    b: float,
    eps_r: float,
    h_offset: float = 0.0,
    t: float = 0.0,
) -> CrossSection:
    """Stripline between two PEC ground planes.

    Parameters
    ----------
    W : float
        Strip width (m).
    b : float
        Total ground-plane separation (m).
    eps_r : float
        Dielectric relative permittivity (fills entire cavity).
    h_offset : float
        Vertical offset of strip from center (m).  0 = centered.
    t : float
        Conductor thickness (m).  0 = zero-thickness.

    Returns
    -------
    CrossSection
    """
    z_strip = b / 2.0 + h_offset
    ground_extent = 100.0 * max(W, b)

    conductors = [
        Conductor(
            name='ground_bot',
            x_min=-ground_extent,
            x_max=+ground_extent,
            y_min=0.0,
            y_max=0.0,
            voltage=0.0,
        ),
        Conductor(
            name='ground_top',
            x_min=-ground_extent,
            x_max=+ground_extent,
            y_min=b,
            y_max=b,
            voltage=0.0,
        ),
    ]

    if t > 0:
        conductors.append(Conductor(
            name='signal',
            x_min=-W / 2.0,
            x_max=+W / 2.0,
            y_min=z_strip - t / 2.0,
            y_max=z_strip + t / 2.0,
            voltage=1.0,
        ))
    else:
        conductors.append(Conductor(
            name='signal',
            x_min=-W / 2.0,
            x_max=+W / 2.0,
            y_min=z_strip,
            y_max=z_strip,
            voltage=1.0,
        ))

    dielectrics = [
        DielectricRegion(
            name='dielectric',
            x_min=-ground_extent,
            x_max=+ground_extent,
            y_min=0.0,
            y_max=b,
            eps_r=eps_r,
        ),
    ]

    return CrossSection(conductors=conductors, dielectric_regions=dielectrics)


def cpw_cross_section(
    W: float,
    S: float,
    eps_r: float,
    h: float,
    W_gnd: Optional[float] = None,
    t: float = 0.0,
) -> CrossSection:
    """Coplanar waveguide with finite-width ground planes.

    Parameters
    ----------
    W : float
        Center conductor width (m).
    S : float
        Gap between center conductor and each ground plane (m).
    eps_r : float
        Substrate relative permittivity.
    h : float
        Substrate thickness (m).
    W_gnd : float, optional
        Ground plane width (m).  None = 5× total CPW span.
    t : float
        Conductor thickness (m).  0 = zero-thickness.

    Returns
    -------
    CrossSection
    """
    if W_gnd is None:
        W_gnd = 5.0 * (W + 2 * S)

    # Bottom ground plane (PEC backing)
    backing_extent = W_gnd + W / 2.0 + S
    conductors = [
        Conductor(
            name='ground_bottom',
            x_min=-backing_extent,
            x_max=+backing_extent,
            y_min=0.0,
            y_max=0.0,
            voltage=0.0,
        ),
    ]

    strip_y_min = h
    strip_y_max = h + t if t > 0 else h

    # Signal conductor (center)
    conductors.append(Conductor(
        name='signal',
        x_min=-W / 2.0,
        x_max=+W / 2.0,
        y_min=strip_y_min,
        y_max=strip_y_max,
        voltage=1.0,
    ))

    # Left coplanar ground
    x_gnd_inner = W / 2.0 + S
    conductors.append(Conductor(
        name='ground_left',
        x_min=-(x_gnd_inner + W_gnd),
        x_max=-x_gnd_inner,
        y_min=strip_y_min,
        y_max=strip_y_max,
        voltage=0.0,
    ))

    # Right coplanar ground
    conductors.append(Conductor(
        name='ground_right',
        x_min=+x_gnd_inner,
        x_max=+(x_gnd_inner + W_gnd),
        y_min=strip_y_min,
        y_max=strip_y_max,
        voltage=0.0,
    ))

    dielectrics = [
        DielectricRegion(
            name='substrate',
            x_min=-backing_extent,
            x_max=+backing_extent,
            y_min=0.0,
            y_max=h,
            eps_r=eps_r,
        ),
    ]

    return CrossSection(conductors=conductors, dielectric_regions=dielectrics)


def coupled_microstrip_cross_section(
    W: float,
    S: float,
    h: float,
    eps_r: float,
    t: float = 0.0,
) -> CrossSection:
    """Coupled microstrip differential pair over PEC ground.

    Parameters
    ----------
    W : float
        Width of each strip (m).
    S : float
        Gap between the two strips (m).
    h : float
        Substrate height (m).
    eps_r : float
        Substrate relative permittivity.
    t : float
        Conductor thickness (m).

    Returns
    -------
    CrossSection
    """
    ground_extent = 100.0 * max(2 * W + S, h)
    strip_y_min = h
    strip_y_max = h + t if t > 0 else h

    conductors = [
        Conductor(
            name='ground',
            x_min=-ground_extent,
            x_max=+ground_extent,
            y_min=0.0,
            y_max=0.0,
            voltage=0.0,
        ),
        Conductor(
            name='signal_1',
            x_min=-(S / 2.0 + W),
            x_max=-S / 2.0,
            y_min=strip_y_min,
            y_max=strip_y_max,
            voltage=1.0,
        ),
        Conductor(
            name='signal_2',
            x_min=+S / 2.0,
            x_max=+(S / 2.0 + W),
            y_min=strip_y_min,
            y_max=strip_y_max,
            voltage=1.0,
        ),
    ]

    dielectrics = [
        DielectricRegion(
            name='substrate',
            x_min=-ground_extent,
            x_max=+ground_extent,
            y_min=0.0,
            y_max=h,
            eps_r=eps_r,
        ),
    ]

    return CrossSection(conductors=conductors, dielectric_regions=dielectrics)


def from_layer_stack(
    layer_stack,
    conductors: List[dict],
) -> CrossSection:
    """Build a CrossSection from an existing pyMoM3d LayerStack.

    Maps the 3D layer stack z-axis to the cross-section y-axis.

    Parameters
    ----------
    layer_stack : LayerStack
        pyMoM3d layer stack definition.
    conductors : list of dict
        Conductor definitions.  Each dict has keys:
        - 'name': str
        - 'x_min', 'x_max': float (transverse extent, m)
        - 'z': float (height in the 3D stackup, m) — maps to y
        - 'voltage': float (1.0 for signal, 0.0 for ground)
        - 't': float, optional (thickness, m; default 0)

    Returns
    -------
    CrossSection
    """
    cond_list = []
    for c in conductors:
        t = c.get('t', 0.0)
        z = c['z']
        cond_list.append(Conductor(
            name=c['name'],
            x_min=c['x_min'],
            x_max=c['x_max'],
            y_min=z,
            y_max=z + t if t > 0 else z,
            voltage=c.get('voltage', 0.0),
        ))

    diel_list = []
    ground_extent = max(abs(c['x_min']) for c in conductors)
    ground_extent = max(ground_extent, max(abs(c['x_max']) for c in conductors))
    ground_extent *= 2.0

    for layer in layer_stack.layers:
        if layer.is_pec:
            # Add PEC layer as a ground conductor
            z_pec = layer.z_top if np.isfinite(layer.z_top) else layer.z_bot
            if not np.isfinite(z_pec):
                continue
            cond_list.append(Conductor(
                name=f'pec_{layer.name}',
                x_min=-ground_extent,
                x_max=+ground_extent,
                y_min=z_pec,
                y_max=z_pec,
                voltage=0.0,
            ))
        elif np.isfinite(layer.z_bot) and np.isfinite(layer.z_top):
            eps_r = float(np.real(layer.eps_r))
            if eps_r != 1.0:
                diel_list.append(DielectricRegion(
                    name=layer.name,
                    x_min=-ground_extent,
                    x_max=+ground_extent,
                    y_min=layer.z_bot,
                    y_max=layer.z_top,
                    eps_r=eps_r,
                ))

    return CrossSection(conductors=cond_list, dielectric_regions=diel_list)
