"""Factory functions for standard transmission line geometries.

Each function creates a complete simulation-ready geometry: mesh with
conformal feed lines, RWG basis, ports with feed signs, and a layer stack.
Analytical reference values are included for validation.

Supported geometries:
  - ``microstrip_geometry`` — single strip on PEC-backed substrate
  - ``stripline_geometry`` — strip embedded between two ground planes
  - ``cpw_geometry`` — coplanar waveguide (center strip + two ground planes)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..mesh.gmsh_mesher import GmshMesher
from ..mesh.rwg_connectivity import compute_rwg_connectivity
from ..mesh.mesh_data import Mesh
from ..mesh.rwg_basis import RWGBasis
from ..medium.layer_stack import Layer, LayerStack
from ..network.port import Port
from ..mom.excitation import find_feed_edges, compute_feed_signs
from ..arrays.linear_array import combine_meshes


@dataclass
class TLGeometry:
    """Complete transmission line geometry ready for simulation.

    Attributes
    ----------
    mesh : Mesh
        Triangular surface mesh with conformal feed lines.
    basis : RWGBasis
        RWG basis functions computed from the mesh.
    ports : list of Port
        Two-port definition with feed signs.
    layer_stack : LayerStack
        Stratified medium definition.
    source_layer_name : str
        Name of the layer containing the strip surface.
    port_separation : float
        Distance between port 1 and port 2 (m).
    analytical_z0 : float
        Analytical characteristic impedance (Ohm).
    analytical_eps_eff : float
        Analytical effective permittivity.
    """

    mesh: Mesh
    basis: RWGBasis
    ports: List[Port]
    layer_stack: LayerStack
    source_layer_name: str
    port_separation: float
    analytical_z0: float
    analytical_eps_eff: float


def microstrip_geometry(
    W: float,
    H: float,
    L: float,
    eps_r: float,
    tel: Optional[float] = None,
    margin: float = 0.0,
    use_phantom: bool = True,
) -> TLGeometry:
    """Create a microstrip transmission line geometry.

    Parameters
    ----------
    W : float
        Strip width (m).
    H : float
        Substrate height (m).
    L : float
        Strip length (m).
    eps_r : float
        Substrate relative permittivity.
    tel : float, optional
        Target edge length (m). Defaults to W/3.
    margin : float
        Distance from strip ends to port locations (m).  Default 0.0
        places ports at the strip edges.  When a port is within one
        ``tel`` of a mesh boundary, the mesh is automatically extended
        by a short stub so the feed edges have two adjacent triangles
        (required for RWG basis functions).
    use_phantom : bool
        If True (default), add a thin phantom air layer above the
        substrate for correct Strata DCIM fitting.  See
        :meth:`LayerStack.make_microstrip_stack`.

    Returns
    -------
    TLGeometry
    """
    from ..analysis.transmission_line import microstrip_z0_hammerstad

    if tel is None:
        tel = W / 3.0

    Z0_ana, eps_eff_ana = microstrip_z0_hammerstad(W, H, eps_r)

    if use_phantom:
        stack = LayerStack.make_microstrip_stack(H, eps_r)
        src_layer = 'phantom_air'
    else:
        stack = LayerStack([
            Layer('pec_ground', z_bot=-np.inf, z_top=0.0,
                  eps_r=1.0, is_pec=True),
            Layer('substrate', z_bot=0.0, z_top=H, eps_r=eps_r),
            Layer('air', z_bot=H, z_top=np.inf, eps_r=1.0),
        ])
        src_layer = 'substrate'

    port1_x = -L / 2.0 + margin
    port2_x = +L / 2.0 - margin

    # Ensure mesh extends beyond each port so feed edges are interior
    # (two adjacent triangles → valid RWG basis).  Use at least W or
    # 3mm of stub for proper mode coupling at the port.
    stub = max(tel, min(W, 3.0e-3))
    mesh_x_lo = min(-L / 2.0, port1_x - stub)
    mesh_x_hi = max(+L / 2.0, port2_x + stub)
    mesh_width = mesh_x_hi - mesh_x_lo
    mesh_cx = (mesh_x_lo + mesh_x_hi) / 2.0

    mesher = GmshMesher(target_edge_length=tel)
    mesh = mesher.mesh_plate_with_feeds(
        width=mesh_width, height=W,
        feed_x_list=[port1_x, port2_x],
        center=(mesh_cx, 0.0, H),
    )
    basis = compute_rwg_connectivity(mesh)

    feed1 = find_feed_edges(mesh, basis, feed_x=port1_x)
    feed2 = find_feed_edges(mesh, basis, feed_x=port2_x)
    signs1 = compute_feed_signs(mesh, basis, feed1)
    signs2 = compute_feed_signs(mesh, basis, feed2)

    ports = [
        Port(name='P1', feed_basis_indices=feed1, feed_signs=signs1),
        Port(name='P2', feed_basis_indices=feed2, feed_signs=signs2),
    ]

    return TLGeometry(
        mesh=mesh,
        basis=basis,
        ports=ports,
        layer_stack=stack,
        source_layer_name=src_layer,
        port_separation=port2_x - port1_x,
        analytical_z0=Z0_ana,
        analytical_eps_eff=eps_eff_ana,
    )


def stripline_geometry(
    W: float,
    H1: float,
    H2: float,
    L: float,
    eps_r: float,
    tel: Optional[float] = None,
    margin: float = 3.0e-3,
) -> TLGeometry:
    """Create a stripline transmission line geometry.

    The strip is embedded between two ground planes.  A thin phantom layer
    with eps_r matching the substrate is used to place the source correctly
    (strip IS embedded in the dielectric, so phantom is appropriate here).

    Parameters
    ----------
    W : float
        Strip width (m).
    H1 : float
        Distance from bottom ground to strip (m).
    H2 : float
        Distance from strip to top ground (m).
    L : float
        Strip length (m).
    eps_r : float
        Substrate relative permittivity.
    tel : float, optional
        Target edge length (m). Defaults to W/3.
    margin : float
        Distance from strip ends to port locations (m).

    Returns
    -------
    TLGeometry
    """
    from ..analysis.transmission_line import stripline_z0_cohn

    if tel is None:
        tel = W / 3.0

    b = H1 + H2  # total ground-to-ground spacing
    Z0_ana = stripline_z0_cohn(W, b, eps_r)
    eps_eff_ana = eps_r  # stripline: eps_eff = eps_r (homogeneous)

    # Phantom layer for stripline: eps_r matches substrate
    delta = min(H1, H2) * 0.1
    phantom_eps = eps_r * (1.0 + 1e-3)  # slightly different to avoid Strata merge

    stack = LayerStack([
        Layer('pec_bot', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
        Layer('substrate_bot', z_bot=0.0, z_top=H1, eps_r=eps_r),
        Layer('phantom', z_bot=H1, z_top=H1 + delta, eps_r=phantom_eps),
        Layer('substrate_top', z_bot=H1 + delta, z_top=b, eps_r=eps_r),
        Layer('pec_top', z_bot=b, z_top=np.inf, eps_r=1.0, is_pec=True),
    ])

    z_mesh = H1 + delta / 2.0
    port1_x = -L / 2.0 + margin
    port2_x = +L / 2.0 - margin

    mesher = GmshMesher(target_edge_length=tel)
    mesh = mesher.mesh_plate_with_feeds(
        width=L, height=W,
        feed_x_list=[port1_x, port2_x],
        center=(0.0, 0.0, z_mesh),
    )
    basis = compute_rwg_connectivity(mesh)

    feed1 = find_feed_edges(mesh, basis, feed_x=port1_x)
    feed2 = find_feed_edges(mesh, basis, feed_x=port2_x)
    signs1 = compute_feed_signs(mesh, basis, feed1)
    signs2 = compute_feed_signs(mesh, basis, feed2)

    ports = [
        Port(name='P1', feed_basis_indices=feed1, feed_signs=signs1),
        Port(name='P2', feed_basis_indices=feed2, feed_signs=signs2),
    ]

    return TLGeometry(
        mesh=mesh,
        basis=basis,
        ports=ports,
        layer_stack=stack,
        source_layer_name='phantom',
        port_separation=port2_x - port1_x,
        analytical_z0=Z0_ana,
        analytical_eps_eff=eps_eff_ana,
    )


def cpw_geometry(
    W: float,
    S: float,
    W_gnd: float,
    H: float,
    L: float,
    eps_r: float,
    tel: Optional[float] = None,
    margin: float = 2.0e-3,
) -> TLGeometry:
    """Create a coplanar waveguide (CPW) geometry.

    Three-conductor CPW: center strip + two ground planes on a
    PEC-backed substrate.

    Parameters
    ----------
    W : float
        Center strip width (m).
    S : float
        Gap between center strip and each ground plane (m).
    W_gnd : float
        Width of each ground plane (m).
    H : float
        Substrate height (m).
    L : float
        Line length (m).
    eps_r : float
        Substrate relative permittivity.
    tel : float, optional
        Target edge length (m). Defaults to min(W, S) / 2.
    margin : float
        Distance from strip ends to port locations (m).

    Returns
    -------
    TLGeometry
    """
    from ..analysis.transmission_line import cpw_z0_conformal

    if tel is None:
        tel = min(W, S) / 2.0

    Z0_ana, eps_eff_ana = cpw_z0_conformal(W, S, eps_r, H)

    stack = LayerStack([
        Layer('pec_ground', z_bot=-np.inf, z_top=0.0, eps_r=1.0, is_pec=True),
        Layer('substrate', z_bot=0.0, z_top=H, eps_r=eps_r),
        Layer('air', z_bot=H, z_top=np.inf, eps_r=1.0),
    ])

    port1_x = -L / 2.0 + margin
    port2_x = +L / 2.0 - margin

    mesher = GmshMesher(target_edge_length=tel)

    # Center strip with feed lines
    center_mesh = mesher.mesh_plate_with_feeds(
        width=L, height=W,
        feed_x_list=[port1_x, port2_x],
        center=(0.0, 0.0, H),
    )

    # Ground planes
    y_left = -(W / 2.0 + S + W_gnd / 2.0)
    y_right = +(W / 2.0 + S + W_gnd / 2.0)

    left_gnd = mesher.mesh_plate(width=L, height=W_gnd, center=(0.0, y_left, H))
    right_gnd = mesher.mesh_plate(width=L, height=W_gnd, center=(0.0, y_right, H))

    combined_mesh, _ = combine_meshes([center_mesh, left_gnd, right_gnd])
    basis = compute_rwg_connectivity(combined_mesh)

    # Feed edges on center strip only (filter by y-range)
    y_min = -W / 2.0 - tel
    y_max = +W / 2.0 + tel
    feed1 = find_feed_edges(combined_mesh, basis, feed_x=port1_x,
                            y_range=(y_min, y_max))
    feed2 = find_feed_edges(combined_mesh, basis, feed_x=port2_x,
                            y_range=(y_min, y_max))
    signs1 = compute_feed_signs(combined_mesh, basis, feed1)
    signs2 = compute_feed_signs(combined_mesh, basis, feed2)

    ports = [
        Port(name='P1', feed_basis_indices=feed1, feed_signs=signs1),
        Port(name='P2', feed_basis_indices=feed2, feed_signs=signs2),
    ]

    return TLGeometry(
        mesh=combined_mesh,
        basis=basis,
        ports=ports,
        layer_stack=stack,
        source_layer_name='substrate',
        port_separation=port2_x - port1_x,
        analytical_z0=Z0_ana,
        analytical_eps_eff=eps_eff_ana,
    )
