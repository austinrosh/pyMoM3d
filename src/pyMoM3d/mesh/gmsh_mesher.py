"""Gmsh-based mesh generation utilities.

Provides GmshMesher class that uses the Gmsh Python API for high-quality
surface meshing with control over element size, curvature adaptation,
and local refinement.
"""

import numpy as np
from typing import Optional, Tuple

from .mesh_data import Mesh

try:
    import gmsh
    HAS_GMSH = True
except ImportError:
    HAS_GMSH = False


def _require_gmsh():
    """Raise ImportError if gmsh is not available."""
    if not HAS_GMSH:
        raise ImportError(
            "gmsh is required for GmshMesher. Install with: pip install gmsh"
        )


class GmshMesher:
    """
    Mesh generator using the Gmsh Python API.

    Provides high-quality surface meshing with control over element size,
    curvature adaptation, and local refinement. Supports all geometry
    primitives and STL/STEP/IGES import.

    Parameters
    ----------
    target_edge_length : float, optional
        Target edge length in meters. Controls global mesh density.
        If None, Gmsh uses its default sizing.
    min_edge_length : float, optional
        Minimum allowed edge length. Defaults to target_edge_length / 5.
    max_edge_length : float, optional
        Maximum allowed edge length. Defaults to target_edge_length * 2.
    curvature_adapt : bool, default True
        Enable curvature-based mesh refinement.
    verbosity : int, default 0
        Gmsh verbosity level (0=silent, 1=errors, 2=warnings, 5=debug).
    """

    def __init__(
        self,
        target_edge_length: Optional[float] = None,
        min_edge_length: Optional[float] = None,
        max_edge_length: Optional[float] = None,
        curvature_adapt: bool = True,
        verbosity: int = 0,
    ):
        _require_gmsh()
        self.target_edge_length = target_edge_length
        self.min_edge_length = min_edge_length
        self.max_edge_length = max_edge_length
        self.curvature_adapt = curvature_adapt
        self.verbosity = verbosity

    def _init_gmsh(self):
        """Initialize a fresh Gmsh session."""
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", self.verbosity)
        gmsh.model.add("pyMoM3d")

    def _set_mesh_sizes(self):
        """Apply global mesh size options."""
        if self.target_edge_length is not None:
            lc = self.target_edge_length
            lc_min = self.min_edge_length if self.min_edge_length is not None else lc / 5.0
            lc_max = self.max_edge_length if self.max_edge_length is not None else lc * 2.0
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_min)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc_max)

        if self.curvature_adapt:
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)

    def _extract_surface_mesh(self) -> Mesh:
        """Extract vertices and triangles from the current Gmsh model.

        Returns
        -------
        mesh : Mesh
            Mesh object with vertices and triangles.
        """
        # Get all nodes
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        # coords is flat [x1,y1,z1,x2,y2,z2,...]
        coords = np.array(coords, dtype=np.float64).reshape(-1, 3)

        # Build tag -> index map (Gmsh tags start at 1 and may have gaps)
        tag_to_idx = {}
        for i, tag in enumerate(node_tags):
            tag_to_idx[int(tag)] = i

        vertices = coords

        # Get all 2D (surface) elements — type 2 = 3-node triangle
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)

        triangles_list = []
        for etype, etags, enodes in zip(elem_types, elem_tags, elem_node_tags):
            if etype == 2:  # 3-node triangle
                nodes = np.array(enodes, dtype=np.int64).reshape(-1, 3)
                for tri in nodes:
                    triangles_list.append([tag_to_idx[int(t)] for t in tri])

        if len(triangles_list) == 0:
            raise RuntimeError("Gmsh generated no surface triangles")

        triangles = np.array(triangles_list, dtype=np.int32)

        return Mesh(vertices, triangles)

    def _finalize_gmsh(self):
        """Finalize the Gmsh session."""
        gmsh.finalize()

    def _generate_mesh(self) -> Mesh:
        """Synchronize, set sizes, generate 2D mesh, and extract."""
        gmsh.model.occ.synchronize()
        self._set_mesh_sizes()
        gmsh.model.mesh.generate(2)
        mesh = self._extract_surface_mesh()
        self._finalize_gmsh()
        return mesh

    def mesh_sphere(
        self,
        radius: float,
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        target_edge_length: Optional[float] = None,
    ) -> Mesh:
        """
        Generate a triangular surface mesh for a sphere.

        Parameters
        ----------
        radius : float
            Sphere radius in meters.
        center : tuple of float
            Center coordinates (x, y, z).
        target_edge_length : float, optional
            Override instance-level target edge length for this mesh.

        Returns
        -------
        mesh : Mesh
        """
        self._init_gmsh()
        try:
            if target_edge_length is not None:
                old = self.target_edge_length
                self.target_edge_length = target_edge_length

            gmsh.model.occ.addSphere(*center, radius)
            mesh = self._generate_mesh()

            if target_edge_length is not None:
                self.target_edge_length = old
        except Exception:
            self._finalize_gmsh()
            raise

        return mesh

    def mesh_plate(
        self,
        width: float,
        height: float,
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        target_edge_length: Optional[float] = None,
    ) -> Mesh:
        """
        Generate a triangular surface mesh for a rectangular plate.

        The plate lies in the xy-plane.

        Parameters
        ----------
        width : float
            Width along x-axis in meters.
        height : float
            Height along y-axis in meters.
        center : tuple of float
            Center coordinates (x, y, z).
        target_edge_length : float, optional
            Override instance-level target edge length.

        Returns
        -------
        mesh : Mesh
        """
        self._init_gmsh()
        try:
            if target_edge_length is not None:
                old = self.target_edge_length
                self.target_edge_length = target_edge_length

            cx, cy, cz = center
            x0 = cx - width / 2.0
            y0 = cy - height / 2.0

            gmsh.model.occ.addRectangle(x0, y0, cz, width, height)
            mesh = self._generate_mesh()

            if target_edge_length is not None:
                self.target_edge_length = old
        except Exception:
            self._finalize_gmsh()
            raise

        return mesh

    def mesh_cylinder(
        self,
        radius: float,
        height: float,
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        target_edge_length: Optional[float] = None,
    ) -> Mesh:
        """
        Generate a triangular surface mesh for a closed cylinder.

        The cylinder axis is along z. The center is the midpoint of the axis.

        Parameters
        ----------
        radius : float
            Cylinder radius in meters.
        height : float
            Cylinder height in meters.
        center : tuple of float
            Center coordinates (x, y, z).
        target_edge_length : float, optional
            Override instance-level target edge length.

        Returns
        -------
        mesh : Mesh
        """
        self._init_gmsh()
        try:
            if target_edge_length is not None:
                old = self.target_edge_length
                self.target_edge_length = target_edge_length

            cx, cy, cz = center
            z0 = cz - height / 2.0
            gmsh.model.occ.addCylinder(cx, cy, z0, 0, 0, height, radius)
            mesh = self._generate_mesh()

            if target_edge_length is not None:
                self.target_edge_length = old
        except Exception:
            self._finalize_gmsh()
            raise

        return mesh

    def mesh_cube(
        self,
        side_length: float,
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        target_edge_length: Optional[float] = None,
    ) -> Mesh:
        """
        Generate a triangular surface mesh for a cube.

        Parameters
        ----------
        side_length : float
            Side length in meters.
        center : tuple of float
            Center coordinates (x, y, z).
        target_edge_length : float, optional
            Override instance-level target edge length.

        Returns
        -------
        mesh : Mesh
        """
        self._init_gmsh()
        try:
            if target_edge_length is not None:
                old = self.target_edge_length
                self.target_edge_length = target_edge_length

            cx, cy, cz = center
            s = side_length
            x0 = cx - s / 2.0
            y0 = cy - s / 2.0
            z0 = cz - s / 2.0
            gmsh.model.occ.addBox(x0, y0, z0, s, s, s)
            mesh = self._generate_mesh()

            if target_edge_length is not None:
                self.target_edge_length = old
        except Exception:
            self._finalize_gmsh()
            raise

        return mesh

    def mesh_pyramid(
        self,
        base_size: float,
        height: float,
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        target_edge_length: Optional[float] = None,
    ) -> Mesh:
        """
        Generate a triangular surface mesh for a square pyramid.

        The base lies in the z=center[2] plane. The apex is at
        z=center[2]+height.

        Parameters
        ----------
        base_size : float
            Side length of the square base in meters.
        height : float
            Pyramid height in meters.
        center : tuple of float
            Center of the base (x, y, z).
        target_edge_length : float, optional
            Override instance-level target edge length.

        Returns
        -------
        mesh : Mesh
        """
        self._init_gmsh()
        try:
            if target_edge_length is not None:
                old = self.target_edge_length
                self.target_edge_length = target_edge_length

            cx, cy, cz = center
            s = base_size / 2.0

            # Define the 5 vertices of the pyramid
            p1 = gmsh.model.occ.addPoint(cx - s, cy - s, cz)
            p2 = gmsh.model.occ.addPoint(cx + s, cy - s, cz)
            p3 = gmsh.model.occ.addPoint(cx + s, cy + s, cz)
            p4 = gmsh.model.occ.addPoint(cx - s, cy + s, cz)
            p5 = gmsh.model.occ.addPoint(cx, cy, cz + height)  # apex

            # Base edges
            l1 = gmsh.model.occ.addLine(p1, p2)
            l2 = gmsh.model.occ.addLine(p2, p3)
            l3 = gmsh.model.occ.addLine(p3, p4)
            l4 = gmsh.model.occ.addLine(p4, p1)

            # Side edges
            l5 = gmsh.model.occ.addLine(p1, p5)
            l6 = gmsh.model.occ.addLine(p2, p5)
            l7 = gmsh.model.occ.addLine(p3, p5)
            l8 = gmsh.model.occ.addLine(p4, p5)

            # Base face
            cl_base = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
            gmsh.model.occ.addPlaneSurface([cl_base])

            # Four triangular side faces
            cl_f = gmsh.model.occ.addCurveLoop([l1, l6, -l5])
            gmsh.model.occ.addPlaneSurface([cl_f])

            cl_r = gmsh.model.occ.addCurveLoop([l2, l7, -l6])
            gmsh.model.occ.addPlaneSurface([cl_r])

            cl_b = gmsh.model.occ.addCurveLoop([l3, l8, -l7])
            gmsh.model.occ.addPlaneSurface([cl_b])

            cl_l = gmsh.model.occ.addCurveLoop([l4, l5, -l8])
            gmsh.model.occ.addPlaneSurface([cl_l])

            mesh = self._generate_mesh()

            if target_edge_length is not None:
                self.target_edge_length = old
        except Exception:
            self._finalize_gmsh()
            raise

        return mesh

    def mesh_plate_with_feed(
        self,
        width: float,
        height: float,
        feed_x: float = 0.0,
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        target_edge_length: Optional[float] = None,
    ) -> Mesh:
        """
        Generate a plate mesh with a forced transverse mesh line at feed_x.

        Splits the plate into two halves joined at x=feed_x, ensuring
        conformal transverse edges exist at the feed location. This is
        essential for delta-gap feed models on strip dipoles.

        Parameters
        ----------
        width : float
            Width along x-axis in meters.
        height : float
            Height along y-axis in meters.
        feed_x : float
            x-coordinate of the feed line (default 0.0).
        center : tuple of float
            Center coordinates (x, y, z).
        target_edge_length : float, optional
            Override instance-level target edge length.

        Returns
        -------
        mesh : Mesh
        """
        self._init_gmsh()
        try:
            if target_edge_length is not None:
                old = self.target_edge_length
                self.target_edge_length = target_edge_length

            cx, cy, cz = center
            x0 = cx - width / 2.0
            y0 = cy - height / 2.0

            # Create two rectangles sharing the feed line
            w_left = feed_x - x0
            w_right = (x0 + width) - feed_x

            rect_left = gmsh.model.occ.addRectangle(x0, y0, cz, w_left, height)
            rect_right = gmsh.model.occ.addRectangle(feed_x, y0, cz, w_right, height)

            # Fragment to merge shared boundary — ensures conformal mesh
            gmsh.model.occ.fragment(
                [(2, rect_left)], [(2, rect_right)]
            )

            mesh = self._generate_mesh()

            if target_edge_length is not None:
                self.target_edge_length = old
        except Exception:
            self._finalize_gmsh()
            raise

        return mesh

    def mesh_plate_with_feeds(
        self,
        width: float,
        height: float,
        feed_x_list: list,
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        target_edge_length: Optional[float] = None,
    ) -> Mesh:
        """Generate a plate mesh with forced transverse mesh lines at multiple x-coordinates.

        Creates N+1 rectangular segments joined at each feed_x, ensuring
        conformal transverse edges at every port location.

        Parameters
        ----------
        width : float
            Width along x-axis in meters.
        height : float
            Height along y-axis in meters.
        feed_x_list : list of float
            x-coordinates of feed lines (must be within plate extent).
        center : tuple of float
            Center coordinates (x, y, z).
        target_edge_length : float, optional
            Override instance-level target edge length.

        Returns
        -------
        mesh : Mesh
        """
        self._init_gmsh()
        try:
            if target_edge_length is not None:
                old = self.target_edge_length
                self.target_edge_length = target_edge_length

            cx, cy, cz = center
            x0 = cx - width / 2.0
            y0 = cy - height / 2.0

            # Sort feed positions and build segment boundaries
            cuts = sorted(feed_x_list)
            boundaries = [x0] + cuts + [x0 + width]

            rects = []
            for i in range(len(boundaries) - 1):
                w_seg = boundaries[i + 1] - boundaries[i]
                if w_seg > 1e-15:
                    tag = gmsh.model.occ.addRectangle(
                        boundaries[i], y0, cz, w_seg, height
                    )
                    rects.append(tag)

            # Fragment all rectangles to merge shared boundaries
            if len(rects) > 1:
                tool_dimtags = [(2, t) for t in rects[1:]]
                gmsh.model.occ.fragment([(2, rects[0])], tool_dimtags)

            # Add midpoint vertices at the center of each short edge
            # so probe feeds land at the correct position.  We split
            # the short-edge boundary curves by inserting a point.
            gmsh.model.occ.synchronize()
            y_mid = cy
            for x_end in [x0, x0 + width]:
                pt = gmsh.model.occ.addPoint(x_end, y_mid, cz)
            gmsh.model.occ.synchronize()

            # Re-fragment to incorporate the new points into the mesh
            surfs = gmsh.model.getEntities(dim=2)
            pts = gmsh.model.getEntities(dim=0)
            if surfs and pts:
                gmsh.model.occ.fragment(surfs, pts)

            mesh = self._generate_mesh()

            if target_edge_length is not None:
                self.target_edge_length = old
        except Exception:
            self._finalize_gmsh()
            raise

        return mesh

    def mesh_stepped_plate(
        self,
        width1: float,
        height1: float,
        width2: float,
        height2: float,
        feed_x_list: list = None,
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        target_edge_length: Optional[float] = None,
    ) -> Mesh:
        """Generate a stepped-width plate mesh (two sections of different widths).

        The step junction is at x = cx. The left section extends from
        cx - width1 to cx with height1, and the right section extends from
        cx to cx + width2 with height2. Both sections are y-centered on cy.

        Parameters
        ----------
        width1 : float
            Length of left section along x (m).
        height1 : float
            Width of left section along y (m).
        width2 : float
            Length of right section along x (m).
        height2 : float
            Width of right section along y (m).
        feed_x_list : list of float, optional
            x-coordinates for conformal transverse edges (ports, ref planes).
        center : tuple of float
            Center of the step junction (x, y, z).
        target_edge_length : float, optional
            Override instance-level target edge length.

        Returns
        -------
        mesh : Mesh
        """
        self._init_gmsh()
        try:
            if target_edge_length is not None:
                old = self.target_edge_length
                self.target_edge_length = target_edge_length

            cx, cy, cz = center
            x_left = cx - width1
            x_right = cx + width2

            # Build sub-rectangles for left section (height1)
            cuts_left = [x_left, cx]
            if feed_x_list:
                for fx in sorted(feed_x_list):
                    if x_left + 1e-15 < fx < cx - 1e-15:
                        cuts_left.append(fx)
            cuts_left = sorted(set(cuts_left))

            all_rects = []
            for i in range(len(cuts_left) - 1):
                w_seg = cuts_left[i + 1] - cuts_left[i]
                if w_seg > 1e-15:
                    tag = gmsh.model.occ.addRectangle(
                        cuts_left[i], cy - height1 / 2.0, cz, w_seg, height1
                    )
                    all_rects.append(tag)

            # Build sub-rectangles for right section (height2)
            cuts_right = [cx, x_right]
            if feed_x_list:
                for fx in sorted(feed_x_list):
                    if cx + 1e-15 < fx < x_right - 1e-15:
                        cuts_right.append(fx)
            cuts_right = sorted(set(cuts_right))

            for i in range(len(cuts_right) - 1):
                w_seg = cuts_right[i + 1] - cuts_right[i]
                if w_seg > 1e-15:
                    tag = gmsh.model.occ.addRectangle(
                        cuts_right[i], cy - height2 / 2.0, cz, w_seg, height2
                    )
                    all_rects.append(tag)

            # Fragment all to merge shared boundaries
            if len(all_rects) > 1:
                tool_dimtags = [(2, t) for t in all_rects[1:]]
                gmsh.model.occ.fragment([(2, all_rects[0])], tool_dimtags)

            mesh = self._generate_mesh()

            if target_edge_length is not None:
                self.target_edge_length = old
        except Exception:
            self._finalize_gmsh()
            raise

        return mesh

    def mesh_t_junction(
        self,
        main_width: float,
        main_height: float,
        stub_width: float,
        stub_height: float,
        stub_x: float = 0.0,
        feed_x_list: list = None,
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        target_edge_length: Optional[float] = None,
    ) -> Mesh:
        """Generate a T-junction mesh (main strip with perpendicular stub).

        The main strip is centered at (cx, cy, cz). The stub extends in
        the +y direction from the top edge of the main strip.

        Parameters
        ----------
        main_width : float
            Main strip length along x (m).
        main_height : float
            Main strip width along y (m).
        stub_width : float
            Stub width along x (m).
        stub_height : float
            Stub length along y, extending in +y from main strip (m).
        stub_x : float
            x-position of stub center (default 0.0).
        feed_x_list : list of float, optional
            x-coordinates for conformal transverse edges on the main strip.
        center : tuple of float
            Center of main strip (x, y, z).
        target_edge_length : float, optional
            Override instance-level target edge length.

        Returns
        -------
        mesh : Mesh
        """
        self._init_gmsh()
        try:
            if target_edge_length is not None:
                old = self.target_edge_length
                self.target_edge_length = target_edge_length

            cx, cy, cz = center
            x_lo = cx - main_width / 2.0
            x_hi = cx + main_width / 2.0
            stub_y_bot = cy + main_height / 2.0

            # Build sub-rectangles for main strip with feed_x subdivisions
            cuts = [x_lo, x_hi]
            if feed_x_list:
                for fx in sorted(feed_x_list):
                    if x_lo + 1e-15 < fx < x_hi - 1e-15:
                        cuts.append(fx)
            cuts = sorted(set(cuts))

            all_rects = []
            for i in range(len(cuts) - 1):
                w_seg = cuts[i + 1] - cuts[i]
                if w_seg > 1e-15:
                    tag = gmsh.model.occ.addRectangle(
                        cuts[i], cy - main_height / 2.0, cz,
                        w_seg, main_height
                    )
                    all_rects.append(tag)

            # Stub extending in +y from the top edge of main strip
            rect_stub = gmsh.model.occ.addRectangle(
                stub_x - stub_width / 2.0, stub_y_bot, cz,
                stub_width, stub_height
            )
            all_rects.append(rect_stub)

            # Fragment all to merge shared boundaries
            if len(all_rects) > 1:
                tool_dimtags = [(2, t) for t in all_rects[1:]]
                gmsh.model.occ.fragment([(2, all_rects[0])], tool_dimtags)

            mesh = self._generate_mesh()

            if target_edge_length is not None:
                self.target_edge_length = old
        except Exception:
            self._finalize_gmsh()
            raise

        return mesh

    def mesh_microstrip_with_edge_ports(
        self,
        width: float,
        length: float,
        substrate_height: float,
        port_edges: list = None,
        center: Tuple[float, float, float] = None,
        feed_x_list: list = None,
        plate_z_offset: float = 0.0,
        target_edge_length: Optional[float] = None,
    ) -> Mesh:
        """Generate a microstrip strip with vertical plates at edge ports.

        Creates a horizontal strip at z = center_z (typically z = h_sub) and
        vertical rectangular plates at the specified ends, extending from the
        strip down toward the ground plane.  The junction edges between strip
        and plates are conformal, so standard RWG basis functions span the
        junction automatically.

        This models the edge-fed port geometry used by Makarov (Ch. 10.3),
        Okhmatovski (Fig. 2a), and Momentum.

        Parameters
        ----------
        width : float
            Strip width along y-axis (m).
        length : float
            Strip length along x-axis (m).
        substrate_height : float
            Height from ground plane to strip (m).
        port_edges : list of str, optional
            Which ends get plates: subset of ['left', 'right'].
            Default: ['left', 'right'] (both ends).
        center : tuple of float, optional
            Center of the strip (x, y, z).  Default: (0, 0, substrate_height).
        feed_x_list : list of float, optional
            Additional x-coordinates for conformal transverse edges on the
            strip (e.g. for SOC reference planes).
        plate_z_offset : float, optional
            Offset of plate bottom from ground plane (m).  Default 0.0.
            A small positive offset (e.g. substrate_height / 10) avoids
            placing mesh triangles at the PEC ground boundary, where the
            layered Green's function image cancellation can cause numerical
            issues in full-wave MPIE assembly.
        target_edge_length : float, optional
            Override instance-level target edge length.

        Returns
        -------
        mesh : Mesh
            Combined mesh with strip and vertical plates.  Feed edges for
            port excitation are at the strip-plate junction (z = center_z,
            x = strip edge).
        """
        if port_edges is None:
            port_edges = ['left', 'right']
        if center is None:
            center = (0.0, 0.0, substrate_height)

        cx, cy, cz = center
        z_ground = cz - substrate_height + plate_z_offset

        x_left = cx - length / 2.0
        x_right = cx + length / 2.0
        y_bot = cy - width / 2.0

        self._init_gmsh()
        try:
            if target_edge_length is not None:
                old = self.target_edge_length
                self.target_edge_length = target_edge_length

            all_surfaces = []

            # --- Horizontal strip segments ---
            # Build strip segments with feed_x subdivisions
            cuts = [x_left, x_right]
            if feed_x_list:
                for fx in sorted(feed_x_list):
                    if x_left + 1e-15 < fx < x_right - 1e-15:
                        cuts.append(fx)
            cuts = sorted(set(cuts))

            for i in range(len(cuts) - 1):
                w_seg = cuts[i + 1] - cuts[i]
                if w_seg > 1e-15:
                    tag = gmsh.model.occ.addRectangle(
                        cuts[i], y_bot, cz, w_seg, width
                    )
                    all_surfaces.append(tag)

            # --- Vertical plates at port edges ---
            if 'left' in port_edges:
                # Plate at x = x_left, in the xz-plane, width along y
                # Gmsh addRectangle places in xy-plane; we need to rotate
                # or build manually. Use manual point-line-surface approach
                # for the vertical plate.
                plate_tag = self._add_vertical_plate(
                    x=x_left, y_bot=y_bot, y_top=y_bot + width,
                    z_bot=z_ground, z_top=cz,
                )
                all_surfaces.append(plate_tag)

            if 'right' in port_edges:
                plate_tag = self._add_vertical_plate(
                    x=x_right, y_bot=y_bot, y_top=y_bot + width,
                    z_bot=z_ground, z_top=cz,
                )
                all_surfaces.append(plate_tag)

            # Fragment all surfaces to merge shared boundaries (junction edges)
            if len(all_surfaces) > 1:
                tool_dimtags = [(2, t) for t in all_surfaces[1:]]
                gmsh.model.occ.fragment(
                    [(2, all_surfaces[0])], tool_dimtags
                )

            mesh = self._generate_mesh()

            if target_edge_length is not None:
                self.target_edge_length = old
        except Exception:
            self._finalize_gmsh()
            raise

        return mesh

    def _add_vertical_plate(
        self, x: float, y_bot: float, y_top: float,
        z_bot: float, z_top: float,
    ) -> int:
        """Add a vertical rectangular plate at constant x using OCC.

        The plate lies in the yz-plane at the given x-coordinate.

        Parameters
        ----------
        x : float
            x-coordinate of the plate.
        y_bot, y_top : float
            y-extents of the plate.
        z_bot, z_top : float
            z-extents of the plate (ground to strip).

        Returns
        -------
        tag : int
            Gmsh surface tag.
        """
        p1 = gmsh.model.occ.addPoint(x, y_bot, z_bot)
        p2 = gmsh.model.occ.addPoint(x, y_top, z_bot)
        p3 = gmsh.model.occ.addPoint(x, y_top, z_top)
        p4 = gmsh.model.occ.addPoint(x, y_bot, z_top)

        l1 = gmsh.model.occ.addLine(p1, p2)  # bottom edge (y-dir)
        l2 = gmsh.model.occ.addLine(p2, p3)  # right edge (z-dir)
        l3 = gmsh.model.occ.addLine(p3, p4)  # top edge (y-dir)
        l4 = gmsh.model.occ.addLine(p4, p1)  # left edge (z-dir)

        cl = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
        surf = gmsh.model.occ.addPlaneSurface([cl])
        return surf

    def mesh_from_geometry(self, geometry, **kwargs) -> Mesh:
        """
        Generate a mesh from a pyMoM3d geometry primitive.

        Dispatches to the appropriate meshing method based on the geometry type.

        Parameters
        ----------
        geometry : RectangularPlate, Sphere, Cylinder, Cube, or Pyramid
            Geometry primitive instance.
        **kwargs
            Additional keyword arguments passed to the specific meshing method.

        Returns
        -------
        mesh : Mesh
        """
        # Import here to avoid circular imports
        from ..geometry.primitives import (
            RectangularPlate, Sphere, Cylinder, Cube, Pyramid
        )

        if isinstance(geometry, Sphere):
            return self.mesh_sphere(
                radius=geometry.radius,
                center=tuple(geometry.center),
                **kwargs,
            )
        elif isinstance(geometry, RectangularPlate):
            return self.mesh_plate(
                width=geometry.width,
                height=geometry.height,
                center=tuple(geometry.center),
                **kwargs,
            )
        elif isinstance(geometry, Cylinder):
            return self.mesh_cylinder(
                radius=geometry.radius,
                height=geometry.height,
                center=tuple(geometry.center),
                **kwargs,
            )
        elif isinstance(geometry, Cube):
            return self.mesh_cube(
                side_length=geometry.side_length,
                center=tuple(geometry.center),
                **kwargs,
            )
        elif isinstance(geometry, Pyramid):
            return self.mesh_pyramid(
                base_size=geometry.base_size,
                height=geometry.height,
                center=tuple(geometry.center),
                **kwargs,
            )
        else:
            raise TypeError(
                f"Unsupported geometry type: {type(geometry).__name__}. "
                f"Use mesh_sphere/mesh_plate/etc. directly or pass a "
                f"supported geometry primitive."
            )

    def mesh_from_file(self, path: str) -> Mesh:
        """
        Load and remesh a geometry file (STL, OBJ, STEP, IGES, etc.).

        For discrete mesh formats (STL, OBJ), the imported triangulation is
        reclassified into geometric surfaces so that Gmsh can generate a
        fresh mesh with the requested element size.  For CAD formats (STEP,
        IGES), standard CAD meshing is used.

        Parameters
        ----------
        path : str
            Path to geometry file.

        Returns
        -------
        mesh : Mesh
        """
        import os
        ext = os.path.splitext(path)[1].lower()

        if ext in ('.stl', '.obj'):
            return self._remesh_discrete(path)

        # CAD formats (STEP, IGES, etc.)
        self._init_gmsh()
        try:
            gmsh.merge(path)
            gmsh.model.occ.synchronize()
            self._set_mesh_sizes()
            gmsh.model.mesh.generate(2)
            mesh = self._extract_surface_mesh()
            self._finalize_gmsh()
        except Exception:
            self._finalize_gmsh()
            raise
        return mesh

    def _remesh_discrete(self, path: str) -> Mesh:
        """Remesh an STL or OBJ file.

        Loads the mesh and, if ``target_edge_length`` is set, uniformly
        refines it until the mean edge length is at or below the target.
        Each refinement pass subdivides every triangle into four,
        halving the mean edge length.

        This approach is robust for arbitrary real-world STL/OBJ files
        and avoids the slow/unreliable parametric re-meshing pipeline
        (classifySurfaces + createGeometry + generate) which can hang
        on complex meshes.
        """
        self._init_gmsh()
        try:
            gmsh.merge(path)

            if self.target_edge_length is not None:
                # Estimate current mean edge length from the imported mesh
                node_tags, coords, _ = gmsh.model.mesh.getNodes()
                elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
                verts = np.array(coords, dtype=np.float64).reshape(-1, 3)
                tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

                edge_set = set()
                for etype, enodes in zip(elem_types, elem_node_tags):
                    if etype == 2:
                        tris = np.array(enodes, dtype=np.int64).reshape(-1, 3)
                        for tri in tris:
                            idxs = [tag_to_idx[int(t)] for t in tri]
                            for a, b in [(0, 1), (1, 2), (2, 0)]:
                                edge_set.add((min(idxs[a], idxs[b]),
                                              max(idxs[a], idxs[b])))

                if edge_set:
                    edges_arr = np.array(list(edge_set))
                    lengths = np.linalg.norm(
                        verts[edges_arr[:, 0]] - verts[edges_arr[:, 1]],
                        axis=1,
                    )
                    mean_edge = float(np.mean(lengths))
                else:
                    mean_edge = 0.0

                # Each refine() halves mean edge length
                target = self.target_edge_length
                refinements = 0
                while mean_edge > target and refinements < 5:
                    gmsh.model.mesh.refine()
                    mean_edge /= 2.0
                    refinements += 1

            mesh = self._extract_surface_mesh()
            self._finalize_gmsh()
        except Exception:
            self._finalize_gmsh()
            raise
        return mesh
