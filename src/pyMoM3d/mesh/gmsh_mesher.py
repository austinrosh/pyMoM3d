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
        Load and mesh a geometry file (STL, STEP, IGES, etc.).

        Parameters
        ----------
        path : str
            Path to geometry file.

        Returns
        -------
        mesh : Mesh
        """
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
