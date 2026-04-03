"""Loop-star decomposition for low-frequency stabilisation of the EFIE.

At low frequencies (kD << 1), the EFIE impedance matrix becomes
ill-conditioned because the scalar-potential term (proportional to 1/omega)
dominates the vector-potential term (proportional to omega) by a factor of
~(kD)^{-2}.  Standard RWG basis functions mix both contributions, making
the system unsolvable when kD < ~0.01.

Loop-star decomposition separates the RWG basis into:

* **Loop functions** (divergence-free): span the null space of the surface
  divergence operator.  The impedance sub-block Z_LL contains only the
  vector-potential (inductance) contribution -- no scalar-potential
  contamination.

* **Star functions** (irrotational complement): carry all the charge
  (scalar-potential) information.  Z_SS is dominated by the scalar potential
  and is well-conditioned after frequency rescaling.

After transformation to the loop-star basis and diagonal frequency
rescaling, the condition number becomes O(1) at all frequencies.

Algorithm
---------
1. Build the **dual graph** of the mesh (nodes = triangles, edges = RWG
   basis functions connecting their T+ and T- triangles).
2. Find a **spanning tree** of the dual graph via BFS.
3. **Tree edges** -> star basis functions (N_t - 1 functions, one per
   non-root triangle).
4. **Co-tree edges** -> loop basis functions.  Each co-tree edge defines a
   fundamental cycle in the dual graph; the loop function is the signed sum
   of RWG functions around the cycle, scaled by 1/l_e so that divergences
   cancel exactly.

References
----------
* Vecchi, G. (1999). "Loop-star decomposition of basis functions in the
  discretization of the EFIE." IEEE Trans. Antennas Propag., 47(2).
* Zhao, J.-S. and Chew, W. C. (2000). "Integral equation solution of
  Maxwell's equations from zero frequency to microwave frequencies."
  IEEE Trans. Antennas Propag., 48(10).
"""

from __future__ import annotations

from collections import deque
from typing import Tuple

import numpy as np
from scipy import sparse


def build_loop_star_basis(rwg_basis, mesh) -> Tuple[sparse.csc_matrix, int]:
    """Construct the loop-star transformation matrix P.

    Parameters
    ----------
    rwg_basis : RWGBasis
        RWG basis function data (t_plus, t_minus, edge_length, etc.).
    mesh : Mesh
        Triangular surface mesh.

    Returns
    -------
    P : scipy.sparse.csc_matrix, shape (N_basis, N_basis)
        Transformation matrix:  J_rwg = P @ J_ls.
        Columns [0 : n_loops] are loop (divergence-free) basis functions.
        Columns [n_loops :] are star basis functions.
    n_loops : int
        Number of loop basis functions.

    Raises
    ------
    ValueError
        If the dual mesh graph is disconnected.
    """
    N = rwg_basis.num_basis
    N_t = len(mesh.triangles)

    # ------------------------------------------------------------------
    # 1. Build dual-graph adjacency list
    #    adj[t] = [(neighbour_triangle, basis_function_index), ...]
    # ------------------------------------------------------------------
    adj = [[] for _ in range(N_t)]
    for n in range(N):
        tp = int(rwg_basis.t_plus[n])
        tm = int(rwg_basis.t_minus[n])
        adj[tp].append((tm, n))
        adj[tm].append((tp, n))

    # ------------------------------------------------------------------
    # 2. BFS spanning tree from triangle 0
    # ------------------------------------------------------------------
    visited = np.zeros(N_t, dtype=bool)
    parent_tri = -np.ones(N_t, dtype=np.int32)
    parent_edge = -np.ones(N_t, dtype=np.int32)
    is_tree_edge = np.zeros(N, dtype=bool)

    root = 0
    visited[root] = True
    queue = deque([root])

    while queue:
        t = queue.popleft()
        for t_nb, edge_idx in adj[t]:
            if not visited[t_nb]:
                visited[t_nb] = True
                parent_tri[t_nb] = t
                parent_edge[t_nb] = edge_idx
                is_tree_edge[edge_idx] = True
                queue.append(t_nb)

    n_visited = int(visited.sum())
    if n_visited != N_t:
        raise ValueError(
            f"Dual mesh graph is disconnected: {n_visited}/{N_t} triangles "
            f"reachable from root.  The mesh may have isolated components."
        )

    tree_edges = np.where(is_tree_edge)[0]
    loop_edges = np.where(~is_tree_edge)[0]
    n_loops = len(loop_edges)
    n_stars = len(tree_edges)

    # ------------------------------------------------------------------
    # 3. Build sparse P matrix
    # ------------------------------------------------------------------
    # Use COO format for efficient construction, convert to CSC at the end.
    rows = []
    cols = []
    vals = []

    # Precompute ancestor lookup depth for LCA computation.
    # For each triangle, store the full path to root (depth-first ancestors).
    # This is O(N_t * tree_depth) but tree_depth is typically O(sqrt(N_t)).
    depth = np.zeros(N_t, dtype=np.int32)
    for t in range(N_t):
        if t == root:
            continue
        d = 0
        cur = t
        while cur != root:
            cur = parent_tri[cur]
            d += 1
        depth[t] = d

    # 3a. Loop columns: fundamental cycles with 1/l_e scaling
    t_plus = rwg_basis.t_plus
    t_minus = rwg_basis.t_minus
    edge_length = rwg_basis.edge_length

    for col_idx, c in enumerate(loop_edges):
        t_a = int(t_plus[c])
        t_b = int(t_minus[c])

        # Find LCA (lowest common ancestor) of t_a and t_b in the tree
        # by walking both paths upward until they meet.
        ancestors_a = set()
        cur = t_a
        path_a = [cur]
        while cur != root:
            cur = int(parent_tri[cur])
            path_a.append(cur)
            ancestors_a.add(cur)
        ancestors_a.add(t_a)

        # Walk from t_b upward until we hit an ancestor of t_a
        path_b = [t_b]
        cur = t_b
        while cur not in ancestors_a:
            cur = int(parent_tri[cur])
            path_b.append(cur)
        lca = cur

        # Trim path_a to stop at LCA
        lca_pos = path_a.index(lca)
        path_a = path_a[: lca_pos + 1]  # t_a -> ... -> lca
        # path_b already ends at lca: t_b -> ... -> lca

        # Edges along path from t_a to LCA (upward: child -> parent)
        for i in range(len(path_a) - 1):
            child = path_a[i]
            e = int(parent_edge[child])
            # Traversal direction: child -> parent
            # sigma = +1 if child = t_plus[e] (forward), -1 otherwise
            if child == t_plus[e]:
                sigma = 1.0
            else:
                sigma = -1.0
            rows.append(e)
            cols.append(col_idx)
            vals.append(sigma / edge_length[e])

        # Edges along path from LCA to t_b (downward: parent -> child)
        # path_b = [t_b, ..., lca], traverse in reverse: lca -> ... -> t_b
        for i in range(len(path_b) - 1, 0, -1):
            child = path_b[i - 1]
            e = int(parent_edge[child])
            parent = int(parent_tri[child])
            # Traversal direction: parent -> child
            # sigma = +1 if parent = t_plus[e] (forward), -1 otherwise
            if parent == t_plus[e]:
                sigma = 1.0
            else:
                sigma = -1.0
            rows.append(e)
            cols.append(col_idx)
            vals.append(sigma / edge_length[e])

        # Closing co-tree edge: t_b -> t_a
        # t_a = t_plus[c], t_b = t_minus[c]
        # Going from t_minus to t_plus => backward => sigma = -1
        rows.append(c)
        cols.append(col_idx)
        vals.append(-1.0 / edge_length[c])

    # 3b. Star columns: individual tree-edge RWG functions (unscaled)
    for i, t_edge in enumerate(tree_edges):
        rows.append(t_edge)
        cols.append(n_loops + i)
        vals.append(1.0)

    P = sparse.coo_matrix(
        (vals, (rows, cols)), shape=(N, N), dtype=np.float64
    ).tocsc()

    # Normalize columns so that cond(P) is O(1).  Loop columns have
    # entries ~1/l_e while star columns are unit vectors — without
    # normalization cond(P) scales as max(l)/min(l) × 1/l, amplifying
    # the condition number of the transformed system.
    col_norms = sparse.linalg.norm(P, axis=0)
    col_norms[col_norms < 1e-30] = 1.0  # safety
    P = P @ sparse.diags(1.0 / col_norms)

    return P, n_loops


def _compute_block_rescaling(
    Z_ls: np.ndarray,
    n_loops: int,
) -> np.ndarray:
    """Compute adaptive diagonal rescaling from actual block norms.

    Balances the loop-loop and star-star diagonal blocks of the
    loop-star impedance matrix so that they have comparable Frobenius
    norms, improving the condition number of the full system.

    Parameters
    ----------
    Z_ls : ndarray, shape (N, N), complex128
        Impedance matrix in the loop-star basis (P^T Z P).
    n_loops : int

    Returns
    -------
    D : ndarray, shape (N,)
    """
    N = Z_ls.shape[0]
    n_stars = N - n_loops

    norm_LL = np.linalg.norm(Z_ls[:n_loops, :n_loops]) / max(n_loops, 1)
    norm_SS = np.linalg.norm(Z_ls[n_loops:, n_loops:]) / max(n_stars, 1)

    D = np.ones(N, dtype=np.float64)
    if norm_LL > 1e-30 and norm_SS > 1e-30:
        # Scale so that D_L^2 * norm_LL = D_S^2 * norm_SS
        # Choose D_S = 1, D_L = sqrt(norm_SS / norm_LL)
        ratio = np.sqrt(norm_SS / norm_LL)
        D[:n_loops] = ratio
    return D


def solve_loop_star(
    Z_sys: np.ndarray,
    V_all: np.ndarray,
    P: sparse.spmatrix,
    n_loops: int,
    rescale: bool = True,
    return_loop_current: bool = False,
):
    """Solve the EFIE system in the loop-star basis.

    Transforms to the loop-star coordinate system where loop (divergence-
    free) and star (irrotational) sub-blocks decouple the vector- and
    scalar-potential contributions.  Optionally applies adaptive diagonal
    rescaling to balance the diagonal block norms.

    Parameters
    ----------
    Z_sys : ndarray, shape (N, N), complex128
        Impedance matrix in the RWG basis.
    V_all : ndarray, shape (N,) or (N, P), complex128
        RHS excitation vector(s).
    P : sparse matrix, shape (N, N)
        Loop-star transformation matrix.
    n_loops : int
        Number of loop basis functions.
    rescale : bool
        If True (default), apply adaptive diagonal rescaling computed from
        the actual Z_LL and Z_SS block norms.
    return_loop_current : bool
        If True, also return the loop-only current (divergence-free
        component) in the RWG basis.  This is useful for low-frequency
        inductance extraction where the full current is contaminated by
        the dominant scalar-potential (star) contribution.

    Returns
    -------
    I_rwg : ndarray, shape (N,) or (N, P), complex128
        Current coefficients in the original RWG basis.
    I_loop_rwg : ndarray, shape (N,) or (N, P), complex128
        Only returned when ``return_loop_current=True``.  Loop-only
        (divergence-free) current coefficients in the RWG basis —
        carries the vector-potential (inductive) response without
        scalar-potential contamination.
    """
    multi_rhs = V_all.ndim == 2

    # 1. Transform to loop-star basis
    Z_ls = np.asarray((P.T @ Z_sys) @ P)

    # 2. Adaptive rescaling from actual block norms
    if rescale:
        D = _compute_block_rescaling(Z_ls, n_loops)
        Z_solve = D[:, None] * Z_ls * D[None, :]
    else:
        D = None
        Z_solve = Z_ls

    # 3. Transform (and optionally rescale) RHS
    V_ls = np.asarray(P.T @ V_all)
    if D is not None:
        V_solve = D[:, None] * V_ls if multi_rhs else D * V_ls
    else:
        V_solve = V_ls

    # 4. Solve
    I_solve = np.linalg.solve(Z_solve, V_solve)

    # 5. Back-transform
    if D is not None:
        I_ls = D[:, None] * I_solve if multi_rhs else D * I_solve
    else:
        I_ls = I_solve

    I_rwg = np.asarray(P @ I_ls)

    if return_loop_current:
        # Extract loop-only coefficients (first n_loops entries of I_ls)
        # and transform back to RWG basis using only loop columns of P
        P_loop = P[:, :n_loops]
        if multi_rhs:
            I_loop_ls = I_ls[:n_loops, :]
        else:
            I_loop_ls = I_ls[:n_loops]
        I_loop_rwg = np.asarray(P_loop @ I_loop_ls)
        return I_rwg, I_loop_rwg

    return I_rwg


def solve_loop_star_hybrid(
    Z_sys: np.ndarray,
    V_all: np.ndarray,
    P: sparse.spmatrix,
    n_loops: int,
    Z_A: np.ndarray,
    rescale: bool = True,
) -> np.ndarray:
    """Solve the EFIE system using a hybrid loop-star decomposition.

    Uses the vector-potential-only matrix ``Z_A`` for all loop-related
    blocks (LL, LS, SL), eliminating scalar-potential quadrature leakage.
    Uses the full EFIE matrix ``Z_sys`` for the star-star block, where the
    scalar-potential contribution is physically correct and needed to drive
    current through open structures (e.g. spiral inductors with separated
    feed/return terminals).

    This is the recommended solve for inductor extraction on open structures
    at low kD.  For closed structures (rings, loops), ``solve_loop_only``
    is simpler and equally accurate.

    Parameters
    ----------
    Z_sys : ndarray, shape (N, N), complex128
        Full EFIE impedance matrix in the RWG basis.
    V_all : ndarray, shape (N,) or (N, P), complex128
        RHS excitation vector(s).
    P : sparse matrix, shape (N, N)
        Loop-star transformation matrix from ``build_loop_star_basis``.
    n_loops : int
        Number of loop basis functions.
    Z_A : ndarray, shape (N, N), complex128
        Vector-potential-only impedance matrix (no scalar-potential term).
    rescale : bool
        If True (default), apply adaptive diagonal rescaling.

    Returns
    -------
    I_rwg : ndarray, shape (N,) or (N, P), complex128
        Current coefficients in the original RWG basis.
    """
    multi_rhs = V_all.ndim == 2
    P_L = P[:, :n_loops]
    P_S = P[:, n_loops:]

    # Loop-related blocks from Z_A (no Phi contamination)
    Z_LL = np.asarray((P_L.T @ Z_A) @ P_L)
    Z_LS = np.asarray((P_L.T @ Z_A) @ P_S)
    Z_SL = np.asarray((P_S.T @ Z_A) @ P_L)
    # Star-star block from full Z (Phi contribution is correct here)
    Z_SS = np.asarray((P_S.T @ Z_sys) @ P_S)

    # Assemble full loop-star system
    n_S = P.shape[1] - n_loops
    Z_ls = np.empty((n_loops + n_S, n_loops + n_S), dtype=np.complex128)
    Z_ls[:n_loops, :n_loops] = Z_LL
    Z_ls[:n_loops, n_loops:] = Z_LS
    Z_ls[n_loops:, :n_loops] = Z_SL
    Z_ls[n_loops:, n_loops:] = Z_SS

    # Adaptive rescaling
    if rescale:
        D = _compute_block_rescaling(Z_ls, n_loops)
        Z_solve = D[:, None] * Z_ls * D[None, :]
    else:
        D = None
        Z_solve = Z_ls

    # Transform RHS
    V_L = np.asarray(P_L.T @ V_all)
    V_S = np.asarray(P_S.T @ V_all)
    if multi_rhs:
        V_ls = np.vstack([V_L, V_S])
    else:
        V_ls = np.concatenate([V_L, V_S])

    if D is not None:
        V_solve = D[:, None] * V_ls if multi_rhs else D * V_ls
    else:
        V_solve = V_ls

    # Solve
    I_solve = np.linalg.solve(Z_solve, V_solve)

    # Back-transform
    if D is not None:
        I_ls = D[:, None] * I_solve if multi_rhs else D * I_solve
    else:
        I_ls = I_solve

    I_rwg = np.asarray(P @ I_ls)
    return I_rwg


def solve_loop_only(
    Z_sys: np.ndarray,
    V_all: np.ndarray,
    P: sparse.spmatrix,
    n_loops: int,
    Z_A: np.ndarray = None,
) -> np.ndarray:
    """Solve only the loop sub-block for low-frequency inductance extraction.

    At low frequencies (kD << 1), the full EFIE impedance matrix mixes
    vector-potential (∝ ω) and scalar-potential (∝ 1/ω) contributions.
    Loop (divergence-free) basis functions eliminate the scalar-potential
    term *analytically*, but quadrature error in the assembled EFIE matrix
    leaks ~0.04% of Z_Phi into the loop-loop block.  At kD << 1 this
    leakage is amplified by (kD)^{-2}, contaminating Z_LL.

    When ``Z_A`` is provided (assembled separately with
    :class:`~pyMoM3d.mom.operators.vector_potential.VectorPotentialOperator`),
    this function uses Z_A instead of the full EFIE matrix for the loop
    sub-block.  Since Z_A contains no scalar-potential term, there is no
    leakage and the extracted inductance is frequency-independent below
    self-resonance.

    Parameters
    ----------
    Z_sys : ndarray, shape (N, N), complex128
        Full EFIE impedance matrix in the RWG basis (unused when ``Z_A``
        is provided, but kept for API compatibility).
    V_all : ndarray, shape (N,) or (N, P), complex128
        RHS excitation vector(s).
    P : sparse matrix, shape (N, N)
        Loop-star transformation matrix from ``build_loop_star_basis``.
    n_loops : int
        Number of loop basis functions.
    Z_A : ndarray, shape (N, N), complex128, optional
        Vector-potential-only impedance matrix (no scalar-potential term).
        When provided, the loop-loop sub-block is formed from Z_A instead
        of Z_sys, eliminating quadrature-error contamination from the
        scalar-potential term.  Assemble with
        ``VectorPotentialOperator`` and ``fill_matrix(..., backend='numpy')``.

    Returns
    -------
    I_loop_rwg : ndarray, shape (N,) or (N, P), complex128
        Loop-only current coefficients in the RWG basis.
    """
    P_loop = P[:, :n_loops]

    # Use Z_A (clean, no Phi leakage) when available; fall back to Z_sys
    Z_for_loops = Z_A if Z_A is not None else Z_sys

    # Transform to loop sub-block
    Z_LL = np.asarray((P_loop.T @ Z_for_loops) @ P_loop)
    V_L = np.asarray(P_loop.T @ V_all)

    # Solve the (n_loops × n_loops) system
    I_L = np.linalg.solve(Z_LL, V_L)

    # Back-transform to RWG basis
    I_loop_rwg = np.asarray(P_loop @ I_L)
    return I_loop_rwg


def verify_divergence_free(
    P: sparse.spmatrix,
    n_loops: int,
    rwg_basis,
    mesh,
    tol: float = 1e-10,
) -> bool:
    """Verify that loop columns of P are divergence-free.

    For each loop column, checks that the signed divergence sums to zero
    in every triangle.

    Parameters
    ----------
    P : sparse matrix, shape (N, N)
    n_loops : int
    rwg_basis : RWGBasis
    mesh : Mesh
    tol : float
        Tolerance for divergence residual.

    Returns
    -------
    ok : bool
        True if all loop functions are divergence-free to within tol.
    """
    N = rwg_basis.num_basis
    N_t = len(mesh.triangles)
    P_dense = P.toarray()

    for col in range(n_loops):
        # Compute divergence in each triangle
        div_per_tri = np.zeros(N_t, dtype=np.float64)
        for n in range(N):
            coeff = P_dense[n, col]
            if abs(coeff) < 1e-30:
                continue
            tp = rwg_basis.t_plus[n]
            tm = rwg_basis.t_minus[n]
            # div(f_n) in T+ = +l_n/(2A+), in T- = -l_n/(2A-)
            # div(coeff * f_n) in T+ = coeff * l_n / (2A+)
            l_n = rwg_basis.edge_length[n]
            div_per_tri[tp] += coeff * l_n / (2.0 * rwg_basis.area_plus[n])
            div_per_tri[tm] -= coeff * l_n / (2.0 * rwg_basis.area_minus[n])

        max_div = np.max(np.abs(div_per_tri))
        if max_div > tol:
            return False

    return True
