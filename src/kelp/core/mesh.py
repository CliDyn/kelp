"""Universal Mesh dataclass and spatial search utilities.

Uses pure NumPy + scipy (no JAX dependency).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import warnings

import numpy as np
from scipy.spatial import cKDTree

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def prange(*args, **kwargs):
        return range(*args)

from .geometry import delta_lon, DEG2RAD, RAD2DEG, R_EARTH_STANDARD


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_AREA_TOL = 1e-10       # tolerance for degenerate triangles
_BARY_EPS = -1e-6       # negative tolerance for barycentric containment


# ---------------------------------------------------------------------------
# Mesh dataclass
# ---------------------------------------------------------------------------

@dataclass
class Mesh:
    """Universal mesh container for any unstructured ocean model.

    Attributes
    ----------
    name : str
        Model identifier ("fesomc", "fesom2", etc.).
    nodes_lon, nodes_lat : np.ndarray
        Node coordinates in degrees, shape (N,).
    depth : np.ndarray
        Node depth in meters (positive down), shape (N,).
    elem_nodes : np.ndarray
        Element-to-node connectivity, shape (E, max_verts), 0-based.
        Padded with -1 for elements with fewer vertices.
        FESOM-C: max_verts=4 (tri: node3 == node0 **or** node3 == -1).
        FESOM2:  max_verts=3.
    elem_center_lon, elem_center_lat : np.ndarray
        Element center coordinates in degrees, shape (E,).
    node_to_elems : np.ndarray
        Node-to-element adjacency, shape (N, max_conn), -1 padded.
    node_to_elems_count : np.ndarray
        Number of elements per node, shape (N,).
    sigma : np.ndarray
        Vertical coordinate interfaces, shape (K,).
        For z-level meshes (FESOM2), these are physical depth interfaces
        in meters, positive-down.
        For sigma meshes (FESOM-C), these are native sigma interfaces.
        Empty (shape (0,)) for 2D.
    vertical_coord : str
        Vertical coordinate family: ``"zlevel"`` or ``"sigma"``.
    r_earth : float
        Earth radius in metres used for mesh-derived areas/metrics.  The
        reader sets the per-family value (FESOM2 ``6_367_500.0``, FESOM-C
        ``6_400_000.0``); defaults to ``R_EARTH_STANDARD`` for hand-built
        meshes.  Element areas enter the barycentric weights only as
        ratios, so this choice cancels there — it matters only for
        file-parity with a specific model build.
    metadata : dict
        Model-specific metadata (e.g. number of triangles/quads).
    """
    name: str
    nodes_lon: np.ndarray
    nodes_lat: np.ndarray
    depth: np.ndarray
    elem_nodes: np.ndarray
    elem_center_lon: np.ndarray
    elem_center_lat: np.ndarray
    node_to_elems: np.ndarray
    node_to_elems_count: np.ndarray
    sigma: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    vertical_coord: str = "zlevel"
    r_earth: float = R_EARTH_STANDARD
    #: Euler angles (deg) of the frame this mesh's coordinates live in;
    #: None = geographic.  Set by readers when mesh.rotation is configured.
    rotation_angles: Optional[Tuple[float, float, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Internal: populated by build_search()
    _node_tree: Optional[cKDTree] = field(default=None, repr=False)
    _tree_cartesian: bool = field(default=False, repr=False)  # True if 3D Cartesian tree

    # Internal: populated by build_grid_hash() — Numba-compatible spatial index
    _gh_offsets: Optional[np.ndarray] = field(default=None, repr=False)
    _gh_node_ids: Optional[np.ndarray] = field(default=None, repr=False)
    _gh_cell_size: float = field(default=0.0, repr=False)
    _gh_lon_min: float = field(default=0.0, repr=False)
    _gh_lat_min: float = field(default=0.0, repr=False)
    _gh_nx: int = field(default=0, repr=False)
    _gh_ny: int = field(default=0, repr=False)
    _gh_wrap_lon: bool = field(default=False, repr=False)
    _gh_far_rows: int = field(default=16, repr=False)   # exact-search row budget (build_grid_hash)
    _gh_max_edge_deg: Optional[float] = field(default=None, repr=False)   # longest element edge (deg), cached
    # Internal: populated by build_grid_hash() — unit-sphere Cartesian
    # coordinates of every node, (N, 3) float64 in node order, evaluated ONCE
    # with the exact scalar expressions the nearest scans used to evaluate
    # per candidate (a per-node table replaced four trig calls per candidate
    # per query, 2026-09-22; bit-identical anchors).  Rebuilt with the hash.
    _node_xyz: Optional[np.ndarray] = field(default=None, repr=False)
    # Internal: populated by apply_grid_hash_cell_rule() — the cell chooser's
    # report (candidates, choice, mode, reason); describe_grid_hash formats it.
    _gh_report: Optional[dict] = field(default=None, repr=False)

    # Internal: populated by grid_hash_dense_for_elements() — certificate
    # that every element-bbox cell has a nonempty 3x3 hash window, i.e. the
    # single-particle widened nearest (fused kernels) is bit-identical to
    # the batch 3x3+KD path for every containable query.  None = not
    # computed; _gh_dense_warned dedups the one-time fused-bary downgrade
    # warning.
    _gh_dense_elems: Optional[bool] = field(default=None, repr=False)
    _gh_dense_warned: bool = field(default=False, repr=False)

    # Internal: populated by build_boundary_info() — coastal element flags
    _elem_is_coastal: Optional[np.ndarray] = field(default=None, repr=False)

    # Internal: populated by build_elem_neighbors() — element adjacency
    # across shared edges, (E, max_verts) int32, -1 = boundary/unused slot.
    _elem_neighbors: Optional[np.ndarray] = field(default=None, repr=False)

    # Internal: populated by build_elem_areas() — element areas (m^2).
    # When the reader finds a model-provided area variable it pre-fills
    # this so build_elem_areas returns the file value (bit-parity, pole-
    # correct); otherwise the spherical-excess fallback computes it.
    _elem_areas: Optional[np.ndarray] = field(default=None, repr=False)

    # Internal: node control-volume areas (m^2), read from a FESOM-C
    # NetCDF when present (tri contribution A_e/3, quads true median-dual).
    # Stored for future use — nothing in kelp consumes them yet.
    _node_areas: Optional[np.ndarray] = field(default=None, repr=False)

    #: (cos, sin) of the local tangent-plane rotation per u/v carrier set
    #: ("element"/"node"); filled eagerly by the reader at load when the
    #: mesh is rotated (the factors need geographic carrier coords).
    _vec_rot_factors: Optional[dict] = field(default=None, repr=False)

    # Internal: populated lazily by numerics.vertical_uv.depth_points() —
    # the depth-linear u/v data-point table per placement ("level",
    # "layer_linear"); each entry holds STRONG references to the sigma,
    # depth and metadata["uv_data_last"] it was built from (validity is
    # checked by identity) and the DepthPoints.  Replaced, never mutated in
    # place, so a shallow copy of the mesh does not see the other's
    # rebuilds.  Never filled by a reader.
    _uv_depth_points: Optional[dict] = field(default=None, repr=False,
                                             compare=False)

    @property
    def n_nodes(self) -> int:
        return self.nodes_lon.shape[0]

    @property
    def n_elements(self) -> int:
        return self.elem_nodes.shape[0]

    @property
    def n_sigma(self) -> int:
        return self.sigma.shape[0]

    @property
    def z_levels(self) -> np.ndarray:
        """Z-level boundaries in meters (positive-down).

        Valid only for z-level meshes. Sigma meshes must use a sigma-aware
        column transform instead of treating ``mesh.sigma`` as physical depth.
        """
        if self.vertical_coord != "zlevel":
            raise NotImplementedError(
                "mesh.z_levels is only valid for z-level meshes; "
                "sigma meshes require sigma-aware vertical geometry"
            )
        return np.abs(self.sigma)

    def elem_is_triangle(self, e: int) -> bool:
        """Check if element *e* is a triangle (vs quadrilateral)."""
        nodes = self.elem_nodes[e]
        max_v = nodes.shape[0]
        if max_v == 3:
            return True
        # FESOM-C convention: quad if node3 != node0 and node3 != -1
        return nodes[3] == nodes[0] or nodes[3] == -1

    def elem_valid_nodes(self, e: int) -> np.ndarray:
        """Return valid (non-padded) node indices for element *e*."""
        nodes = self.elem_nodes[e]
        mask = nodes >= 0
        # Handle FESOM-C tri convention: node3 == node0
        if nodes.shape[0] == 4 and nodes[3] == nodes[0] and nodes[0] >= 0:
            mask[3] = False
        return nodes[mask]


# ---------------------------------------------------------------------------
# Spatial search: KD-tree
# ---------------------------------------------------------------------------

# Hard-coded switch: False = legacy 2D (lon,lat), True = 3D Cartesian.
# 3D Cartesian eliminates dateline discontinuity.  Set to False for
# comparison with Fortran reference output (validated at < 0.6 mm).
_USE_CARTESIAN_TREE = True


def _lonlat_to_xyz(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Convert lon/lat (degrees) to unit-sphere Cartesian (x, y, z).

    Returns array of shape (N, 3).
    """
    lon_r = np.asarray(lon, dtype=np.float64) * DEG2RAD
    lat_r = np.asarray(lat, dtype=np.float64) * DEG2RAD
    cos_lat = np.cos(lat_r)
    return np.column_stack((
        cos_lat * np.cos(lon_r),
        cos_lat * np.sin(lon_r),
        np.sin(lat_r),
    ))


def build_search(mesh: Mesh, leaf_size: int = 64) -> None:
    """Build spatial search structures for nearest-node queries.

    Builds the grid hash (Numba-compatible, preferred).  The KD-tree
    (scipy) is built lazily on first fallback access only — this saves
    2-3 seconds on large meshes (11M+ nodes).

    Parameters
    ----------
    mesh : Mesh
        The mesh to build search structures for.
    leaf_size : int
        KD-tree leaf size (stored for lazy construction).
    """
    mesh._kdtree_leaf_size = leaf_size

    # Build grid hash (fast, Numba-compatible)
    build_grid_hash(mesh)

    # KD-tree is built lazily by _ensure_kdtree() on first fallback access
    # (when Numba is unavailable or grid hash not built)


def _ensure_kdtree(mesh: Mesh) -> None:
    """Build KD-tree lazily on first access (fallback for non-Numba path)."""
    if mesh._node_tree is not None:
        return
    leaf_size = getattr(mesh, "_kdtree_leaf_size", 64)
    if _USE_CARTESIAN_TREE:
        pts = _lonlat_to_xyz(mesh.nodes_lon, mesh.nodes_lat)
        mesh._tree_cartesian = True
    else:
        pts = np.column_stack((mesh.nodes_lon, mesh.nodes_lat))
        mesh._tree_cartesian = False
    mesh._node_tree = cKDTree(pts, leafsize=leaf_size)


# ---------------------------------------------------------------------------
# Spatial search: Grid hash (Numba-compatible alternative to KD-tree)
# ---------------------------------------------------------------------------

#: Nodes in the fullest single hash cell above which the auto cell size is
#: reported as pathological (the in-kernel nearest search scans a 3×3
#: window, so the per-query cost is ~9× the occupancy).  Quasi-uniform
#: meshes sit at a handful of nodes per cell; only meshes whose
#: resolution spans orders of magnitude reach thousands.
GRID_HASH_WARN_OCCUPANCY = 2000

#: Target occupancy used when suggesting a cell size in that warning.
_GRID_HASH_TARGET_OCCUPANCY = 200


def set_grid_hash_cell(mesh: Mesh, cell_size: float) -> None:
    """Pin an explicit grid-hash cell size on ``mesh`` and rebuild the hash.

    The size is stored in ``mesh.metadata["grid_hash_cell"]``, which
    :func:`build_grid_hash` honours whenever it is called without an
    explicit ``cell_size`` — so a later :func:`build_search` by any other
    code (e.g. ``particles_gen._filter_to_mesh``) keeps the pinned cell
    instead of silently reverting to the auto size.

    Parameters
    ----------
    mesh : Mesh
        Mesh to pin the cell size on (rebuilt in place).
    cell_size : float
        Grid cell size in degrees; must lie in (0, 45).
    """
    if isinstance(cell_size, (bool, np.bool_)):
        raise ValueError(f"grid-hash cell_size must be a number, got {cell_size!r}")
    cs = float(cell_size)
    # build_grid_hash refuses a pinned cell outside (0, 45) on every later
    # rebuild; refuse it here, before the pin is stored.
    if not np.isfinite(cs) or not 0.0 < cs < 45.0:
        raise ValueError(
            "grid-hash cell_size must be a finite value in (0, 45) degrees, "
            f"got {cell_size!r}"
        )
    mesh.metadata["grid_hash_cell"] = cs
    mesh.metadata["grid_hash_cell_rule"] = "explicit"   # apply_grid_hash_cell_rule re-labels its own pin
    mesh.metadata.pop("grid_hash_cell_mode", None)
    build_grid_hash(mesh, cell_size=cs)


def set_exact_nearest(mesh: Mesh, flag: bool) -> None:
    """Switch the exact nearest-node anchor guard on/off for ``mesh``.

    Stored in ``mesh.metadata["exact_nearest"]`` (survives hash rebuilds).
    When on, every nearest-node query — the batch path and every fused
    kernel's evaluation site — accepts its 3x3 hash-window hit only when
    the certificate PROVES it is the true nearest (:func:`_anchor_provable`)
    and otherwise runs the in-kernel exact search
    (:func:`_grid_hash_nearest_exact`) — in the batch path and inside every
    fused kernel alike, so no stage anchor is left for a heal round.
    Off keeps the historical behaviour byte for byte.
    """
    mesh.metadata["exact_nearest"] = bool(flag)


#: mesh.grid_hash_cell_rule values
GRID_HASH_CELL_RULES = ("occupancy", "legacy")

#: scan-cost proxy weights (node distance evaluations = 1): a cell visit of
#: the widening (offsets read + loop), and one certificate evaluation
#: (`_anchor_provable_rect` + `_cap_extents`: arcsin/arccos/sin/cos) per
#: growth iteration.  Calibrated 2026-09-22 against the measured exact batch
#: search on CORE2 (200k seeds: the proxy's argmin must land on the measured
#: optimum 0.49-0.69 deg, not one ladder step below it) and on the 80-88 N
#: structured mesh of benchmarks/exact_nearest_cost.py (every anchor widens
#: there and the cost grows with 1/cell: the iteration term is what makes
#: the smaller cell lose, as it does in the measurement).
_GH_COST_CELL = 1.0
_GH_COST_CERT = 20.0


@njit(cache=True)
def _gh_rect_sum(cum, nx, ny, wrap_lon, ci, cj, rx, ry):
    """Nodes in the cell rectangle ``ci-rx..ci+rx`` x ``cj-ry..cj+ry`` from
    the 2-D prefix sum ``cum`` ((ny+1, nx+1), ``cum[j, i]`` = nodes in rows
    < j, columns < i); rows clipped to the grid, columns wrapped on a cyclic
    grid (one or two slices) and clipped otherwise."""
    j0 = cj - ry
    if j0 < 0:
        j0 = 0
    j1 = cj + ry + 1
    if j1 > ny:
        j1 = ny
    if wrap_lon and 2 * rx + 1 >= nx:
        return cum[j1, nx] - cum[j0, nx]
    i0 = ci - rx
    i1 = ci + rx + 1
    if wrap_lon:
        if i0 < 0:
            return ((cum[j1, i1] - cum[j0, i1]) + (cum[j1, nx] - cum[j0, nx])
                    - (cum[j1, i0 + nx] - cum[j0, i0 + nx]))
        if i1 > nx:
            return ((cum[j1, nx] - cum[j0, nx]) - (cum[j1, i0] - cum[j0, i0])
                    + (cum[j1, i1 - nx] - cum[j0, i1 - nx]))
    else:
        if i0 < 0:
            i0 = 0
        if i1 > nx:
            i1 = nx
    return (cum[j1, i1] - cum[j0, i1]) - (cum[j1, i0] - cum[j0, i0])


@njit(parallel=True, cache=True)
def _gh_sample_stats(lons, lats, nn, nodes_lon, nodes_lat, cum, cell_size,
                     lon_min, lat_min, nx, ny, wrap_lon, occ_out, cost_out, unprov_out,
                     approx_out):
    """Per sampled query at a candidate cell: the 3x3 window occupancy, the
    exact search's scan-cost proxy and the widening flag.

    ``unprov_out`` is the exact search's OWN verdict on the 3x3 window
    (:func:`_anchor_provable_rect` with ``rx = ry = 1``, the certificate the
    search runs on its first hit; a window that is empty or does not hold
    the true nearest node widens as well) — not the uniform-coverage bound
    ``theta <= cell``, which over-counts the seam cell of a wrap grid and
    the clipped edge rows (review 2026-09-22: 7.4 % vs 3.8 % real on CORE2).

    The proxy counts what the exact search touches: node distance
    evaluations plus cell visits.  A provable 3x3 hit costs its occupancy
    plus the 9 window cells.  Otherwise the search grows to the rectangle
    the certificate needs (half-widths ``rx = ceil(ext / cell)``, ``ry =
    ceil(theta / cell)``, every column when the cap reaches the pole; the
    growth from an empty window ends inside the same rectangle): its node
    count, plus the cells the growth visits — each new column pair is
    scanned cell by cell over the current rows (``rx * (2 ry + 1)``), each
    new row as a slice (``2 ry``); a full-width row costs its slice only.
    Counting the cells is what keeps a cell far below the node spacing from
    looking free (its rectangle holds few nodes but hundreds of cells).

    ``approx_out``: the TRUE nearest node lies outside the 3x3 window — the
    plain 3x3 scan (guard off) would anchor on another node.  Provability is
    only a sufficient condition for a correct 3x3 hit; this is the actual
    condition, the one the guard-off mode must bound."""
    for i in prange(lons.shape[0]):
        ci, cj = _gh_cell_of(lons[i], lats[i], cell_size, lon_min, lat_min, nx, ny, wrap_lon)
        occ = _gh_rect_sum(cum, nx, ny, wrap_lon, ci, cj, 1, 1)
        occ_out[i] = occ
        cin, cjn = _gh_cell_of(nodes_lon[nn[i]], nodes_lat[nn[i]], cell_size, lon_min, lat_min,
                               nx, ny, wrap_lon)
        di = cin - ci
        if di < 0:
            di = -di
        if wrap_lon and nx - di < di:
            di = nx - di
        dj = cjn - cj
        if dj < 0:
            dj = -dj
        approx_out[i] = 1 if (di > 1 or dj > 1) else 0
        theta, ext = _cap_extents(lons[i], lats[i], nodes_lon[nn[i]], nodes_lat[nn[i]])
        provable = occ > 0 and approx_out[i] == 0
        if provable:
            provable = _anchor_provable_rect(
                lons[i], lats[i], nn[i], 1, 1, cell_size, lon_min, lat_min, nx, ny,
                wrap_lon, nodes_lon, nodes_lat) == 0
        if provable:
            # one certificate evaluation, the 3x3 window
            cost_out[i] = occ + 9.0 * _GH_COST_CELL + _GH_COST_CERT
            unprov_out[i] = 0
        else:
            ry = int(np.ceil(theta / cell_size))
            if ry < 1:
                ry = 1
            if ry > ny:
                ry = ny
            if ext >= 1.0e29:
                rx = nx
            else:
                rx = int(np.ceil(ext / cell_size))
                if rx < 1:
                    rx = 1
            if rx * 2 + 1 >= nx:
                rx = nx
                cells = 9 + 2 * ry
                iters = ry
            else:
                cells = 9 + rx * (2 * ry + 1) + 2 * ry
                iters = rx if rx > ry else ry
            # the growth re-certifies once per iteration (rows and columns
            # grow together, so ~max(rx, ry) iterations), plus the first check
            cost_out[i] = (_gh_rect_sum(cum, nx, ny, wrap_lon, ci, cj, rx, ry)
                           + cells * _GH_COST_CELL + (iters + 1) * _GH_COST_CERT)
            unprov_out[i] = 1


def choose_grid_hash_cell(mesh: Mesh, *, exact: Optional[bool] = None,
                          sample: int = 200_000, seed: int = 0, min_median: int = 8,
                          approx_tol: float = 0.005, max_far_rows: int = 64,
                          steps: int = 12, min_gain: float = 2.75,
                          max_unprovable: float = 0.20, max_hash_mb: float = 128.0):
    """Measure a ladder of candidate cells on sampled element centres and pick
    one (docs/localization.md §3).

    ``exact`` (default: the mesh's ``exact_nearest`` flag) selects the mode:
    True -> the candidate with the smallest mean scan cost (an empty window
    only costs the widening, the anchor stays exact); False -> the smallest
    candidate with no empty window, median occupancy >= ``min_median`` and
    an approximate-anchor share (the true nearest node outside the 3x3
    window) <= the legacy cell's + ``approx_tol`` (in that mode the fused
    kernels take the 3x3 best: an empty or a too-small window makes the
    anchor approximate).  Candidates: the current cell divided by sqrt(2)^k, k = 0..steps,
    while ``_gh_far_rows_for`` <= ``max_far_rows`` and the grid fits
    ``_GRID_HASH_MAX_CELLS``.  In exact mode the choice leaves the legacy
    cell only when the proxy predicts at least ``min_gain`` (2.75×, the
    mid-gap between the two calibrated classes of 2026-09-22): on
    quasi-uniform meshes the proxy predicts 2.0–2.2× (HEALPix CORE2 twins
    2.03–2.06, CORE2 2.16), which buys ~1.6× on the search in isolation and
    nothing measurable on the step of a global release, while at high
    latitude the added widening measurably costs (HEALPix 80–88 °N +24 %,
    land-cap seeds ×3.5); structure-resolving meshes predict ≥ 3.6× (the
    North Sea mesh 3.6, the 11.5 M-node global mesh 4.9, the estuary ≫)
    and measure 1.6–3× on the search.  The proxy's own error is up to ~2×
    across meshes and hardware, so only a large predicted gain is trusted;
    a release confined to a refined region (CORE2 Arctic: −28 % per step at
    0.5°) still wants the explicit cell.  Exact mode also stays inside
    the regime the proxy was validated in: a candidate is eligible only
    while at most ``max_unprovable`` of the sampled anchors need a widened
    search (the proxy's certificate/rectangle terms dominate beyond that
    and their weights are hardware-dependent — on the 11.5 M-node global
    mesh the proxy predicted 5.9× where 1.4–3.1× was measured and the
    argmin was the cap-limited last candidate, review 2026-09-22) and while
    the hash offsets stay within ``max_hash_mb`` per process (4 bytes per
    cell: 128 MB = 32 M cells; that mesh's argmin needed 185 MB).  The
    legacy cell is always eligible.  Returns ``(cell, report)``; the
    hash cell is not changed (the hash is built if missing and the longest
    element edge cached).  Deterministic (fixed sample seed and ladder).
    Without Numba the chooser keeps the reader's cell: the exact batch
    search and the per-candidate statistics are compiled kernels, and
    their Python fallbacks would take minutes before the first step.
    """
    if mesh._gh_offsets is None:
        build_grid_hash(mesh)
    if exact is None:
        exact = bool((mesh.metadata or {}).get("exact_nearest", False))
    legacy = float(mesh._gh_cell_size)
    mode = "exact" if exact else "approx"

    def _keep(reason, sample_n=0):
        return legacy, {"rule": "occupancy", "mode": mode, "legacy": legacy,
                        "chosen": legacy, "candidates": [], "sample": int(sample_n),
                        "reason": reason}

    if not HAS_NUMBA:
        return _keep("numba unavailable: the reader's automatic cell")
    E = mesh.n_elements
    if E == 0:                                    # nothing to sample: keep the reader's cell
        return _keep("no elements to sample: the reader's automatic cell")
    if E > sample:
        pick = np.sort(np.random.default_rng(seed).choice(E, sample, replace=False))
    else:
        pick = np.arange(E)
    qlon = np.ascontiguousarray(mesh.elem_center_lon[pick], dtype=np.float64)
    qlat = np.ascontiguousarray(mesh.elem_center_lat[pick], dtype=np.float64)
    finite = np.isfinite(qlon) & np.isfinite(qlat)      # a fill-valued centre is no sample
    if not finite.all():
        qlon, qlat = qlon[finite], qlat[finite]
    if qlon.size == 0:
        return _keep("no finite element centre to sample: the reader's automatic cell")
    # the exact nearest node of every sample, whatever the guard flag says
    nn = np.empty(qlon.size, dtype=np.int32)
    _grid_hash_nearest_batch(
        qlon.size, qlon, qlat, mesh._gh_offsets, mesh._gh_node_ids,
        mesh._gh_cell_size, mesh._gh_lon_min, mesh._gh_lat_min, mesh._gh_nx,
        mesh._gh_ny, mesh._gh_wrap_lon, mesh.nodes_lon, mesh.nodes_lat,
        _require_node_xyz(mesh), nn, True, int(mesh._gh_far_rows))
    found = nn >= 0                                       # -1 only on an empty grid
    if not found.all():
        qlon, qlat, nn = qlon[found], qlat[found], nn[found]
    if qlon.size == 0:
        return _keep("no sample has a nearest node: the reader's automatic cell")
    lon_b, wrap = _gh_lon_branch(mesh.nodes_lon)
    lat = mesh.nodes_lat
    nodes_lon = np.ascontiguousarray(mesh.nodes_lon, dtype=np.float64)
    nodes_lat = np.ascontiguousarray(mesh.nodes_lat, dtype=np.float64)
    rows = []
    occ = np.empty(qlon.size, dtype=np.int64)
    cost = np.empty(qlon.size, dtype=np.float64)
    unprov = np.empty(qlon.size, dtype=np.int8)
    approx = np.empty(qlon.size, dtype=np.int8)
    for k in range(steps + 1):
        cell = legacy / (np.sqrt(2.0) ** k)
        far_rows = _gh_far_rows_for(mesh, cell)
        if far_rows > max_far_rows and k > 0:      # the legacy cell is always evaluated
            break
        lon_min, lat_min, nx, ny = _gh_grid_geometry(lon_b, lat, cell, wrap)
        if nx * ny > _GRID_HASH_MAX_CELLS and k > 0:   # the legacy grid always fits
            break
        idx = _gh_cell_index(lon_b, lat, cell, lon_min, lat_min, nx, ny, wrap)
        counts = np.bincount(idx, minlength=nx * ny).reshape(ny, nx)
        del idx
        # the prefix sum in place (no axis-0 temporary), int32 when the node
        # count allows: the transient per candidate is then counts (int64)
        # + cum (int32) instead of three full-grid int64 arrays (~1.2 GB at
        # the 50 M-cell cap — review 2026-09-22, 11.5 M-node global mesh)
        cdt = np.int32 if lat.size < 2 ** 31 - 1 else np.int64
        cum = np.zeros((ny + 1, nx + 1), dtype=cdt)
        np.cumsum(counts, axis=0, dtype=cdt, out=cum[1:, 1:])
        del counts
        np.cumsum(cum[1:, 1:], axis=1, dtype=cdt, out=cum[1:, 1:])
        _gh_sample_stats(qlon, qlat, nn, nodes_lon, nodes_lat, cum, float(cell),
                         float(lon_min), float(lat_min), int(nx), int(ny), bool(wrap),
                         occ, cost, unprov, approx)
        del cum
        unprov_share = float(np.mean(unprov))
        rows.append({"cell": float(cell), "far_rows": int(far_rows), "n_cells": int(nx * ny),
                     "hash_mb": float(4.0 * (nx * ny + 1) / 1e6),
                     "median": float(np.median(occ)), "p95": float(np.percentile(occ, 95)),
                     "empty": float(np.mean(occ == 0)), "unprovable": unprov_share,
                     "approx": float(np.mean(approx)), "mean_cost": float(np.mean(cost)),
                     "eligible": bool(k == 0 or (unprov_share <= max_unprovable
                                                 and 4.0 * (nx * ny + 1) / 1e6 <= max_hash_mb))})
    base = rows[0]
    if exact:
        eligible = [r for r in rows if r["eligible"]]
        best = min(eligible, key=lambda r: (r["mean_cost"], -r["cell"]))
        gain = base["mean_cost"] / best["mean_cost"] if best["mean_cost"] > 0 else 1.0
        if best is not base and gain < min_gain:
            reason = (f"keep the legacy cell: the cheapest eligible candidate {best['cell']:.4g} "
                      f"deg predicts only {gain:.2f}x (< {min_gain:g}x) on the scan cost "
                      f"({best['mean_cost']:.1f} vs {base['mean_cost']:.1f} nodes/query)")
            best = base
        else:
            reason = (f"smallest mean scan cost {best['mean_cost']:.1f} nodes/query "
                      f"(legacy {base['mean_cost']:.1f}, {gain:.2f}x) among the candidates "
                      f"with widening <= {100 * max_unprovable:.0f} % and hash <= "
                      f"{max_hash_mb:g} MB")
    else:
        ok = [r for r in rows if r["empty"] == 0.0 and r["median"] >= min_median
              and r["approx"] <= base["approx"] + approx_tol
              and (r is base or r["hash_mb"] <= max_hash_mb)]
        best = min(ok, key=lambda r: r["cell"]) if ok else base
        reason = (f"smallest cell with no empty window, median >= {min_median} and no added "
                  f"approximate anchor" if ok else
                  "no smaller cell keeps every window non-empty without approximate anchors")
    report = {"rule": "occupancy", "mode": mode, "legacy": legacy,
              "chosen": float(best["cell"]), "reason": reason, "candidates": rows,
              "sample": int(qlon.size)}
    return float(best["cell"]), report


def apply_grid_hash_cell_rule(mesh: Mesh, rule: str = "occupancy", **chooser_kwargs) -> dict:
    """Apply ``mesh.grid_hash_cell_rule`` and return the report (also kept
    as ``mesh._gh_report``).

    ``legacy`` keeps the reader's automatic cell (stamped, no pin);
    ``occupancy`` runs :func:`choose_grid_hash_cell` in the mode of the
    mesh's ``exact_nearest`` flag (or ``exact=``) and pins a different
    choice through :func:`set_grid_hash_cell`, so every later rebuild
    keeps it.  Stamps ``metadata["grid_hash_cell_rule"]`` (``occupancy`` /
    ``legacy`` / ``explicit``) and, for ``occupancy``, ``grid_hash_cell_mode``
    (``exact`` / ``approx``).  On a reused mesh object:

    - an EXPLICIT pin (:func:`set_grid_hash_cell`, ``grid_hash_cell_size``)
      is never overridden — reported as ``explicit``;
    - the same rule in the same mode returns the earlier report without
      re-running the chooser (a driver looping ``track_particles`` pays
      the ladder once);
    - a different rule or mode after an earlier ``occupancy`` apply drops
      that pin, rebuilds the reader's cell and decides afresh — the strict
      (guard-off) rule's guarantees do not carry over from a cell chosen in
      exact mode, and ``legacy`` after ``occupancy`` really restores the
      reader's cell (review 2026-09-22).
    """
    rule = str(rule).lower()
    if rule not in GRID_HASH_CELL_RULES:
        raise ValueError(
            f"grid_hash_cell_rule must be one of {GRID_HASH_CELL_RULES}, got {rule!r}")
    md = mesh.metadata
    if mesh._gh_offsets is None:
        build_grid_hash(mesh)
    exact = chooser_kwargs.get("exact")
    if exact is None:
        exact = bool(md.get("exact_nearest", False))
    mode = "exact" if exact else "approx"
    earlier = md.get("grid_hash_cell_rule")
    if "grid_hash_cell" in md and earlier in (None, "explicit"):
        md["grid_hash_cell_rule"] = "explicit"
        report = {"rule": "explicit", "chosen": float(mesh._gh_cell_size), "legacy": None,
                  "mode": None, "reason": "explicit cell (grid_hash_cell_size)",
                  "candidates": [], "sample": 0}
        mesh._gh_report = report
        return report
    same_mode = rule == "legacy" or md.get("grid_hash_cell_mode") == mode
    if (earlier == rule and same_mode and mesh._gh_report is not None
            and mesh._gh_report.get("chosen") == float(mesh._gh_cell_size)):
        return mesh._gh_report                       # the same decision, already taken
    if earlier is not None:                          # an earlier rule's decision: start over
        md.pop("grid_hash_cell", None)
        md.pop("grid_hash_cell_rule", None)
        md.pop("grid_hash_cell_mode", None)
        build_grid_hash(mesh)                        # the reader's automatic cell
    if rule == "legacy":
        report = {"rule": "legacy", "chosen": float(mesh._gh_cell_size),
                  "legacy": float(mesh._gh_cell_size), "mode": None,
                  "reason": "legacy rule: the reader's automatic cell", "candidates": [],
                  "sample": 0}
    else:
        cell, report = choose_grid_hash_cell(mesh, **chooser_kwargs)
        if cell != mesh._gh_cell_size:
            set_grid_hash_cell(mesh, cell)           # stamps explicit; re-labelled below
        md["grid_hash_cell_mode"] = mode
    md["grid_hash_cell_rule"] = rule
    mesh._gh_report = report
    return report


def describe_grid_hash(mesh: Mesh) -> str:
    """One line for the run log / ``kelp info``: the cell, the rule, and the
    window statistics of the choice next to the legacy cell's."""
    r = mesh._gh_report
    cell = float(mesh._gh_cell_size)
    if r is not None and r.get("chosen") != cell:      # a rebuild since: the report is stale
        r = None
    if r is None or not r.get("candidates"):
        rule = (r or {}).get("rule") or mesh.metadata.get("grid_hash_cell_rule", "reader auto")
        return f"grid hash: cell {cell:.4g} deg ({rule}), far_rows {int(mesh._gh_far_rows)}"
    rows = {c["cell"]: c for c in r["candidates"]}
    base, ch = r["candidates"][0], rows[r["chosen"]]
    return (f"grid hash: cell {ch['cell']:.4g} deg by occupancy/{r['mode']} "
            f"(legacy {base['cell']:.4g}): window median {ch['median']:.0f} / p95 "
            f"{ch['p95']:.0f} nodes (legacy {base['median']:.0f} / {base['p95']:.0f}), "
            f"empty {100 * ch['empty']:.1f} %, widening {100 * ch['unprovable']:.1f} % "
            f"(legacy {100 * base['unprovable']:.1f} %), scan cost {ch['mean_cost']:.0f} vs "
            f"{base['mean_cost']:.0f} nodes/query, far_rows {ch['far_rows']}, "
            f"{r['sample']} centres — {r['reason']}")


#: Hard cap on the number of hash cells: the cell index is int32 and
#: the CSR offsets array is n_cells+1 int32 (50M cells = 200 MB), so a
#: mistakenly tiny cell size must fail loudly, not allocate silently.
_GRID_HASH_MAX_CELLS = 50_000_000


@njit(parallel=True, cache=True)
def _node_xyz_fill(nodes_lon: np.ndarray, nodes_lat: np.ndarray,
                   out: np.ndarray) -> None:
    """Unit-sphere Cartesian coordinates of every node into ``out`` (N, 3).

    The VERBATIM scalar expressions the nearest scans evaluated per
    candidate before 2026-09-22, evaluated by Numba — never by numpy or
    ``math``: their cos/sin round differently in ~0.1-0.25 % of values
    (LLVM lowers a sin/cos pair of one argument that sits in one basic
    block to a single ``sincos`` call, and other libraries round the pair
    differently), and one ulp flips a near-tie anchor.  The precondition
    for bit-identity with the scans is therefore that BOTH code shapes are
    lowered to the same math: keep each cos/sin pair here in one basic
    block (no branch between them) and run without a vector math library
    (``numba.config.USING_SVML`` False — Intel SVML would vectorise this
    contiguous loop but not the gather/argmin scans).  ``build_grid_hash``
    verifies it on a node sample (``_node_xyz_selfcheck``) and warns.
    Each node is independent, so ``prange`` keeps the result deterministic.
    """
    for n in prange(nodes_lon.shape[0]):
        lat_n_r = nodes_lat[n] * DEG2RAD
        lon_n_r = nodes_lon[n] * DEG2RAD
        cos_lat_n = np.cos(lat_n_r)
        out[n, 0] = cos_lat_n * np.cos(lon_n_r)
        out[n, 1] = cos_lat_n * np.sin(lon_n_r)
        out[n, 2] = np.sin(lat_n_r)


@njit(cache=True)
def _node_xyz_selfcheck(sample_ids: np.ndarray, nodes_lon: np.ndarray,
                        nodes_lat: np.ndarray, node_xyz: np.ndarray,
                        lon_p: float, lat_p: float):
    """``(mismatches, best)``: how many sampled nodes get DIFFERENT
    coordinates from the old per-candidate evaluation than the table holds.

    The loop is deliberately shaped like the pre-table scans — a gather
    through ``sample_ids`` (the CSR node order) with the same six
    expressions inside an argmin against a query point — so it is lowered
    the way the scans were, not the way the contiguous fill loop is.  A
    nonzero count means the two shapes were compiled to different math
    (a vector math library on the fill loop): the search stays exact and
    self-consistent, but anchors at exact ties may differ from
    pre-2026-09-22 runs; ``build_grid_hash`` warns.  ``best`` is returned
    only so the argmin is not dead code.
    """
    lat_r = lat_p * DEG2RAD
    lon_r = lon_p * DEG2RAD
    cos_lat_p = np.cos(lat_r)
    xp = cos_lat_p * np.cos(lon_r)
    yp = cos_lat_p * np.sin(lon_r)
    zp = np.sin(lat_r)
    best = -1
    best_d2 = 1e30
    mismatches = 0
    for k in range(sample_ids.shape[0]):
        n = sample_ids[k]
        lat_n_r = nodes_lat[n] * DEG2RAD
        lon_n_r = nodes_lon[n] * DEG2RAD
        cos_lat_n = np.cos(lat_n_r)
        xn = cos_lat_n * np.cos(lon_n_r)
        yn = cos_lat_n * np.sin(lon_n_r)
        zn = np.sin(lat_n_r)
        if xn != node_xyz[n, 0] or yn != node_xyz[n, 1] or zn != node_xyz[n, 2]:
            mismatches += 1
        d2 = (xn - xp) ** 2 + (yn - yp) ** 2 + (zn - zp) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best = n
    return mismatches, best


#: nodes sampled (in CSR order) by the build-time table self-check
_NODE_XYZ_SELFCHECK_SAMPLE = 4096


def _require_node_xyz(mesh: Mesh) -> np.ndarray:
    """The node coordinate table, or a clear error when the hash exists
    without it — reachable only by clearing the private field or through a
    ``Mesh`` object created before the table existed (2026-09-22)."""
    xyz = mesh._node_xyz
    if xyz is None:
        raise RuntimeError(
            "grid hash without its node coordinate table — call "
            "build_search(mesh) (or build_grid_hash) first")
    return xyz


def _gh_lon_branch(nodes_lon: np.ndarray):
    """``(lon, wrap)``: the node longitudes in whichever branch — [0, 360) or
    [-180, 180) — keeps the mesh contiguous, and the cyclic-grid flag (a
    near-global mesh, span > 350 deg in both branches).  The build's
    arithmetic, verbatim; the cell chooser reuses it."""
    lon_0360 = (nodes_lon + 360.0) % 360.0
    lon_signed = (nodes_lon + 180.0) % 360.0 - 180.0
    span_0360 = lon_0360.max() - lon_0360.min()
    span_signed = lon_signed.max() - lon_signed.min()
    wrap = bool(min(span_0360, span_signed) > 350.0)
    if wrap or span_0360 <= span_signed:
        return lon_0360, wrap
    return lon_signed, wrap


def _gh_grid_geometry(lon: np.ndarray, lat: np.ndarray, cell_size: float, wrap: bool):
    """``(lon_min, lat_min, nx, ny)`` of the hash grid for ``cell_size`` —
    one-cell padding in latitude (and in longitude on a non-cyclic grid);
    a cyclic grid covers exactly [0, 360) so the query-side index wrap
    ``ci2 +- nx`` is an exact +-360 deg shift."""
    lat_min = lat.min() - cell_size
    lat_max = lat.max() + cell_size
    ny = int(np.ceil((lat_max - lat_min) / cell_size))
    if wrap:
        lon_min = 0.0
        nx = int(np.ceil(360.0 / cell_size))
    else:
        lon_min = lon.min() - cell_size
        lon_max = lon.max() + cell_size
        nx = int(np.ceil((lon_max - lon_min) / cell_size))
    return lon_min, lat_min, nx, ny


def _gh_cell_index(lon: np.ndarray, lat: np.ndarray, cell_size: float,
                   lon_min: float, lat_min: float, nx: int, ny: int,
                   wrap: bool) -> np.ndarray:
    """Flat cell index ``cj * nx + ci`` of every node (int32)."""
    if wrap:
        ci = np.minimum((lon / cell_size).astype(np.int32), nx - 1)
    else:
        ci = np.clip(((lon - lon_min) / cell_size).astype(np.int32), 0, nx - 1)
    cj = np.clip(((lat - lat_min) / cell_size).astype(np.int32), 0, ny - 1)
    return cj * nx + ci


def build_grid_hash(mesh: Mesh, cell_size: float = 0.0) -> None:
    """Build a uniform grid hash for Numba-compatible nearest-node queries.

    The grid hash provides O(1) nearest-node lookups that can be called
    from inside ``@njit`` kernels (unlike scipy's KD-tree).

    Longitudes are indexed in whichever branch — [0, 360) or [-180, 180) —
    keeps the mesh contiguous, so regional meshes crossing the prime
    meridian or the dateline both work.  Near-global meshes use a cyclic
    grid spanning exactly 360° with exact index wrap at the seam.

    Parameters
    ----------
    mesh : Mesh
        The mesh to build grid hash for.
    cell_size : float
        Grid cell size in degrees.  If 0 (default), the pinned
        ``mesh.metadata["grid_hash_cell"]`` is used when present (see
        :func:`set_grid_hash_cell`), else it is auto-determined from mesh
        density: ``sqrt(domain_area / n_nodes) * 2``, floored at 0.05°.
        The auto size assumes a quasi-uniform mesh; on a mesh whose
        resolution spans orders of magnitude the floor can leave
        thousands of nodes in one cell, which degrades the in-kernel 3×3
        nearest scan to a linear search — that case is reported as a
        ``RuntimeWarning`` naming ``mesh.grid_hash_cell_size``.
    """
    # Pick the longitude branch in which the mesh is contiguous.  A fixed
    # branch cut fragments any mesh that crosses it ([0, 360) breaks
    # prime-meridian meshes, [-180, 180) breaks dateline meshes) — the
    # branch with the smaller span is the unfragmented one.  Near-global
    # meshes (span > 350° in both branches) get a cyclic grid instead.
    lon, wrap = _gh_lon_branch(mesh.nodes_lon)
    lon_span = lon.max() - lon.min()
    lat = mesh.nodes_lat
    N = len(lon)

    explicit = cell_size > 0
    if not explicit:
        # A mesh can pin its own cell (set_grid_hash_cell / config
        # mesh.grid_hash_cell_size) so ANY later rebuild keeps it.
        md = mesh.metadata or {}
        if "grid_hash_cell" in md:
            # A PRESENT key must be valid: silently falling back to the
            # auto size would hide the very pathology it was set to
            # avoid.  Bounded because a stray tiny value allocates a
            # gigantic grid and overflows the int32 cell index.
            raw_pin = md["grid_hash_cell"]
            try:
                pinned = float(raw_pin)
            except (TypeError, ValueError):
                pinned = float("nan")
            if not np.isfinite(pinned) or not (0.0 < pinned < 45.0):
                raise ValueError(
                    "mesh.metadata['grid_hash_cell'] must be a finite cell "
                    f"size in (0, 45) degrees, got {raw_pin!r}")
            cell_size = pinned
            explicit = True
        else:
            lat_span = lat.max() - lat.min()
            area = max(lon_span * lat_span, 1.0)
            cell_size = max(0.05, np.sqrt(area / N) * 2)

    # Grid geometry: one-cell padding in latitude; a cyclic grid covers
    # exactly [0, 360) (with padding the grid would span 360 deg + 2 cells
    # and the wrapped window would miss the cells near the seam).
    lon_min, lat_min, nx, ny = _gh_grid_geometry(lon, lat, cell_size, wrap)
    n_cells = nx * ny
    if n_cells > _GRID_HASH_MAX_CELLS:
        raise ValueError(
            f"grid hash: cell size {cell_size:.6g} deg over this mesh's "
            f"extent needs {n_cells} cells (> {_GRID_HASH_MAX_CELLS}); the "
            "index would not fit int32 and the offsets array would be "
            "GB-sized. Use a larger cell.")

    # Assign each node to a cell
    cell_idx = _gh_cell_index(lon, lat, cell_size, lon_min, lat_min, nx, ny, wrap)

    # Count nodes per cell
    counts = np.zeros(n_cells, dtype=np.int32)
    for c in cell_idx:
        counts[c] += 1

    # Build CSR offsets
    offsets = np.zeros(n_cells + 1, dtype=np.int32)
    np.cumsum(counts, out=offsets[1:])

    # Sort nodes into cells
    node_ids = np.empty(N, dtype=np.int32)
    fill = offsets[:-1].copy()
    for i in range(N):
        c = cell_idx[i]
        node_ids[fill[c]] = i
        fill[c] += 1

    # Pathological-occupancy check (free: `counts` is already computed).
    # The nearest-node kernels scan a 3×3 window, so a cell holding
    # thousands of nodes turns an O(1) lookup into a linear search —
    # ~ms per particle per evaluation on structure-resolving meshes.
    if not explicit:
        max_occ = int(counts.max()) if n_cells else 0
        if max_occ > GRID_HASH_WARN_OCCUPANCY:
            suggested = cell_size * np.sqrt(
                _GRID_HASH_TARGET_OCCUPANCY / max_occ)
            warnings.warn(
                f"grid hash: auto cell size {cell_size:.4g} deg leaves up to "
                f"{max_occ} nodes in one cell on this highly non-uniform mesh "
                f"— the in-kernel 3x3 nearest scan degrades to a linear "
                f"search (measurably ~ms per particle per step). Set "
                f"mesh.grid_hash_cell_size (try ~{suggested:.3g}) and validate "
                f"the anchor trade-off on a short run.",
                RuntimeWarning, stacklevel=2)

    # Node coordinate table for the scans (raw lon/lat — the kernels read
    # mesh.nodes_lon, never the branch-folded `lon` used for cell indexing:
    # cos(lon + 360) != cos(lon) in floating point).  The old table is
    # released first so a rebuild never holds two (N, 3) tables at once.
    mesh._node_xyz = None
    lon_raw = np.ascontiguousarray(mesh.nodes_lon, dtype=np.float64)
    lat_raw = np.ascontiguousarray(mesh.nodes_lat, dtype=np.float64)
    xyz = np.empty((N, 3), dtype=np.float64)
    if HAS_NUMBA:
        _node_xyz_fill(lon_raw, lat_raw, xyz)
        if N > 0:
            # Bit-identity precondition check (see _node_xyz_fill): the
            # table must equal what the scans' own code shape evaluates.
            stride = max(1, N // _NODE_XYZ_SELFCHECK_SAMPLE)
            sample = np.ascontiguousarray(node_ids[::stride])
            mismatches, _ = _node_xyz_selfcheck(
                sample, lon_raw, lat_raw, xyz, float(lon_raw[0]), float(lat_raw[0]))
            if mismatches:
                warnings.warn(
                    f"grid hash: the node coordinate table differs from the "
                    f"scans' own evaluation at {mismatches} of {sample.size} "
                    f"sampled nodes — the fill loop and the scans were compiled "
                    f"to different math (a vector math library such as Intel "
                    f"SVML? numba.config.USING_SVML). The search stays exact, "
                    f"but anchors at exact ties may differ from pre-2026-09-22 "
                    f"runs; set NUMBA_DISABLE_INTEL_SVML=1 for bit-identity.",
                    RuntimeWarning, stacklevel=2)
    else:
        # No Numba: the KD-tree serves the queries (its own numpy
        # conversion); the table follows that convention.  Bit-identity
        # with the Numba kernels is not a contract of this mode.
        xyz[:] = _lonlat_to_xyz(lon_raw, lat_raw)
    mesh._node_xyz = xyz

    mesh._gh_offsets = offsets
    mesh._gh_node_ids = node_ids
    mesh._gh_cell_size = cell_size
    mesh._gh_lon_min = lon_min
    mesh._gh_lat_min = lat_min
    mesh._gh_nx = nx
    mesh._gh_ny = ny
    mesh._gh_wrap_lon = wrap
    mesh._gh_far_rows = _gh_far_rows_for(mesh, cell_size)
    mesh._gh_dense_elems = None      # certificate invalidated with the hash
    if mesh._gh_report is not None and mesh._gh_report.get("chosen") != cell_size:
        mesh._gh_report = None       # the chooser's report described another cell


#: floor / cap of the exact-search row budget (cells)
_GH_FAR_ROWS_MIN = 4
_GH_FAR_ROWS_MAX = 4096


def _gh_max_edge_for(mesh: Mesh) -> float:
    """Longest element edge in degrees of great circle (seam-safe via
    ``delta_lon``): every consecutive slot pair plus the closing edge
    (last valid slot -> slot 0, the third edge of a -1-padded triangle).
    Chunked over elements (1 M per pass) to bound the transient memory on
    the 23 M-element global mesh; cell-independent, so ``build_grid_hash``
    caches it on ``mesh._gh_max_edge_deg`` (review round 4)."""
    en = mesh.elem_nodes
    if en is None or en.size == 0:
        return 0.0
    lon = np.asarray(mesh.nodes_lon, dtype=np.float64)
    lat = np.asarray(mesh.nodes_lat, dtype=np.float64)
    nv = en.shape[1]
    max_edge = 0.0
    step = 1_000_000
    for e0 in range(0, en.shape[0], step):
        blk = en[e0:e0 + step]
        last_valid = np.clip((blk >= 0).sum(axis=1) - 1, 0, nv - 1)
        pairs = [(blk[:, k], blk[:, (k + 1) % nv]) for k in range(nv)]
        pairs.append((blk[np.arange(blk.shape[0]), last_valid], blk[:, 0]))
        for a, b in pairs:
            ok = (a >= 0) & (b >= 0)
            if not ok.any():
                continue
            a = a[ok]; b = b[ok]
            dl = delta_lon(lon[b], lon[a]) * DEG2RAD
            la1 = lat[a] * DEG2RAD; la2 = lat[b] * DEG2RAD
            cosd = np.sin(la1) * np.sin(la2) + np.cos(la1) * np.cos(la2) * np.cos(dl)
            d = np.arccos(np.clip(cosd, -1.0, 1.0)) * RAD2DEG
            max_edge = max(max_edge, float(d.max()))
    return max_edge


def _gh_far_rows_for(mesh: Mesh, cell_size: float) -> int:
    """Row budget of :func:`_grid_hash_nearest_exact` for this mesh and cell:
    the longest element edge (cached, :func:`_gh_max_edge_for`) in cells,
    plus two.  A point inside an element is never farther from its
    nearest vertex than that, so a query whose nearest node lies beyond
    the budget is outside every element (review round 3: a fixed
    budget broke the guard on a multi-resolution mesh whose coarse
    elements span 50 cells of the fine patch's pinned cell).  Floored at
    ``_GH_FAR_ROWS_MIN``; capped at ``_GH_FAR_ROWS_MAX`` with a warning,
    because a cell that small for the mesh voids the guarantee."""
    if mesh._gh_max_edge_deg is None:
        mesh._gh_max_edge_deg = _gh_max_edge_for(mesh)
    # -1e-9: an edge that is an exact multiple of the cell (a 3-deg edge on a
    # 0.5-deg cell) must not round up through arccos noise; the +2 covers it
    rows = int(np.ceil(mesh._gh_max_edge_deg / cell_size - 1.0e-9)) + 2
    if rows > _GH_FAR_ROWS_MAX:
        warnings.warn(
            f"grid hash: the longest element edge ({mesh._gh_max_edge_deg:.3g} deg) spans "
            f"{rows - 2} cells of {cell_size:.3g} deg — the exact-search budget (far_rows) is "
            f"capped at {_GH_FAR_ROWS_MAX}, so mesh.exact_nearest cannot guarantee the anchor of "
            "a point deep inside such an element; use a larger cell.", RuntimeWarning)
        rows = _GH_FAR_ROWS_MAX
    if rows < _GH_FAR_ROWS_MIN:
        rows = _GH_FAR_ROWS_MIN
    return int(rows)


@njit(cache=True)
def _gh_elements_covered(elem_nodes, nodes_lon, nodes_lat, covered_flat,
                         cell_size, lon_min, lat_min, nx, ny, wrap_lon):
    """Return True iff every cell of every element's bbox is 'covered'
    (has a nonempty 3x3 hash window).  Bbox raster is over-conservative
    for cells the element does not actually reach — safe direction (may
    deny the certificate, never grants it wrongly).  Dateline-safe: the
    lon extent is accumulated as wrapped deltas from the first vertex, so
    a seam-straddling element rasters its own few cells, not the globe.

    SOUNDNESS INVARIANT (pinned by test_fused_bary_tinterp.py): this
    raster uses the SAME first-valid-vertex wrapped-delta bbox as the
    `_element_contains` prefilter, so every point that prefilter can
    accept maps (via `_gh_cell_of`, monotone + clamped) into the
    rasterized cell range.  Re-anchoring either bbox (e.g. on the
    centroid) without the other breaks "contained => covered cell" and
    silently resurrects the approximate-anchor blocker the certificate
    exists to prevent (review 2026-08-16)."""
    E, max_v = elem_nodes.shape
    for e in range(E):
        first = -1
        lon0 = 0.0
        lo_min = 0.0; lo_max = 0.0
        la_min = 0.0; la_max = 0.0
        for k in range(max_v):
            nd = elem_nodes[e, k]
            if nd < 0:
                continue
            la = nodes_lat[nd]
            if first < 0:
                first = nd
                lon0 = nodes_lon[nd]
                dl = 0.0
                la_min = la; la_max = la
            else:
                dl = ((nodes_lon[nd] - lon0 + 180.0) % 360.0) - 180.0
                if la < la_min: la_min = la
                if la > la_max: la_max = la
            if dl < lo_min: lo_min = dl
            if dl > lo_max: lo_max = dl
        if first < 0:
            continue
        ci_w, cj_s = _gh_cell_of(lon0 + lo_min, la_min, cell_size,
                                 lon_min, lat_min, nx, ny, wrap_lon)
        ci_e, cj_n = _gh_cell_of(lon0 + lo_max, la_max, cell_size,
                                 lon_min, lat_min, nx, ny, wrap_lon)
        n_i = ci_e - ci_w
        if n_i < 0:
            if wrap_lon:
                n_i = n_i + nx
            else:
                # A reversed range on a non-wrap grid means the element
                # straddles the grid's branch cut — cannot happen for a
                # contiguously-built hash, but if it ever does, DENY
                # rather than raster a single wrong column (the only
                # failure direction that could grant on bad evidence).
                return False
        for jj in range(cj_s, cj_n + 1):
            row = jj * nx
            for t in range(n_i + 1):
                ii = ci_w + t
                if ii >= nx:
                    ii -= nx
                if not covered_flat[row + ii]:
                    return False
    return True


def grid_hash_dense_for_elements(mesh: Mesh) -> bool:
    """Certificate: the grid hash is dense over every element's footprint.

    True ⇔ every hash cell touched by an element bbox has a NONEMPTY 3×3
    window.  Under it, `_grid_hash_nearest_widened` never actually widens
    for any query that lies inside some element (the only queries whose
    anchor can affect a containment-based interpolation): it reduces to
    `_grid_hash_nearest_3x3`, which is exactly what the batch
    `find_nearest_nodes` path uses on nonempty windows — so the fused
    single-particle anchors are bit-identical to the unfused KD-healed
    ones.  Queries outside every element can still widen, but a
    containment-requiring consumer (barycentric) fails them around ANY
    anchor — same verdict, same failure code.

    Quasi-uniform FESOM meshes pass (cell ≈ 2× mean node spacing); a
    HEALPix dual mesh fails at the polar rows (pixel lon-width grows as
    1/cos(lat) while cells are uniform in degrees) — exactly where the
    widened anchor was measured to diverge from the exact one, killing
    particles the unfused path keeps (review 2026-08-16).  Consumers gate
    the fused barycentric path on this certificate and fall back to the
    unfused (exact) machinery when it is denied.

    Computed lazily once per mesh (`mesh._gh_dense_elems`; invalidated by
    `build_grid_hash`).  Bbox raster is over-conservative: it may deny
    the certificate on exotic geometries the fused path could in fact
    serve — the safe direction (perf loss, never a verdict change).
    """
    if mesh._gh_dense_elems is not None:
        return mesh._gh_dense_elems
    if mesh._gh_offsets is None:
        build_grid_hash(mesh)
    nx, ny = mesh._gh_nx, mesh._gh_ny
    counts = np.diff(mesh._gh_offsets)
    nonempty = (counts > 0).reshape(ny, nx)
    # Chebyshev-1 dilation (separable): a cell is covered iff its 3x3
    # window holds any node.  y edges clip (windows clip there too), x
    # wraps only on cyclic grids (matching the ci2 +/- nx window wrap).
    tmp = nonempty.copy()
    tmp[:-1] |= nonempty[1:]
    tmp[1:] |= nonempty[:-1]
    covered = tmp.copy()
    if mesh._gh_wrap_lon:
        covered |= np.roll(tmp, 1, axis=1)
        covered |= np.roll(tmp, -1, axis=1)
    else:
        covered[:, :-1] |= tmp[:, 1:]
        covered[:, 1:] |= tmp[:, :-1]
    dense = bool(_gh_elements_covered(
        mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat,
        covered.reshape(-1), mesh._gh_cell_size,
        mesh._gh_lon_min, mesh._gh_lat_min, nx, ny, mesh._gh_wrap_lon))
    mesh._gh_dense_elems = dense
    return dense


@njit(cache=True)
def _gh_fold_dlon(lon_p: float, cell_size: float, lon_min: float, nx: int,
                  wrap_lon: bool) -> float:
    """Query longitude relative to ``lon_min`` in the grid's own branch — the
    ONE fold rule shared by :func:`_gh_cell_of` and the exactness certificate
    :func:`_anchor_provable` (review 2026-09-12: the certificate must reason
    about the cell the window actually searched)."""
    dlon = (lon_p - lon_min) % 360.0
    if not wrap_lon:
        # Out-of-domain queries: fold to negative when the query is
        # angularly closer to the WEST edge than to the EAST edge, so the
        # clamp picks the nearer side (mirrors the plain east clamp).
        grid_span = nx * cell_size
        if dlon > grid_span and (dlon - grid_span) > (360.0 - dlon):
            dlon -= 360.0
    return dlon


@njit(cache=True)
def _gh_cell_of(lon_p: float, lat_p: float, cell_size: float,
                lon_min: float, lat_min: float, nx: int, ny: int,
                wrap_lon: bool):
    """Map a query point to its (ci, cj) hash cell — ONE implementation.

    The lon-branch fold below was itself a past bugfix
    (TestGridHashLonBranch); the 3x3 core and the ring-widened miss path
    must share it so a future fix cannot desync their start cells
    (review 2026-08-16).
    """
    dlon = _gh_fold_dlon(lon_p, cell_size, lon_min, nx, wrap_lon)
    ci = int(dlon / cell_size)
    cj = int((lat_p - lat_min) / cell_size)
    if ci < 0:
        ci = 0
    if ci >= nx:
        ci = nx - 1
    if cj < 0:
        cj = 0
    if cj >= ny:
        cj = ny - 1
    return ci, cj


@njit(cache=True)
def _grid_hash_nearest_3x3(
    lon_p: float, lat_p: float,
    gh_offsets: np.ndarray, gh_node_ids: np.ndarray,
    cell_size: float, lon_min: float, lat_min: float,
    nx: int, ny: int, wrap_lon: bool,
    nodes_lon: np.ndarray, nodes_lat: np.ndarray, node_xyz: np.ndarray,
) -> int:
    """3x3-window nearest-node lookup (returns -1 on an empty window).

    The EXACT-contract building block: the vectorized batch path calls
    THIS and leaves misses as -1 so ``find_nearest_nodes`` can heal them
    with its exact KD-tree fallback (review 2026-08-16 — the widened
    single kernel below must never shadow that fallback).

    Uses Cartesian 3D distance on the unit sphere — the same metric as the
    scipy KD-tree — giving identical results for in-domain queries.  The
    candidates' coordinates are read from ``node_xyz`` (``Mesh._node_xyz``,
    built with the hash by ``_node_xyz_fill`` with these very expressions),
    the query's are computed here once.  Far
    out-of-domain (off-mesh) queries clamp to a sensible boundary node via
    the 3×3 cell window, which may differ from the global KD-tree nearest
    (e.g. antipodal points never see lat-extreme cells).
    """
    # Normalize the query longitude into the grid's own branch
    # [lon_min, lon_min + 360) — matches both the [0, 360)-based and the
    # signed build, and for cyclic grids (lon_min = 0, nx·cell ≈ 360°)
    # makes the window wrap ci2 ± nx an exact ±360° shift.
    ci, cj = _gh_cell_of(lon_p, lat_p, cell_size, lon_min, lat_min,
                         nx, ny, wrap_lon)

    # Query point in Cartesian 3D (unit sphere)
    lat_r = lat_p * DEG2RAD
    lon_r = lon_p * DEG2RAD
    cos_lat_p = np.cos(lat_r)
    xp = cos_lat_p * np.cos(lon_r)
    yp = cos_lat_p * np.sin(lon_r)
    zp = np.sin(lat_r)

    best = -1
    best_d2 = 1e30

    for dj in range(-1, 2):
        cj2 = cj + dj
        if cj2 < 0 or cj2 >= ny:
            continue
        for di in range(-1, 2):
            ci2 = ci + di
            if ci2 < 0:
                ci2 = ci2 + nx if wrap_lon else -1
            elif ci2 >= nx:
                ci2 = ci2 - nx if wrap_lon else -1
            if ci2 < 0:
                continue

            cell = cj2 * nx + ci2
            start = gh_offsets[cell]
            end = gh_offsets[cell + 1]
            for k in range(start, end):
                n = gh_node_ids[k]
                xn = node_xyz[n, 0]
                yn = node_xyz[n, 1]
                zn = node_xyz[n, 2]
                d2 = (xn - xp) ** 2 + (yn - yp) ** 2 + (zn - zp) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best = n

    return best


@njit(cache=True)
def _cap_extents(lon_p: float, lat_p: float, lon_n: float, lat_n: float):
    """(theta_deg, ext_lon_deg): the spherical cap of angular radius theta
    (query to found node) spans theta in latitude and ext_lon in longitude;
    ext_lon = 1e30 when the cap reaches a pole (unbounded)."""
    lat_pr = lat_p * DEG2RAD
    lat_nr = lat_n * DEG2RAD
    cosd = (np.sin(lat_pr) * np.sin(lat_nr)
            + np.cos(lat_pr) * np.cos(lat_nr) * np.cos((lon_p - lon_n) * DEG2RAD))
    if cosd > 1.0:
        cosd = 1.0
    if cosd < -1.0:
        cosd = -1.0
    theta = np.arccos(cosd)
    if theta <= 0.0:
        return 0.0, 0.0
    theta_deg = theta * RAD2DEG
    if abs(lat_p) + theta_deg >= 90.0:
        return theta_deg, 1.0e30
    r = np.sin(theta) / np.cos(lat_pr)
    if r >= 1.0:
        return theta_deg, 1.0e30
    return theta_deg, np.arcsin(r) * RAD2DEG


@njit(cache=True)
def _nearest_provably_exact(lon_p: float, lat_p: float, lon_n: float,
                            lat_n: float, cell_size: float) -> bool:
    """Uniform-coverage core of the certificate: the cap around the query
    fits in one cell of coverage on every side (theta <= cell and the
    longitude extent <= cell).  :func:`_anchor_provable` is the same test
    against the intervals a hash window actually searched."""
    theta_deg, ext_lon = _cap_extents(lon_p, lat_p, lon_n, lat_n)
    return theta_deg <= cell_size and ext_lon <= cell_size


@njit(cache=True)
def _anchor_provable(
    lon_p: float, lat_p: float, node: int,
    cell_size: float, lon_min: float, lat_min: float,
    nx: int, ny: int, wrap_lon: bool,
    nodes_lon: np.ndarray, nodes_lat: np.ndarray,
) -> bool:
    """Geometry-aware exactness certificate for a 3x3-window hit — the
    ``rx = ry = 1`` case of :func:`_anchor_provable_rect` (one
    implementation; review round 1 + 2, 2026-09-12): the cap of angular
    radius theta around the query must fit inside the longitude/latitude
    intervals the window ACTUALLY searched (a wrap grid's seam cell is
    narrower than ``cell_size``; clipped rows/columns reach the grid
    edge, beyond which no node exists), and on a non-wrap grid its
    +-360 deg images must miss the node range (a near-global mesh stored
    on one branch).
    """
    return _anchor_provable_rect(
        lon_p, lat_p, node, 1, 1, cell_size, lon_min, lat_min,
        nx, ny, wrap_lon, nodes_lon, nodes_lat) == 0


@njit(cache=True)
def _grid_hash_nearest_widened(
    lon_p: float, lat_p: float,
    gh_offsets: np.ndarray, gh_node_ids: np.ndarray,
    cell_size: float, lon_min: float, lat_min: float,
    nx: int, ny: int, wrap_lon: bool,
    nodes_lon: np.ndarray, nodes_lat: np.ndarray, node_xyz: np.ndarray,
) -> int:
    """Nearest node for the SINGLE-particle numba callers (fused kernels,
    RK/RK45 stages) — no KD-tree is reachable from inside numba, so an
    empty 3x3 window ring-widens outward instead of returning -1 (the
    old -1 made stages probing an empty seam gap fail for no physical
    reason).  The healed answer is an APPROXIMATE anchor: the scan stops
    one ring after the first hit, which is exact for quasi-uniform
    gaps but approximate in general (a later ring can hold a closer node
    in skewed geometries and near the poles) — for the consuming interpolations an approximate
    anchor is strictly better than a dead stage.  Populated windows are
    bit-identical to :func:`_grid_hash_nearest_3x3` (the widening fires
    only where it returned -1).  The vectorized batch path deliberately
    does NOT use this function (it keeps -1 → exact KD fallback).
    """
    best = _grid_hash_nearest_3x3(
        lon_p, lat_p, gh_offsets, gh_node_ids, cell_size,
        lon_min, lat_min, nx, ny, wrap_lon, nodes_lon, nodes_lat, node_xyz)
    if best >= 0:
        return best

    # Recompute the cell/query geometry (miss path only — cheap).
    ci, cj = _gh_cell_of(lon_p, lat_p, cell_size, lon_min, lat_min,
                         nx, ny, wrap_lon)
    lat_r = lat_p * DEG2RAD
    lon_r = lon_p * DEG2RAD
    cos_lat_p = np.cos(lat_r)
    xp = cos_lat_p * np.cos(lon_r)
    yp = cos_lat_p * np.sin(lon_r)
    zp = np.sin(lat_r)
    best_d2 = 1e30

    # Ring-widening on an EMPTY 3x3 window (seam-gap heal,
    # 2026-08-16): a query inside an empty longitude gap wider than
    # a cell on a near-global wrap grid returned -1 and made the
    # fused / RK-stage callers fail for no physical reason (the
    # vectorized ``find_nearest_nodes`` heals via its KD-tree
    # fallback; this single-particle kernel has none).  Scan outward
    # ring by ring; after the first ring that yields a node, scan
    # ONE more ring so a closer diagonal-cell node cannot lose, then
    # stop.  Fires ONLY where the old code returned -1, so populated
    # windows stay bit-identical.
    max_ring = nx if nx > ny else ny
    found_ring = -1
    r = 2
    while r <= max_ring:
        for dj in range(-r, r + 1):
            cj2 = cj + dj
            if cj2 < 0 or cj2 >= ny:
                continue
            if dj == -r or dj == r:
                di_step = 1          # full top/bottom edges
            else:
                di_step = 2 * r      # only the two side cells
            di = -r
            while di <= r:
                ci2 = ci + di
                if wrap_lon:
                    ci2 = ci2 % nx
                elif ci2 < 0 or ci2 >= nx:
                    di += di_step
                    continue
                cell = cj2 * nx + ci2
                start = gh_offsets[cell]
                end = gh_offsets[cell + 1]
                for k in range(start, end):
                    n = gh_node_ids[k]
                    xn = node_xyz[n, 0]
                    yn = node_xyz[n, 1]
                    zn = node_xyz[n, 2]
                    d2 = (xn - xp) ** 2 + (yn - yp) ** 2 + (zn - zp) ** 2
                    if d2 < best_d2:
                        best_d2 = d2
                        best = n
                di += di_step
        if best >= 0:
            if found_ring < 0:
                found_ring = r
            elif r > found_ring:
                break
        r += 1
    return best


@njit(cache=True)
def _gh_scan_cell(cell, gh_offsets, gh_node_ids, node_xyz,
                  xp, yp, zp, best, best_d2):
    """Scan one hash cell for the node nearest (unit-sphere chord) to the
    query ``(xp, yp, zp)``; returns the updated ``(best, best_d2)``.
    Node coordinates come from the per-node table ``node_xyz`` (built with
    the hash), not from per-candidate trig."""
    start = gh_offsets[cell]
    end = gh_offsets[cell + 1]
    for k in range(start, end):
        n = gh_node_ids[k]
        xn = node_xyz[n, 0]
        yn = node_xyz[n, 1]
        zn = node_xyz[n, 2]
        d2 = (xn - xp) ** 2 + (yn - yp) ** 2 + (zn - zp) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best = n
    return best, best_d2


@njit(cache=True)
def _anchor_provable_rect(
    lon_p: float, lat_p: float, node: int, rx: int, ry: int,
    cell_size: float, lon_min: float, lat_min: float,
    nx: int, ny: int, wrap_lon: bool,
    nodes_lon: np.ndarray, nodes_lat: np.ndarray,
) -> int:
    """Certificate for the ``(2rx+1) x (2ry+1)`` window of cells centred on
    the query's cell (:func:`_anchor_provable` is the 3x3 case): ``node``
    is the best node of that window.  Returns 0 when it is provably the
    true nearest, else a bitmask of what the window lacks — 1: rows (the
    cap's latitude extent is not covered), 2: columns (its longitude
    extent is not), 4: EVERY column (the cap reaches a pole, or its
    +-360 deg image reaches the far end of a non-wrap grid).  Same
    geometry as the 3x3 certificate: a wrap grid's seam cell is narrower
    than ``cell_size``, clipped rows/columns reach the grid edge, a
    window that spans all columns needs no longitude coverage at all.
    """
    theta_deg, ext_lon = _cap_extents(lon_p, lat_p, nodes_lon[node],
                                      nodes_lat[node])
    if theta_deg <= 0.0:
        return 0
    dlon = _gh_fold_dlon(lon_p, cell_size, lon_min, nx, wrap_lon)
    lon_q = lon_min + dlon
    ci_raw = int(np.floor(dlon / cell_size))
    cj_raw = int(np.floor((lat_p - lat_min) / cell_size))
    INF = 1.0e30
    cj = cj_raw
    if cj < 0:
        cj = 0
    if cj > ny - 1:
        cj = ny - 1
    cov_dn = lat_p - (lat_min + (cj - ry) * cell_size) if cj - ry >= 0 else INF
    cov_up = (lat_min + (cj + ry + 1) * cell_size) - lat_p if cj + ry <= ny - 1 else INF
    code = 0
    if theta_deg > cov_dn or theta_deg > cov_up:
        code |= 1
    if wrap_lon:
        if 2 * rx + 1 >= nx:
            return code                        # every column searched
        if ext_lon >= INF:
            return code | 4
        w_last = 360.0 - (nx - 1) * cell_size  # the seam cell (index nx-1) is narrower
        ci = ci_raw % nx
        left = lon_min + (ci - rx) * cell_size
        if ci - rx < 0:                        # the seam cell is among the rx columns to the left
            left += cell_size - w_last
        right = lon_min + (ci + rx + 1) * cell_size
        if ci + rx >= nx - 1:                  # ... or among the rx columns to the right
            right -= cell_size - w_last
        if ext_lon > lon_q - left or ext_lon > right - lon_q:
            code |= 2
        return code
    ci = ci_raw
    if ci < 0:
        ci = 0
    if ci > nx - 1:
        ci = nx - 1
    cov_l = lon_q - (lon_min + (ci - rx) * cell_size) if ci - rx >= 0 else INF
    cov_r = (lon_min + (ci + rx + 1) * cell_size) - lon_q if ci + rx <= nx - 1 else INF
    if cov_l >= INF and cov_r >= INF:
        return code                            # every column searched
    if ext_lon >= INF:
        return code | 4
    if ext_lon > cov_l or ext_lon > cov_r:
        code |= 2
    # a +-360 image of the cap must not reach the node range of the grid
    n_lo = lon_min + cell_size
    n_hi = lon_min + (nx - 1) * cell_size
    for shift in (-360.0, 360.0):
        a = lon_q - ext_lon + shift
        b = lon_q + ext_lon + shift
        if b >= n_lo and a <= n_hi:
            return code | 4
    return code


@njit(cache=True)
def _gh_scan_row_range(cj2, c_lo, c_hi, nx, gh_offsets, gh_node_ids,
                       node_xyz, xp, yp, zp, best, best_d2):
    """Scan cells ``c_lo..c_hi`` (inclusive, no seam crossing) of row
    ``cj2`` as ONE contiguous node slice: the hash is CSR over
    ``cell = cj * nx + ci``, so a row segment's nodes are adjacent in
    ``gh_node_ids`` and the cost is the nodes it holds, not the cells it
    spans (polar rows on a fine cell are thousands of empty cells)."""
    start = gh_offsets[cj2 * nx + c_lo]
    end = gh_offsets[cj2 * nx + c_hi + 1]
    for k in range(start, end):
        n = gh_node_ids[k]
        xn = node_xyz[n, 0]
        yn = node_xyz[n, 1]
        zn = node_xyz[n, 2]
        d2 = (xn - xp) ** 2 + (yn - yp) ** 2 + (zn - zp) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best = n
    return best, best_d2


@njit(cache=True)
def _gh_scan_row_window(cj2, ci, rx, nx, wrap_lon, gh_offsets, gh_node_ids,
                        node_xyz, xp, yp, zp, best, best_d2):
    """Scan columns ``ci-rx..ci+rx`` of row ``cj2``: every column once a
    wrap grid's window spans the circle, else one or two contiguous
    slices (a seam crossing splits the range)."""
    if wrap_lon and 2 * rx + 1 >= nx:
        return _gh_scan_row_range(cj2, 0, nx - 1, nx, gh_offsets, gh_node_ids,
                                  node_xyz, xp, yp, zp, best, best_d2)
    c_lo = ci - rx
    c_hi = ci + rx
    if wrap_lon:
        if c_lo < 0:
            best, best_d2 = _gh_scan_row_range(
                cj2, c_lo + nx, nx - 1, nx, gh_offsets, gh_node_ids,
                node_xyz, xp, yp, zp, best, best_d2)
            c_lo = 0
        elif c_hi > nx - 1:
            best, best_d2 = _gh_scan_row_range(
                cj2, 0, c_hi - nx, nx, gh_offsets, gh_node_ids,
                node_xyz, xp, yp, zp, best, best_d2)
            c_hi = nx - 1
    else:
        if c_lo < 0:
            c_lo = 0
        if c_hi > nx - 1:
            c_hi = nx - 1
    return _gh_scan_row_range(cj2, c_lo, c_hi, nx, gh_offsets, gh_node_ids,
                              node_xyz, xp, yp, zp, best, best_d2)


#: an EMPTY window jumps straight to full rows where a row is angularly this
#: short (cells; the polar case — a row there holds few nodes), else it grows
#: ring by ring until the first hit
_GH_EXACT_SHORT_ROW_CELLS = 32


@njit(cache=True)
def _grid_hash_nearest_exact(
    lon_p: float, lat_p: float,
    gh_offsets: np.ndarray, gh_node_ids: np.ndarray,
    cell_size: float, lon_min: float, lat_min: float,
    nx: int, ny: int, wrap_lon: bool,
    nodes_lon: np.ndarray, nodes_lat: np.ndarray, node_xyz: np.ndarray,
    far_rows: int,
) -> int:
    """EXACT nearest node for the single-particle numba callers (the fused
    kernels under ``mesh.exact_nearest``) AND the batch path in exact mode
    (2026-09-12).

    The 3x3 window first (accepted iff :func:`_anchor_provable`), then the
    searched window is widened as a RECTANGLE driven by the certificate's
    deficits (:func:`_anchor_provable_rect`): more rows when the cap's
    latitude extent is uncovered, more columns when its longitude extent
    is, all columns at once when the cap reaches a pole or its +-360 image
    reaches a non-wrap grid's far end — until the best node found is
    provably the nearest or every cell has been scanned.  Row segments are
    scanned as contiguous CSR slices, so the cost is the NODES the
    rectangle holds plus one lookup per column cell, never the empty cells
    it spans (a polar row on a fine cell).  An empty 3x3 window grows ring
    by ring until the first hit, except where a row is angularly short
    (``_GH_EXACT_SHORT_ROW_CELLS``, the polar case: few nodes per row), where
    it jumps to full rows.  Unit-sphere chord metric, first-found wins on
    a tie — the batch path calls this same function, so fused and unfused
    anchors agree bit for bit.

    Budget ``far_rows`` (``mesh._gh_far_rows``, derived by ``build_grid_hash``
    from the longest element edge in cells + 2): a query whose nearest node
    is farther than that (in latitude terms) is outside every element —
    far outside the grid, or deep inside a land hole — and its particle
    dies this step; the search returns the best node seen (or the plain
    widened-window anchor) instead of scanning the grid.  The guarantee
    (exact for every query whose nearest node is within the budget) holds
    for queries inside the grid's node range; a query outside it in
    longitude may keep the clamped edge window's node or the clearance
    exit's best-so-far, which is not always the nearest (over-the-pole
    paths; the column clearance is a grid-frame offset) but is outside
    every element too.
    """
    far = far_rows * cell_size
    # far outside the grid in latitude (a latitude offset is a lower bound of
    # the angular distance; a longitude offset is not, that case is caught by
    # the theta check on the clamped edge window below): the widened window
    # anchor, no scan at all
    if lat_p < lat_min - far or lat_p > lat_min + ny * cell_size + far:
        return _grid_hash_nearest_widened(
            lon_p, lat_p, gh_offsets, gh_node_ids, cell_size,
            lon_min, lat_min, nx, ny, wrap_lon, nodes_lon, nodes_lat, node_xyz)
    best = _grid_hash_nearest_3x3(
        lon_p, lat_p, gh_offsets, gh_node_ids, cell_size,
        lon_min, lat_min, nx, ny, wrap_lon, nodes_lon, nodes_lat, node_xyz)
    code = 0
    if best >= 0:
        code = _anchor_provable_rect(
            lon_p, lat_p, best, 1, 1, cell_size, lon_min, lat_min,
            nx, ny, wrap_lon, nodes_lon, nodes_lat)
        if code == 0:
            return best
        theta_deg, _ext = _cap_extents(lon_p, lat_p, nodes_lon[best],
                                       nodes_lat[best])
        if theta_deg > far:
            return best                        # far: the widened anchor (non-empty window)

    ci, cj = _gh_cell_of(lon_p, lat_p, cell_size, lon_min, lat_min,
                         nx, ny, wrap_lon)
    lat_r = lat_p * DEG2RAD
    lon_r = lon_p * DEG2RAD
    cos_lat_p = np.cos(lat_r)
    xp = cos_lat_p * np.cos(lon_r)
    yp = cos_lat_p * np.sin(lon_r)
    zp = np.sin(lat_r)
    best_d2 = 1.0e30
    if best >= 0:
        xn = node_xyz[best, 0]
        yn = node_xyz[best, 1]
        zn = node_xyz[best, 2]
        best_d2 = (xn - xp) ** 2 + (yn - yp) ** 2 + (zn - zp) ** 2

    rx = 1
    ry = 1
    # a row's angular length: short rows (the polar cap) are scanned whole on
    # an empty window, long rows are approached as a growing square
    short_row = 360.0 * cos_lat_p <= _GH_EXACT_SHORT_ROW_CELLS * cell_size
    # widest useful half-widths: every column (wrap: 2*rx+1 >= nx; else
    # both sides clipped) / every row
    if wrap_lon:
        rx_max = nx // 2
        if rx_max < 1:
            rx_max = 1
    else:
        rx_max = ci if ci > nx - 1 - ci else nx - 1 - ci
    ry_max = cj if cj > ny - 1 - cj else ny - 1 - cj
    abs_lat = lat_p if lat_p >= 0.0 else -lat_p
    dlon_q = _gh_fold_dlon(lon_p, cell_size, lon_min, nx, wrap_lon)   # query offset in the grid
    INF = 1.0e30
    while True:
        if best >= 0 and code == 0:            # (re)certify the current rectangle
            code = _anchor_provable_rect(
                lon_p, lat_p, best, rx, ry, cell_size, lon_min, lat_min,
                nx, ny, wrap_lon, nodes_lon, nodes_lat)
            if code == 0:
                return best
        # The far exit can only matter while nothing within the budget has
        # been seen (a best within it is proven by the certificate as soon
        # as the rectangle covers its cap); then the clearance — no node
        # outside the scanned rectangle can be closer than this: rows, the
        # band's latitude offset; columns, the longitude offset at the
        # band's poleward edge (none once every column is in) — decides.
        # Beyond the budget the query is far from every node (review round
        # 3: waiting for every column made a land-hole seed scan the
        # whole latitude band).
        if best < 0 or 2.0 * np.arcsin(0.5 * np.sqrt(best_d2)) * RAD2DEG > far:
            if wrap_lon:
                cols_full = 2 * rx + 1 >= nx
            else:
                cols_full = ci - rx <= 0 and ci + rx >= nx - 1
            clear = INF if ry >= ry_max else ry * cell_size
            if not cols_full:
                lat_edge = abs_lat + (ry + 1) * cell_size
                if lat_edge >= 90.0:
                    clear_lon = 0.0
                else:
                    # longitude offset to the first unscanned column: rx cells
                    # (a wrap grid's narrower seam cell shortens it by at most
                    # cell - w_last, absorbed because the exit needs rx*cell
                    # strictly beyond the budget); on a non-wrap grid the far
                    # end of the grid is also reachable across the branch cut,
                    # at least gap + the offset to the nearer grid edge away
                    dl = rx * cell_size
                    if not wrap_lon:
                        x = dlon_q if dlon_q > 0.0 else 0.0
                        w = nx * cell_size
                        if x > w:
                            x = w
                        near_edge = x if x < w - x else w - x
                        dl_cut = 360.0 - w + near_edge
                        if dl_cut < dl:
                            dl = dl_cut
                    if dl > 180.0:
                        dl = 180.0
                    clear_lon = np.arcsin(np.sin(dl * DEG2RAD)
                                          * np.cos(lat_edge * DEG2RAD)) * RAD2DEG
                if clear_lon < clear:
                    clear = clear_lon
            if clear >= far:
                if best >= 0:
                    return best                # far from every node: see the budget above
                return _grid_hash_nearest_widened(
                    lon_p, lat_p, gh_offsets, gh_node_ids, cell_size,
                    lon_min, lat_min, nx, ny, wrap_lon, nodes_lon, nodes_lat,
                    node_xyz)
        if best >= 0:
            grow_x = (code & 2) != 0
            grow_y = (code & 1) != 0
            all_x = (code & 4) != 0
            code = 0                           # consumed: certify again after the scan
        elif short_row:                        # nothing found yet, polar cap: full rows
            grow_x = False
            grow_y = True
            all_x = True
        else:                                  # nothing found yet: a growing square
            grow_x = True
            grow_y = True
            all_x = False
        if rx >= rx_max:
            grow_x = False
            all_x = False
        if ry >= ry_max:
            grow_y = False
        if not (grow_x or grow_y or all_x):
            if best >= 0:
                return best                    # every cell scanned
            return _grid_hash_nearest_widened(
                lon_p, lat_p, gh_offsets, gh_node_ids, cell_size,
                lon_min, lat_min, nx, ny, wrap_lon, nodes_lon, nodes_lat,
                node_xyz)
        rx_new = rx_max if all_x else (rx + 1 if grow_x else rx)
        ry_new = ry + 1 if grow_y else ry
        # 1. new column pairs rx+1..rx_new over the OLD rows cj-ry..cj+ry
        j_lo = cj - ry if cj - ry > 0 else 0
        j_hi = cj + ry if cj + ry < ny - 1 else ny - 1
        if rx_new > rx:
            if wrap_lon and 2 * rx_new + 1 >= nx and rx_new - rx > 1:
                # a jump to full width: the old rows' remaining columns as
                # (at most two) contiguous slices instead of cell by cell
                # (columns ci+rx+1 .. ci-rx-1 mod nx; 2*rx+1 < nx here)
                for cj2 in range(j_lo, j_hi + 1):
                    c_lo = (ci + rx + 1) % nx
                    c_hi = (ci - rx - 1) % nx
                    if c_lo <= c_hi:
                        best, best_d2 = _gh_scan_row_range(
                            cj2, c_lo, c_hi, nx, gh_offsets, gh_node_ids,
                            node_xyz, xp, yp, zp, best, best_d2)
                    else:
                        best, best_d2 = _gh_scan_row_range(
                            cj2, c_lo, nx - 1, nx, gh_offsets, gh_node_ids,
                            node_xyz, xp, yp, zp, best, best_d2)
                        best, best_d2 = _gh_scan_row_range(
                            cj2, 0, c_hi, nx, gh_offsets, gh_node_ids,
                            node_xyz, xp, yp, zp, best, best_d2)
            elif not wrap_lon and rx_new - rx > 1:
                # non-wrap jump to the grid edges: the unscanned column
                # ranges on each side as contiguous slices
                for cj2 in range(j_lo, j_hi + 1):
                    if ci - rx - 1 >= 0:
                        c_lo = ci - rx_new if ci - rx_new > 0 else 0
                        best, best_d2 = _gh_scan_row_range(
                            cj2, c_lo, ci - rx - 1, nx, gh_offsets, gh_node_ids,
                            node_xyz, xp, yp, zp, best, best_d2)
                    if ci + rx + 1 <= nx - 1:
                        c_hi = ci + rx_new if ci + rx_new < nx - 1 else nx - 1
                        best, best_d2 = _gh_scan_row_range(
                            cj2, ci + rx + 1, c_hi, nx, gh_offsets, gh_node_ids,
                            node_xyz, xp, yp, zp, best, best_d2)
            else:
                for r in range(rx + 1, rx_new + 1):
                    if wrap_lon:
                        c_left = (ci - r) % nx
                        two = (2 * r + 1 <= nx)   # else ci+r duplicates ci-r
                        c_right = (ci + r) % nx
                        for cj2 in range(j_lo, j_hi + 1):
                            best, best_d2 = _gh_scan_cell(
                                cj2 * nx + c_left, gh_offsets, gh_node_ids,
                                node_xyz, xp, yp, zp, best, best_d2)
                            if two:
                                best, best_d2 = _gh_scan_cell(
                                    cj2 * nx + c_right, gh_offsets, gh_node_ids,
                                    node_xyz, xp, yp, zp, best, best_d2)
                    else:
                        for c2 in (ci - r, ci + r):
                            if c2 < 0 or c2 > nx - 1:
                                continue
                            for cj2 in range(j_lo, j_hi + 1):
                                best, best_d2 = _gh_scan_cell(
                                    cj2 * nx + c2, gh_offsets, gh_node_ids,
                                    node_xyz, xp, yp, zp, best, best_d2)
        # 2. new row pairs ry+1..ry_new over the NEW column range (slices)
        for r in range(ry + 1, ry_new + 1):
            for cj2 in (cj - r, cj + r):
                if cj2 < 0 or cj2 > ny - 1:
                    continue
                best, best_d2 = _gh_scan_row_window(
                    cj2, ci, rx_new, nx, wrap_lon, gh_offsets, gh_node_ids,
                    node_xyz, xp, yp, zp, best, best_d2)
        rx = rx_new
        ry = ry_new


@njit(parallel=True, cache=True)
def _grid_hash_nearest_batch(
    P: int, lons: np.ndarray, lats: np.ndarray,
    gh_offsets: np.ndarray, gh_node_ids: np.ndarray,
    cell_size: float, lon_min: float, lat_min: float,
    nx: int, ny: int, wrap_lon: bool,
    nodes_lon: np.ndarray, nodes_lat: np.ndarray, node_xyz: np.ndarray,
    out: np.ndarray, exact: bool, far_rows: int,
):
    """Batch nearest-node query via grid hash.  Parallel over particles.
    ``exact`` runs the in-kernel exact search instead of the plain 3x3
    lookup — the SAME function the fused kernels anchor with under
    ``mesh.exact_nearest``, so the batch (unfused) and fused paths agree
    bit for bit, exact ties included (review round 2, 2026-09-12; a
    KD-tree heal broke ties in a different order)."""
    for i in prange(P):
        if exact:
            out[i] = _grid_hash_nearest_exact(
                lons[i], lats[i],
                gh_offsets, gh_node_ids,
                cell_size, lon_min, lat_min,
                nx, ny, wrap_lon,
                nodes_lon, nodes_lat, node_xyz, far_rows,
            )
        else:
            out[i] = _grid_hash_nearest_3x3(
                lons[i], lats[i],
                gh_offsets, gh_node_ids,
                cell_size, lon_min, lat_min,
                nx, ny, wrap_lon,
                nodes_lon, nodes_lat, node_xyz,
            )


def find_nearest_nodes(mesh: Mesh,
                       lons: np.ndarray,
                       lats: np.ndarray) -> np.ndarray:
    """Find nearest mesh node for each query point.

    Uses grid hash (Numba prange) when available, falls back to KD-tree.

    Parameters
    ----------
    mesh : Mesh
        Mesh with search structures built (call ``build_search`` first).
    lons, lats : np.ndarray
        Query point coordinates in degrees, shape (P,).

    Returns
    -------
    np.ndarray
        Nearest node indices, shape (P,), dtype int32.

    Raises
    ------
    RuntimeError
        If ``build_search`` has not been called.
    """
    # Prefer grid hash (Numba-compatible, parallel)
    if mesh._gh_offsets is not None and HAS_NUMBA:
        node_xyz = _require_node_xyz(mesh)
        lons = np.asarray(lons, dtype=np.float64)
        lats = np.asarray(lats, dtype=np.float64)
        P = lons.shape[0]
        out = np.empty(P, dtype=np.int32)
        _grid_hash_nearest_batch(
            P, lons, lats,
            mesh._gh_offsets, mesh._gh_node_ids,
            mesh._gh_cell_size, mesh._gh_lon_min, mesh._gh_lat_min,
            mesh._gh_nx, mesh._gh_ny, mesh._gh_wrap_lon,
            mesh.nodes_lon, mesh.nodes_lat, node_xyz,
            out, bool((mesh.metadata or {}).get("exact_nearest", False)),
            int(mesh._gh_far_rows),
        )
        # Grid-hash miss: every cell of the 3x3 window empty -> -1 (under the
        # exact_nearest guard the in-kernel search never misses on a non-empty
        # mesh: it widens to the whole grid if it must).  Never
        # happens on quasi-uniform-in-lon/lat meshes (FESOM in-domain), but
        # DOES on strongly anisotropic node layouts — e.g. HEALPix polar
        # rings, whose longitude gaps (90deg/ring) dwarf the auto cell size
        # at high nside — and can for far off-domain queries.  A -1 anchor
        # is not a valid contract downstream (node_to_elems[-1] wraps
        # Python-style into the LAST node's row — silently wrong scan), so
        # fall back to the exact KD-tree for the missed queries only;
        # results are byte-identical whenever the grid hash succeeds.
        if (out < 0).any():
            miss = np.flatnonzero(out < 0)
            _ensure_kdtree(mesh)
            if mesh._tree_cartesian:
                pts = _lonlat_to_xyz(lons[miss], lats[miss])
            else:
                pts = np.column_stack((lons[miss], lats[miss]))
            _, idx = mesh._node_tree.query(pts, k=1)
            out[miss] = np.asarray(idx, dtype=np.int32)
        return out

    # Fallback: KD-tree (lazy build on first access)
    _ensure_kdtree(mesh)
    if mesh._tree_cartesian:
        pts = _lonlat_to_xyz(lons, lats)
    else:
        pts = np.column_stack((np.asarray(lons, dtype=np.float64),
                               np.asarray(lats, dtype=np.float64)))
    _, idx = mesh._node_tree.query(pts, k=1)
    return np.asarray(idx, dtype=np.int32)


# ---------------------------------------------------------------------------
# Element containment: barycentric weights
# ---------------------------------------------------------------------------

@njit(cache=True)
def _seam_safe_lon0(nodes_lon: np.ndarray, verts: np.ndarray, n_valid: int) -> float:
    """Longitude origin of an element's local projection plane.

    The raw mean of the vertex longitudes (the same summation order as the
    barycentric kernels always used, so off-seam elements keep their
    arithmetic bit for bit); for an element whose raw longitude span
    exceeds 180 deg -- it straddles the antimeridian -- the first vertex
    plus the mean of the wrapped offsets from it, the rule of
    :func:`_project_to_local_xy`.  The raw mean of a quad split two-and-two
    across +-180 lands 180 deg away from the element; the wrapped vertex
    offsets then split around +-180 and the element projects to a
    360-deg-wide shape that contains nothing.
    """
    first = nodes_lon[verts[0]]
    lon_min = first
    lon_max = first
    s = 0.0
    for v in range(n_valid):
        x = nodes_lon[verts[v]]
        s += x
        if x < lon_min:
            lon_min = x
        if x > lon_max:
            lon_max = x
    if lon_max - lon_min > 180.0:
        d = 0.0
        for v in range(n_valid):
            d += ((nodes_lon[verts[v]] - first + 180.0) % 360.0) - 180.0
        return first + d / n_valid
    return s / n_valid


@njit(cache=True)
def _project_to_local_xy(lon_nodes: np.ndarray, lat_nodes: np.ndarray,
                          lon_p: float, lat_p: float):
    """Project polygon nodes and a point to local Cartesian (meters).

    Uses dateline-safe longitude differences and projects relative to
    the polygon centroid.
    """
    # Seam gate: a cell whose raw lon span exceeds 180 deg straddles the
    # antimeridian; the raw-mean lon0 then lands ~180 deg away and the
    # wrapped vertex deltas split around +/-180, projecting the polygon to
    # garbage that rejects its own interior.  Anchor the centroid to the
    # first vertex (wrapped-delta mean) ONLY in that case — every
    # downstream delta is already wrapped vs lon0, so non-seam cells keep
    # the original raw-mean arithmetic byte-for-byte (branch contract).
    n = lon_nodes.shape[0]
    lon_min = lon_nodes[0]
    lon_max = lon_nodes[0]
    for k in range(n):
        if lon_nodes[k] < lon_min:
            lon_min = lon_nodes[k]
        if lon_nodes[k] > lon_max:
            lon_max = lon_nodes[k]
    if lon_max - lon_min > 180.0:
        ref = lon_nodes[0]
        dsum = 0.0
        for k in range(n):
            dsum += ((lon_nodes[k] - ref + 180.0) % 360.0) - 180.0
        lon0 = ref + dsum / n
    else:
        lon0 = np.mean(lon_nodes)
    lat0 = np.mean(lat_nodes)
    cos_lat0 = np.cos(lat0 * DEG2RAD)

    dlon_n = ((lon_nodes - lon0 + 180.0) % 360.0) - 180.0
    dlat_n = lat_nodes - lat0
    dlon_p = ((lon_p - lon0 + 180.0) % 360.0) - 180.0
    dlat_p = lat_p - lat0

    # Convert to meters
    x_n = dlon_n * DEG2RAD * R_EARTH_STANDARD * cos_lat0
    y_n = dlat_n * DEG2RAD * R_EARTH_STANDARD
    x_p = dlon_p * DEG2RAD * R_EARTH_STANDARD * cos_lat0
    y_p = dlat_p * DEG2RAD * R_EARTH_STANDARD

    return x_n, y_n, x_p, y_p


@njit(cache=True)
def _barycentric_weights_tri(px: float, py: float,
                              tri_x: np.ndarray, tri_y: np.ndarray) -> np.ndarray:
    """Compute barycentric weights for a point in a triangle.

    Returns array of 3 weights.  If the triangle is degenerate,
    returns [-1, -1, -1].
    """
    ax, ay = tri_x[0], tri_y[0]
    bx, by = tri_x[1], tri_y[1]
    cx, cy = tri_x[2], tri_y[2]

    denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if abs(denom) < _AREA_TOL:
        return np.array([-1.0, -1.0, -1.0])

    w1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denom
    w2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denom
    w3 = 1.0 - w1 - w2
    return np.array([w1, w2, w3])


@njit(cache=True)
def _wachspress_weights_quad(px: float, py: float,
                             x_n: np.ndarray, y_n: np.ndarray) -> np.ndarray:
    """Wachspress coordinates of a point in a convex quad — product form.

    ``w_i ∝ C_i · A_{i+1} · A_{i+2}``, where ``A_j`` = signed area of
    triangle ``(p, v_j, v_{j+1})`` and ``C_i`` = signed area of the
    corner triangle ``(v_{i-1}, v_i, v_{i+1})``, indices cyclic (all
    doubled areas — the factors cancel in the normalized ratio).  This
    is the classical ``w_i = C_i / (A_{i-1}·A_i)`` multiplied through by
    ``Π_j A_j``: division-free, hence **no on-edge singularity** — on an
    edge the form reduces exactly to linear interpolation along it, at a
    vertex to that vertex's indicator.  Winding-invariant without any
    orientation handling: each term carries three sign-flipping factors,
    so CW input negates all weights uniformly and the normalization
    cancels it.  Exact for linear fields; smooth inside the cell; every
    vertex contributes.

    Convexity and non-degeneracy of quads are a MESH CONTRACT (FESOM
    quads are always convex — reviewer guidance 2026-07): there are no
    guards and no fallback sentinel here.
    """
    A = np.empty(4)
    C = np.empty(4)
    for k in range(4):
        i1 = (k + 1) % 4
        im = (k - 1) % 4
        A[k] = ((x_n[k] - px) * (y_n[i1] - py)
                - (x_n[i1] - px) * (y_n[k] - py))
        C[k] = ((x_n[k] - x_n[im]) * (y_n[i1] - y_n[im])
                - (x_n[i1] - x_n[im]) * (y_n[k] - y_n[im]))

    out = np.empty(4)
    s = 0.0
    for k in range(4):
        w = C[k] * A[(k + 1) % 4] * A[(k + 2) % 4]
        out[k] = w
        s += w
    for k in range(4):
        out[k] = out[k] / s
    return out


@njit(cache=True)
def _polygon_weights(px: float, py: float,
                     x_n: np.ndarray, y_n: np.ndarray, nv: int) -> np.ndarray:
    """Vertex weights of a point inside a convex polygon (tri or quad).

    ``nv == 3`` -> the untouched classical barycentric primitive
    (pure-tri meshes keep their results bit-for-bit; its
    degenerate-triangle sentinel ``[-1, -1, -1]`` passes through).
    ``nv == 4`` -> product-form Wachspress, guard-free (convex quads are
    a mesh contract).  Callers guarantee ``nv ∈ {3, 4}`` — compacted
    FESOM element rings have no other arity.
    """
    if nv == 3:
        w3 = _barycentric_weights_tri(px, py, x_n[:3], y_n[:3])
        out = np.full(nv, -1.0)
        out[0], out[1], out[2] = w3[0], w3[1], w3[2]
        return out
    return _wachspress_weights_quad(px, py, x_n[:4], y_n[:4])


@njit(cache=True)
def _polygon_contains_bool(lon_nodes, lat_nodes, lon_p, lat_p):
    """Containment TEST only — no weights computed.

    Identical accept/reject semantics to ``_polygon_contains`` (same
    projection, same fan triangulation, same ``_BARY_EPS``), used by the
    hot containment path (``_element_contains``) where the weights are
    discarded.  ``_polygon_contains`` (with Wachspress weights) remains
    the API for callers that consume weights.
    """
    n = lon_nodes.shape[0]
    if n < 3:
        return False

    x_n, y_n, px, py = _project_to_local_xy(lon_nodes, lat_nodes, lon_p, lat_p)

    if n == 3:
        w = _barycentric_weights_tri(px, py, x_n, y_n)
        return bool(np.all(w >= _BARY_EPS))

    # Quad: fan from vertex 0 — containment TEST only (exact for convex
    # polygons), mirroring _polygon_contains's accept loop without the
    # Wachspress weight solve that the caller would discard.
    for i in range(1, n - 1):
        idx = np.array([0, i, i + 1], dtype=np.int32)
        w = _barycentric_weights_tri(px, py, x_n[idx], y_n[idx])
        if np.all(w >= _BARY_EPS):
            return True

    return False


@njit(cache=True)
def _polygon_contains(lon_nodes: np.ndarray, lat_nodes: np.ndarray,
                       lon_p: float, lat_p: float) -> Optional[np.ndarray]:
    """Check if point is inside a polygon and return weights.

    For triangles: direct barycentric test.
    For quads: fan triangulation (0-1-2, 0-2-3).

    Returns
    -------
    np.ndarray or None
        Weights (len = n_vertices) if inside, else None.
    """
    n = lon_nodes.shape[0]
    if n < 3:
        return None

    x_n, y_n, px, py = _project_to_local_xy(lon_nodes, lat_nodes, lon_p, lat_p)

    if n == 3:
        w = _barycentric_weights_tri(px, py, x_n, y_n)
        if np.all(w >= _BARY_EPS):
            return w
        return None

    # Quad: fan from vertex 0 — containment TEST only (exact for convex
    # polygons; edge semantics inherited from the unchanged fan loop +
    # _BARY_EPS, with the tri machinery pinned by the jitter suite).  The
    # returned WEIGHTS come from the dispatcher: product-form Wachspress,
    # valid on edges and vertices too (guard chain removed 2026-07).
    for i in range(1, n - 1):
        idx = np.array([0, i, i + 1], dtype=np.int32)
        w = _barycentric_weights_tri(px, py, x_n[idx], y_n[idx])
        if np.all(w >= _BARY_EPS):
            return _polygon_weights(px, py, x_n, y_n, n)

    return None


@njit(cache=True)
def _element_contains(
    c: int,
    lon_p: float,
    lat_p: float,
    elem_nodes: np.ndarray,
    nodes_lon: np.ndarray,
    nodes_lat: np.ndarray,
) -> bool:
    """Test whether element ``c`` contains the point (bbox prefilter +
    fan-triangle sign test; no weights are computed)."""
    nodes = elem_nodes[c]
    mask = nodes >= 0
    if nodes.shape[0] == 4 and nodes[3] == nodes[0] and nodes[0] >= 0:
        mask[3] = False

    n_valid = 0
    for v in mask:
        if v:
            n_valid += 1
    if n_valid < 3:
        return False

    valid = np.zeros(n_valid, dtype=np.int32)
    idx = 0
    for j in range(nodes.shape[0]):
        if mask[j]:
            valid[idx] = nodes[j]
            idx += 1

    lon_n = nodes_lon[valid]
    lat_n = nodes_lat[valid]

    # Quick bounding-box check (dateline-safe for longitude).  Anchor on
    # the first vertex and run the whole bbox in WRAPPED-delta space: a
    # raw mean of [179.8, -179.8, ...] collapses to ~0 (centroid 180 deg
    # away) AND the raw min/max span the seam, so a seam-straddling cell
    # would reject every interior point.
    # SOUNDNESS INVARIANT: `_gh_elements_covered` (the grid-hash density
    # certificate) rasters exactly THIS first-valid-vertex wrapped-delta
    # bbox — any point this prefilter accepts lies inside the certified
    # cell range.  Re-anchoring this bbox (e.g. on the centroid) without
    # updating the raster breaks the certificate's "contained => covered
    # cell" theorem (review 2026-08-16; pinned by
    # test_fused_bary_tinterp.py).
    lon_ref = lon_n[0]
    dlon_n = ((lon_n - lon_ref + 180.0) % 360.0) - 180.0
    dlon_p = ((lon_p - lon_ref + 180.0) % 360.0) - 180.0
    if dlon_p > np.max(dlon_n) or dlon_p < np.min(dlon_n):
        return False
    if lat_p > np.max(lat_n) or lat_p < np.min(lat_n):
        return False

    return _polygon_contains_bool(lon_n, lat_n, lon_p, lat_p)


@njit(parallel=True, cache=True)
def _loop_find_containing_elements(
    P: int, lons: np.ndarray, lats: np.ndarray, nn_nodes: np.ndarray,
    node_to_elems_count: np.ndarray, node_to_elems: np.ndarray,
    elem_nodes: np.ndarray, nodes_lon: np.ndarray, nodes_lat: np.ndarray,
    elem_ids: np.ndarray, in_domain: np.ndarray
):
    # Delegates to ``_find_containing_element_local`` so the batch and
    # single-particle paths share one implementation (incl. the 2-ring
    # fallback for irregular meshes).
    for i in prange(P):
        eid, ok = _find_containing_element_local(
            lons[i], lats[i], int(nn_nodes[i]),
            node_to_elems, node_to_elems_count,
            elem_nodes, nodes_lon, nodes_lat,
        )
        if ok:
            elem_ids[i] = eid
            in_domain[i] = True

def find_containing_elements(mesh: Mesh,
                              lons: np.ndarray,
                              lats: np.ndarray,
                              nn_nodes: np.ndarray
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """Find the containing element for each particle.

    Searches elements adjacent to the nearest node.  Falls back to
    marking the particle as outside if no containing element is found.

    Parameters
    ----------
    mesh : Mesh
    lons, lats : np.ndarray
        Particle positions, shape (P,).
    nn_nodes : np.ndarray
        Nearest-node indices from ``find_nearest_nodes``, shape (P,).

    Returns
    -------
    elem_ids : np.ndarray
        Containing element index for each particle, shape (P,).
        Set to -1 if outside domain.
    in_domain : np.ndarray
        Boolean mask, True if a containing element was found.
    """
    P = lons.shape[0]
    elem_ids = np.full(P, -1, dtype=np.int32)
    in_domain = np.zeros(P, dtype=np.bool_)
    
    if HAS_NUMBA:
        _loop_find_containing_elements(
            P, lons, lats, nn_nodes,
            mesh.node_to_elems_count, mesh.node_to_elems,
            mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat,
            elem_ids, in_domain
        )
    else:
        # Fallback to python loops (omitted here for brevity, typically would match the numba implementation)
        _loop_find_containing_elements(
            P, lons, lats, nn_nodes,
            mesh.node_to_elems_count, mesh.node_to_elems,
            mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat,
            elem_ids, in_domain
        )

    return elem_ids, in_domain


# ---------------------------------------------------------------------------
# Single-particle Numba helpers (for fused integrator kernels)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _local_nearest_node(
    lon_p: float,
    lat_p: float,
    old_nn: int,
    node_to_elems: np.ndarray,
    node_to_elems_count: np.ndarray,
    elem_nodes: np.ndarray,
    nodes_lon: np.ndarray,
    nodes_lat: np.ndarray,
) -> int:
    """Find nearest node by searching the old node and its topological neighbours.

    Collects all nodes from elements adjacent to ``old_nn`` (~15-20 candidates)
    and returns the closest one using scaled Euclidean distance.  O(1) per
    particle — replaces the scipy KD-tree call for post-displacement checks.

    Parameters
    ----------
    lon_p, lat_p : float
        Query point in degrees.
    old_nn : int
        Pre-displacement nearest node index.

    Returns
    -------
    int
        Index of the locally nearest node.
    """
    cos_lat = np.cos(lat_p * DEG2RAD)

    best = old_nn
    dlon = ((lon_p - nodes_lon[old_nn]) + 180.0) % 360.0 - 180.0
    best_d2 = (dlon * cos_lat) ** 2 + (lat_p - nodes_lat[old_nn]) ** 2

    count = int(node_to_elems_count[old_nn])
    for k in range(count):
        e = int(node_to_elems[old_nn, k])
        if e < 0:
            continue
        for v in range(elem_nodes.shape[1]):
            n = int(elem_nodes[e, v])
            if n < 0 or n == best:
                continue
            dlon_n = ((lon_p - nodes_lon[n]) + 180.0) % 360.0 - 180.0
            d2 = (dlon_n * cos_lat) ** 2 + (lat_p - nodes_lat[n]) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = n

    return best


@njit(cache=True)
def _find_containing_element_local(
    lon_p: float,
    lat_p: float,
    nn_node: int,
    node_to_elems: np.ndarray,
    node_to_elems_count: np.ndarray,
    elem_nodes: np.ndarray,
    nodes_lon: np.ndarray,
    nodes_lat: np.ndarray,
) -> tuple[int, bool]:
    """Single-particle domain check: find the containing element around a node.

    Scans the elements adjacent to ``nn_node`` (1-ring) first.  On a miss
    it falls back to the 2-ring — the elements adjacent to every vertex
    of the 1-ring elements.  On irregular (e.g. jittered) triangulations
    the nearest node to a point is not always a vertex of the containing
    element, so the 1-ring scan alone produces false OUTSIDE verdicts
    (see tests/test_jitter_regression.py).  The fallback only runs on the
    rare 1-ring miss, so the common path stays as cheap as before.

    Known limitation: on extremely distorted meshes (node perturbation
    approaching half the grid spacing) the containing element can lie
    beyond the 2-ring; an element-adjacency walk would be needed for
    full generality.

    Returns ``(elem_id, in_domain)``.
    """
    count = int(node_to_elems_count[nn_node])
    if count == 0:
        return -1, False

    # --- Pass 1: elements adjacent to the nearest node (1-ring) ---------
    for k in range(count):
        c = int(node_to_elems[nn_node, k])
        if c < 0:
            continue
        if _element_contains(c, lon_p, lat_p, elem_nodes,
                             nodes_lon, nodes_lat):
            return c, True

    # --- Pass 2: 2-ring fallback ----------------------------------------
    # Scan the elements adjacent to every vertex of every 1-ring element.
    # Duplicate element checks are accepted (this path only runs on the
    # rare 1-ring miss) — no fixed-size scratch buffer that could
    # silently truncate on high-valence nodes.
    for k in range(count):
        e = int(node_to_elems[nn_node, k])
        if e < 0:
            continue
        for j in range(elem_nodes.shape[1]):
            v = int(elem_nodes[e, j])
            if v < 0 or v == nn_node:
                continue
            count_v = int(node_to_elems_count[v])
            for kk in range(count_v):
                c = int(node_to_elems[v, kk])
                if c < 0:
                    continue
                # Skip elements already tested in pass 1: those are
                # exactly the elements that have nn_node as a vertex.
                touches_nn = False
                for jj in range(elem_nodes.shape[1]):
                    if int(elem_nodes[c, jj]) == nn_node:
                        touches_nn = True
                        break
                if touches_nn:
                    continue
                if _element_contains(c, lon_p, lat_p, elem_nodes,
                                     nodes_lon, nodes_lat):
                    return c, True

    return -1, False


# ---------------------------------------------------------------------------
# Edge-walk containing-element search (elem_search: walk)
# ---------------------------------------------------------------------------

#: iteration cap for the edge walk — generous vs the 0-3 crossings a real
#: CFL-bounded step produces; exceeding it falls back to the global search.
WALK_MAX_STEPS = 256
#: relative accept tolerance on the normalized signed-area coordinates —
#: the same tolerance class as ``_BARY_EPS`` (both are dimensionless,
#: area-normalized), keeping walk accept semantics aligned with the fan
#: containment test.
_WALK_EPS = 1e-6
#: normalized margin above which a walk accept skips the exact-predicate
#: verification: at >= 1e-2 the point sits >= ~5e-3*L from every edge
#: (metric), while the walk's and the local test's projections disagree by
#: only O((L/R)^2 * L) — ~25 m for 100-km elements, safe up to ~800-km
#: cells.  Only the thin band [-_WALK_EPS, _WALK_VERIFY_BAND) pays the
#: verification, so the accept fast path stays one ring test.
_WALK_VERIFY_BAND = 1e-2


@njit(cache=True)
def _walk_containing_element(
    lon_p: float,
    lat_p: float,
    start_elem: int,
    elem_nodes: np.ndarray,
    nodes_lon: np.ndarray,
    nodes_lat: np.ndarray,
    elem_neighbors: np.ndarray,
    max_steps: int = WALK_MAX_STEPS,
) -> tuple[int, bool]:
    """Find the containing element by walking across edges from a start guess.

    Per visited element: project the compacted vertex ring and the point to
    a local metric plane (``_project_to_local_xy`` — dateline-safe), form
    the signed area ``A_k`` of triangle ``(v_k, v_{k+1}, p)`` per edge
    (winding-corrected by the polygon's own signed area) and normalize by
    the polygon area.  All normalized coordinates ``>= -_WALK_EPS`` means
    the point is inside (same tolerance class as the fan containment test);
    otherwise the point left through the MOST NEGATIVE edge — cross to
    ``elem_neighbors[e, k]`` and repeat.

    Returns ``(elem_id, True)`` when contained.  Every non-contained
    outcome returns ``(-1, False)`` and the CALLER must fall back to the
    global search — this covers: a boundary edge crossed (the point may
    still be inside across a bay: concave domain boundary), the iteration
    cap, a degenerate cell, re-crossing the same shared edge (per-element
    projection noise on near-edge points), or an invalid start.  The walk
    is therefore a pure fast path: containment VERDICTS are identical to
    the global machinery.
    """
    e = int(start_elem)
    n_elems = elem_nodes.shape[0]
    if e < 0 or e >= n_elems:
        return -1, False
    prev = -1
    max_v = elem_nodes.shape[1]
    verts = np.empty(max_v, dtype=np.int64)
    for _ in range(max_steps):
        # Compacted vertex ring (idiom of the interp kernels /
        # build_elem_neighbors: skip -1 pads and repeats of the first
        # vertex) — slot k below matches elem_neighbors slot k.
        nv = 0
        for k in range(max_v):
            nd = int(elem_nodes[e, k])
            if nd < 0:
                continue
            if nv > 0 and nd == verts[0]:
                continue
            verts[nv] = nd
            nv += 1
        if nv < 3:
            return -1, False
        lon_n = nodes_lon[verts[:nv]]
        lat_n = nodes_lat[verts[:nv]]
        x_n, y_n, px, py = _project_to_local_xy(lon_n, lat_n, lon_p, lat_p)

        # Polygon signed area (x2, shoelace): winding sign + scale.
        S = 0.0
        for k in range(nv):
            j = (k + 1) % nv
            S += x_n[k] * y_n[j] - x_n[j] * y_n[k]
        absS = abs(S)
        if absS < _AREA_TOL:
            return -1, False
        sgn = 1.0 if S > 0.0 else -1.0

        # worst = the MINIMUM normalized coordinate (not just the largest
        # violation): interior points then carry their true edge margin,
        # which the verify-band test below needs.
        worst = 1.0e100
        worst_k = -1
        for k in range(nv):
            j = (k + 1) % nv
            A_k = ((x_n[j] - x_n[k]) * (py - y_n[k])
                   - (px - x_n[k]) * (y_n[j] - y_n[k]))
            w = A_k * sgn / absS
            if w < worst:
                worst = w
                worst_k = k
        if worst >= -_WALK_EPS:
            if worst >= _WALK_VERIFY_BAND:
                # Comfortably interior: the two predicates provably agree
                # (margin >> projection disagreement) — skip the verify.
                return e, True
            # Near-edge accept: verify with the EXACT local predicate
            # (strict bbox prefilter + fan test).  The signed-area accept
            # admits a ~_WALK_EPS-normalized OUTSIDE band that the local
            # machinery rejects at boundary edges — a life/death verdict
            # divergence, not an on-edge tie (review 2026-07-02).
            # Any disagreement falls back to the global path, which
            # reproduces the local verdict exactly.
            if _element_contains(e, lon_p, lat_p, elem_nodes,
                                 nodes_lon, nodes_lat):
                return e, True
            return -1, False
        nb = int(elem_neighbors[e, worst_k])
        if nb < 0:
            return -1, False
        if nb == prev:
            return -1, False
        prev = e
        e = nb
    return -1, False


@njit(parallel=True, cache=True)
def _loop_walk_containing(
    P, lons, lats, starts,
    elem_nodes, nodes_lon, nodes_lat, node_xyz, elem_neighbors,
    gh_offsets, gh_node_ids, gh_cell_size, gh_lon_min, gh_lat_min,
    gh_nx, gh_ny, gh_wrap_lon,
    node_to_elems, node_to_elems_count,
    elem_ids, in_domain,
):
    for i in prange(P):
        e0 = int(starts[i])
        if e0 >= 0:
            e, ok = _walk_containing_element(
                lons[i], lats[i], e0,
                elem_nodes, nodes_lon, nodes_lat, elem_neighbors,
                WALK_MAX_STEPS)
            if ok:
                elem_ids[i] = e
                in_domain[i] = True
                continue
        # Fallback: the existing global machinery (grid-hash nearest node
        # + 1/2-ring containment scan) — identical verdicts to
        # find_containing_elements.
        nn = _grid_hash_nearest_widened(
            lons[i], lats[i],
            gh_offsets, gh_node_ids, gh_cell_size,
            gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
            nodes_lon, nodes_lat, node_xyz)
        e, ok = _find_containing_element_local(
            lons[i], lats[i], nn,
            node_to_elems, node_to_elems_count,
            elem_nodes, nodes_lon, nodes_lat)
        if ok:
            elem_ids[i] = e
            in_domain[i] = True


def find_containing_elements_walk(
    mesh: Mesh,
    lons: np.ndarray,
    lats: np.ndarray,
    start_elems: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Batch containing-element lookup via the edge walk, with fallback.

    Walks from each particle's carried element (``start_elems``); any
    non-contained walk outcome (boundary crossing, cap, invalid start,
    degenerate cell) falls back to the global search for that particle,
    so the returned verdicts are identical to
    :func:`find_containing_elements` wherever the local 1/2-ring scan is
    itself exhaustive — all realistic FESOM meshes (up to on-edge ties,
    where the containing element is not unique).  On extremely distorted
    triangulations the walk finds beyond-2-ring containing elements the
    local scan falsely rejects (strictly more complete).  In no-numba
    environments the FALLBACK anchors on the grid hash while
    ``find_nearest_nodes`` falls back to the KD-tree — near-tie anchors
    can differ, so pure-Python runs may rarely diverge from ``local`` for
    off-mesh queries.

    Parameters
    ----------
    mesh : Mesh
        Mesh with search structures built (``build_search``); the
        edge-adjacency table is built lazily on first call.
    lons, lats : np.ndarray, shape (P,)
        Query positions (degrees).
    start_elems : np.ndarray, shape (P,)
        Carried element per particle; ``-1`` = no guess (global search).

    Returns
    -------
    elem_ids : np.ndarray, shape (P,), int32
        Containing element, -1 if outside.
    in_domain : np.ndarray of bool, shape (P,)
    """
    lons = np.asarray(lons, dtype=np.float64)
    lats = np.asarray(lats, dtype=np.float64)
    starts = np.asarray(start_elems, dtype=np.int32)
    P = lons.shape[0]
    if starts.shape[0] != P:
        raise ValueError(
            f"start_elems length {starts.shape[0]} != positions length {P}")
    elem_ids = np.full(P, -1, dtype=np.int32)
    in_domain = np.zeros(P, dtype=np.bool_)
    if P == 0:
        return elem_ids, in_domain
    neighbors = build_elem_neighbors(mesh)
    if mesh._gh_offsets is None:
        build_grid_hash(mesh)
    node_xyz = _require_node_xyz(mesh)
    _loop_walk_containing(
        P, lons, lats, starts,
        mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat, node_xyz, neighbors,
        mesh._gh_offsets, mesh._gh_node_ids, mesh._gh_cell_size,
        mesh._gh_lon_min, mesh._gh_lat_min, mesh._gh_nx,
        mesh._gh_ny, mesh._gh_wrap_lon,
        mesh.node_to_elems, mesh.node_to_elems_count,
        elem_ids, in_domain,
    )
    # Exactness heal (review 2026-08-16): the in-loop fallback anchors on
    # the numba-only WIDENED nearest (no KD-tree is reachable inside the
    # loop); on empty-window geometries that anchor is approximate and
    # its ring scan can miss an element the KD-exact anchor would find.
    # Re-run the not-contained remainder through the exact global
    # machinery so the returned verdicts stay identical-or-more-complete
    # vs find_containing_elements (the documented walk contract).
    miss = ~in_domain
    if np.any(miss):
        idx = np.where(miss)[0]
        nn_exact = find_nearest_nodes(mesh, lons[idx], lats[idx])
        e2, ok2 = find_containing_elements(mesh, lons[idx], lats[idx],
                                           nn_exact)
        elem_ids[idx] = np.where(ok2, e2, elem_ids[idx])
        in_domain[idx] = ok2
    return elem_ids, in_domain


# ---------------------------------------------------------------------------
# Topology builder (used by readers)
# ---------------------------------------------------------------------------

@njit(cache=True)
def build_node_to_elems(n_nodes: int, elem_nodes: np.ndarray):
    """Build node-to-element adjacency from element connectivity.

    Parameters
    ----------
    n_nodes : int
        Total number of nodes.
    elem_nodes : np.ndarray
        Element connectivity, shape (E, max_verts), -1 padded.

    Returns
    -------
    node_to_elems : np.ndarray
        Shape (N, max_conn), -1 padded.
    node_to_elems_count : np.ndarray
        Shape (N,).
    """
    n_elems = elem_nodes.shape[0]
    max_verts = elem_nodes.shape[1]

    # First pass: count
    counts = np.zeros(n_nodes, dtype=np.int32)
    for e in range(n_elems):
        seen = set()
        for v in range(max_verts):
            n = elem_nodes[e, v]
            if n < 0 or n in seen:
                continue
            seen.add(n)
            counts[n] += 1

    max_conn = int(counts.max()) if n_nodes > 0 else 0
    node_to_elems = np.full((n_nodes, max_conn), -1, dtype=np.int32)
    fill = np.zeros(n_nodes, dtype=np.int32)

    # Second pass: fill
    for e in range(n_elems):
        seen = set()
        for v in range(max_verts):
            n = elem_nodes[e, v]
            if n < 0 or n in seen:
                continue
            seen.add(n)
            node_to_elems[n, fill[n]] = e
            fill[n] += 1

    return node_to_elems, counts


def build_boundary_info(mesh: Mesh) -> None:
    """Flag coastal elements: any element owning a boundary edge.

    A boundary edge belongs to exactly one element.  Handles triangles,
    quads, and FESOM-C padding (-1 fill / repeated first vertex) by
    building each element's edge ring over its COMPACTED valid vertices —
    the closing edge (last valid -> first valid) is always emitted.
    Island holes produce interior boundary loops — same machinery.
    Cached on the mesh; idempotent.  Deliberately NOT @njit: the
    tuple-append loop + np.unique(return_inverse) aren't numba-friendly
    and this runs once at startup (~3.7 s for 1M elements).

    Parameters
    ----------
    mesh : Mesh
        Mesh whose ``_elem_is_coastal`` flag array is populated in place.
        No-op if it is already set.
    """
    if mesh._elem_is_coastal is not None:
        return
    E = mesh.n_elements
    if E == 0:
        mesh._elem_is_coastal = np.zeros(0, dtype=np.bool_)
        return
    rows = []      # (min_node, max_node, elem_id) per edge
    for e in range(E):
        verts = mesh.elem_valid_nodes(e)
        nv = len(verts)
        # A real polygon ring needs >= 3 distinct vertices.  A degenerate
        # 2-vertex element [a, b, -1, -1] would otherwise emit (a, b) AND
        # (b, a) — the SAME undirected edge twice from ONE element — pushing
        # its count to 2 and hiding a genuine boundary edge that a real
        # triangle shares with it (false interior).  Skip such elements.
        if nv < 3:
            continue
        # Per-element undirected-edge dedup: guards against a malformed ring
        # repeating an edge within a single element (e.g. degenerate
        # compaction) inflating the global count for the same reason.
        seen = set()
        for j in range(nv):
            a = int(verts[j]); b = int(verts[(j + 1) % nv])
            if a == b:
                continue
            ek = (a, b) if a < b else (b, a)
            if ek in seen:
                continue
            seen.add(ek)
            rows.append((ek[0], ek[1], e))
    coastal = np.zeros(E, dtype=np.bool_)
    if rows:
        arr = np.asarray(rows, dtype=np.int64)
        # radix n_nodes+1 > max node id => (a, b) packs collision-free
        # into int64 (11.5M-node global mesh: max key ~1.3e14 << 2**63)
        key = arr[:, 0] * np.int64(mesh.n_nodes + 1) + arr[:, 1]
        _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        coastal[arr[cnt[inv] == 1, 2]] = True
    mesh._elem_is_coastal = coastal


def build_elem_neighbors(mesh: Mesh) -> np.ndarray:
    """Element adjacency across shared edges (FESOM's ``elem_neighbors``).

    Slot ``k`` of row ``e`` holds the element sharing the edge
    ``(verts[k], verts[(k+1) % nv])`` of element ``e``'s COMPACTED vertex
    ring — the same compaction the interpolation kernels use (``-1`` pads
    and the FESOM-C repeated-first-vertex slot are dropped) — or ``-1``
    when that edge is a boundary edge.  Unused trailing slots (a triangle
    stored in a 4-wide row) also stay ``-1``: consumers derive ``nv`` from
    ``elem_nodes`` and never read past slot ``nv - 1``.

    A triangle therefore has up to 3 neighbors, a quad up to 4 — the
    stencil of the FESOM-style edge-neighbor least-squares interpolation
    (and the side-lookup table for edge-walk element tracking).

    An edge owned by more than two elements would make the mesh
    non-manifold (impossible in FESOM/FESOM-C output) and raises
    ``ValueError`` — fail fast rather than silently mis-pairing.

    Lazy/idempotent: cached as ``mesh._elem_neighbors`` (pattern of
    ``build_boundary_info``).  Deliberately NOT ``@njit``: one-time
    startup cost, same order as ``build_boundary_info``.

    Parameters
    ----------
    mesh : Mesh

    Returns
    -------
    np.ndarray, shape (E, max_verts), int32
        Neighbor element id per local edge slot, -1 = boundary/unused.
    """
    if mesh._elem_neighbors is not None:
        return mesh._elem_neighbors
    E = mesh.n_elements
    max_verts = mesh.elem_nodes.shape[1] if E > 0 else 0
    out = np.full((E, max_verts), -1, dtype=np.int32)
    if E == 0:
        mesh._elem_neighbors = out
        return out
    en = mesh.elem_nodes
    rows = []      # (min_node, max_node, elem_id, local_edge_slot)
    for e in range(E):
        verts = []
        for j in range(max_verts):
            nd = int(en[e, j])
            if nd < 0:
                continue
            if verts and nd == verts[0]:
                continue          # FESOM-C tri stored as quad repeats vertex 0
            verts.append(nd)
        nv = len(verts)
        if nv < 3:
            continue              # degenerate row: no polygon ring
        seen = set()
        for k in range(nv):
            a = verts[k]; b = verts[(k + 1) % nv]
            if a == b:
                continue
            ek = (a, b) if a < b else (b, a)
            if ek in seen:
                continue          # malformed ring repeating an edge
            seen.add(ek)
            rows.append((ek[0], ek[1], e, k))
    if rows:
        arr = np.asarray(rows, dtype=np.int64)
        # radix n_nodes+1 > max node id => (a, b) packs collision-free
        key = arr[:, 0] * np.int64(mesh.n_nodes + 1) + arr[:, 1]
        order = np.argsort(key, kind="stable")
        key_s = key[order]
        arr_s = arr[order]
        starts = np.flatnonzero(np.r_[True, key_s[1:] != key_s[:-1]])
        counts = np.diff(np.r_[starts, key_s.size])
        if (counts > 2).any():
            bad = arr_s[starts[counts > 2][0]]
            raise ValueError(
                f"non-manifold mesh: edge ({bad[0]}, {bad[1]}) is shared "
                "by more than two elements"
            )
        two = starts[counts == 2]
        e1, k1 = arr_s[two, 2], arr_s[two, 3]
        e2, k2 = arr_s[two + 1, 2], arr_s[two + 1, 3]
        out[e1, k1] = e2
        out[e2, k2] = e1
    mesh._elem_neighbors = out
    return out


#: relative corner-quality floor for the load-time convexity check: a corner
#: cross-product below this fraction of the element scale is "flat" —
#: indistinguishable from degenerate for the product-form Wachspress weights
#: (docs/interpolation.md §7: the safety margin over the containment
#: tolerance is proportional to the worst normalized corner quality).
_QUAD_CONVEXITY_TOL = 1e-9


def find_nonconvex_quads(elem_nodes: np.ndarray,
                         nodes_lon: np.ndarray,
                         nodes_lat: np.ndarray) -> np.ndarray:
    """Ids of quad elements violating the convexity mesh contract.

    The product-form Wachspress weights (guard-free since 2026-07) assume
    every quad is strictly convex and non-degenerate — a violating quad
    silently produces NaN or garbage vertex weights.  This one-time O(E)
    vectorized check lets readers fail LOUD at load instead.

    A row is a *quad* only when it has 4 valid, distinct vertices (the
    FESOM-C tri encodings — ``-1`` pad or repeated first vertex — are
    skipped, matching the kernels' ring compaction).  A quad is flagged
    when its four corner cross-products (in a seam-safe local metric,
    longitudes anchored to the first vertex by wrapped deltas) are not all
    of one sign, or any corner is flat below a relative tolerance
    (collinear/degenerate).  Winding-agnostic: CW quads pass.

    Parameters
    ----------
    elem_nodes : np.ndarray, shape (E, max_verts)
        0-based connectivity, -1 padded.
    nodes_lon, nodes_lat : np.ndarray, shape (N,)
        Node coordinates in degrees.

    Returns
    -------
    np.ndarray
        Sorted int64 ids of the offending quad elements (empty = contract
        holds).
    """
    en = np.asarray(elem_nodes)
    if en.shape[0] == 0 or en.shape[1] < 4:
        return np.empty(0, dtype=np.int64)
    q = en[:, :4]
    is_quad = ((q >= 0).all(axis=1)
               & (q[:, 3] != q[:, 0]) & (q[:, 1] != q[:, 0])
               & (q[:, 2] != q[:, 0]) & (q[:, 2] != q[:, 1])
               & (q[:, 3] != q[:, 1]) & (q[:, 3] != q[:, 2]))
    if not is_quad.any():
        return np.empty(0, dtype=np.int64)
    ids = np.flatnonzero(is_quad)
    v = q[ids]                                  # (Q, 4)
    lon = np.deg2rad(np.asarray(nodes_lon, dtype=np.float64)[v])
    lat = np.deg2rad(np.asarray(nodes_lat, dtype=np.float64)[v])
    # Tangent-plane projection via 3D unit vectors (review 2026-07-05: a
    # lon/lat chart anchored on vertex 0 is not topology-preserving for
    # cells spanning >180 deg from the anchor — legitimate near the pole,
    # where cos(lat) also collapses the x-scale).  3D is seam- AND
    # pole-proof: centroid direction C, east/north basis in the tangent
    # plane at C (x-axis fallback when C is polar), vertices projected.
    cl = np.cos(lat)
    V3 = np.stack((cl * np.cos(lon), cl * np.sin(lon), np.sin(lat)),
                  axis=-1)                       # (Q, 4, 3)
    C = V3.mean(axis=1)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    # e_east = normalize(z_hat x C); degenerate (polar centroid) -> use
    # x_hat x C instead.
    zxC = np.stack((-C[:, 1], C[:, 0], np.zeros(len(C))), axis=-1)
    nrm = np.linalg.norm(zxC, axis=1, keepdims=True)
    polar = (nrm[:, 0] < 1e-9)
    if polar.any():
        xxC = np.stack((np.zeros(len(C)), -C[:, 2], C[:, 1]), axis=-1)
        zxC[polar] = xxC[polar]
        nrm = np.linalg.norm(zxC, axis=1, keepdims=True)
    e1 = zxC / nrm
    e2 = np.cross(C, e1)
    x = np.einsum("qvk,qk->qv", V3, e1)
    y = np.einsum("qvk,qk->qv", V3, e2)
    # Corner cross products z_k = (v_{k+1}-v_k) x (v_{k+2}-v_{k+1}).
    ex = np.roll(x, -1, axis=1) - x
    ey = np.roll(y, -1, axis=1) - y
    cross = ex * np.roll(ey, -1, axis=1) - ey * np.roll(ex, -1, axis=1)
    scale = (ex * ex + ey * ey).max(axis=1)     # ~ (element edge)^2
    flat = np.abs(cross) <= _QUAD_CONVEXITY_TOL * scale[:, None]
    sign_ok = (cross > 0).all(axis=1) | (cross < 0).all(axis=1)
    bad = ~sign_ok | flat.any(axis=1)
    return np.sort(ids[bad]).astype(np.int64)


def validate_quad_convexity(elem_nodes, nodes_lon, nodes_lat,
                            *, source: str = "mesh") -> None:
    """Raise ``ValueError`` when any quad violates the convexity contract.

    Thin load-time wrapper over :func:`find_nonconvex_quads` used by the
    readers; lists the count and the first offending element ids.
    """
    bad = find_nonconvex_quads(elem_nodes, nodes_lon, nodes_lat)
    if bad.size:
        head = ", ".join(str(int(b)) for b in bad[:10])
        more = f" (+{bad.size - 10} more)" if bad.size > 10 else ""
        raise ValueError(
            f"{source}: {bad.size} quad element(s) are non-convex or "
            f"degenerate — element ids [{head}]{more}.  Convex quads are a "
            "mesh contract (the Wachspress vertex weights are guard-free); "
            "fix the mesh file or its connectivity.")


def build_elem_areas(mesh: Mesh) -> np.ndarray:
    """Element areas (m^2) via spherical excess — pole-safe, lazy cache.

    When the reader supplied **model-provided** areas (``mesh._elem_areas``
    pre-filled from a FESOM2 mesh-diag ``elem_area`` / a FESOM-C NetCDF
    area variable) this returns them untouched — bit-parity with the model
    and pole-correct by construction.  Otherwise it computes the area on
    the unit sphere as a sum of spherical-excess triangle areas of the
    vertex fan, scaled by ``mesh.r_earth**2``.

    Spherical excess uses 3D unit vectors, so it is **frame-invariant**
    (no projection plane, no dateline anchoring) and exact at the poles —
    unlike the previous planar ``cos(lat)`` shoelace, which collapsed to
    near-zero area on polar caps.  Each fan triangle ``(v0, vk, vk+1)``
    contributes ``E = 2*atan2(|a·(b×c)|, 1 + a·b + b·c + a·c)`` (the
    numerically stable l'Huilier-equivalent form); the fan split is exact
    for convex cells on the sphere at FESOM element scales.

    FESOM-C connectivity quirks are absorbed by mapping ``-1`` pads and the
    repeated-first-vertex slot to the first vertex: the corresponding fan
    triangle then has two coincident vertices, its triple product is zero,
    so it contributes zero excess — padded and repeated-vertex triangle
    rows give identical areas.  Vectorised numpy over all elements and both
    fan triangles (no per-row Python loop); one-time cost, cached as
    ``mesh._elem_areas`` (pattern of ``build_boundary_info``).

    Areas are consumed only as RATIOS of neighbouring elements (the
    area-weighted element->node velocity projection), so the radius choice
    cancels there.

    Parameters
    ----------
    mesh : Mesh
        Mesh whose ``elem_nodes`` / node coordinates define the elements.

    Returns
    -------
    np.ndarray, shape (E,)
        Element areas in m^2 (always non-negative), cached on the mesh.
        Model-provided areas are returned unchanged when present.
    """
    if mesh._elem_areas is not None:
        return mesh._elem_areas
    en = mesh.elem_nodes
    first = en[:, 0:1]
    # Map -1 pads AND the FESOM-C repeated-first-vertex slot to the first
    # vertex: a fan triangle that reuses v0 collapses (two coincident
    # vertices => zero triple product => zero excess), so padded and
    # repeated rows contribute identically.
    idx = np.where(en >= 0, en, first)
    max_v = en.shape[1]
    # Per-column 3D unit vectors (E, max_v, 3) — frame-invariant, no
    # projection plane.
    lat_r = mesh.nodes_lat[idx] * DEG2RAD           # (E, max_v)
    lon_r = mesh.nodes_lon[idx] * DEG2RAD
    cos_lat = np.cos(lat_r)
    V = np.stack((cos_lat * np.cos(lon_r),
                  cos_lat * np.sin(lon_r),
                  np.sin(lat_r)), axis=-1)           # (E, max_v, 3)

    def _excess(a, b, c):
        # 2*atan2(|a.(bxc)|, 1 + a.b + b.c + a.c) per row, vectorised.
        triple = np.abs(np.einsum("ij,ij->i", a, np.cross(b, c)))
        ab = np.einsum("ij,ij->i", a, b)
        bc = np.einsum("ij,ij->i", b, c)
        ac = np.einsum("ij,ij->i", a, c)
        return 2.0 * np.arctan2(triple, 1.0 + ab + bc + ac)

    v0 = V[:, 0, :]
    exc = np.zeros(en.shape[0], dtype=np.float64)
    # Fan triangles (v0, v_k, v_{k+1}) for k = 1 .. max_v-2.  Tris
    # (max_v == 3) -> one triangle; quads -> two; a padded/repeat 4th
    # column collapses its triangle to zero excess (handled above).
    for k in range(1, max_v - 1):
        exc += _excess(v0, V[:, k, :], V[:, k + 1, :])
    mesh._elem_areas = exc * (mesh.r_earth ** 2)
    return mesh._elem_areas


__all__ = [
    "Mesh",
    "build_search",
    "build_grid_hash",
    "grid_hash_dense_for_elements",
    "find_nearest_nodes",
    "find_containing_elements",
    "_grid_hash_nearest_widened",
    "_grid_hash_nearest_batch",
    "_grid_hash_nearest_exact",
    "_anchor_provable",
    "_anchor_provable_rect",
    "_gh_scan_cell",
    "_gh_scan_row_range",
    "_gh_scan_row_window",
    "_gh_far_rows_for",
    "_local_nearest_node",
    "_find_containing_element_local",
    "_wachspress_weights_quad",
    "_polygon_weights",
    "_polygon_contains",
    "_polygon_contains_bool",
    "build_node_to_elems",
    "build_boundary_info",
    "build_elem_areas",
    "build_elem_neighbors",
    "find_nonconvex_quads",
    "validate_quad_convexity",
    "_walk_containing_element",
    "find_containing_elements_walk",
    "WALK_MAX_STEPS",
]
