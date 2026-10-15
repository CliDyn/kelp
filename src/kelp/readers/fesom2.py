"""FESOM2 model reader.

Reads mesh from ASCII files (nod2d.out, elem2d.out), NetCDF
(fesom.mesh.diag.nc), or Zarr (fesom.mesh.diag.region.zarr).
Velocity is loaded from NetCDF or Zarr files.
"""
from __future__ import annotations

import time as _time
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import xarray as xr

from ..core.config import MeshConfig, ModeConfig
from ..core.io import VelocityField
from ..core.mesh import Mesh, build_node_to_elems, build_search
from . import (
    register,
    _detect_engine,
    _read_kh_fields,
    resolve_w_sign,
    _detect_field_layout,
    _classify_layout_meta,
    _align_onto_time_axis,
)

#: Appended to a refused W attachment: the usual cause.
_W_FILE_HINT = ("A W written by utils/reconstruct_w.py before 2026-10-04 "
                "carries relative seconds as its time axis and must be "
                "regenerated.")

#: Mesh latitudes beyond this (degrees) trigger the near-pole warning when
#: rotation is OFF (the lon/lat step metric degenerates near the pole).
POLE_WARN_LAT = 88.0


def _resolve_rotation_angles(mesh_cfg):
    """Resolve ``mesh_cfg.rotation`` to a concrete Euler-angle tuple or None.

    ``None`` -> geographic frame (no rotation); ``"fesom2"`` -> the standard
    FESOM2 angles; an explicit 3-sequence -> a float tuple.
    """
    rotation = getattr(mesh_cfg, "rotation", None)
    if rotation is None:
        return None
    from ..core.rotation import FESOM2_ANGLES
    if rotation == "fesom2":
        return FESOM2_ANGLES
    return tuple(float(x) for x in rotation)


def _apply_mesh_rotation(nodes_lon, nodes_lat, elem_center_lon,
                         elem_center_lat, rotation_angles):
    """Rotate node + element-centre coordinates AS POINTS into the frame.

    The per-carrier vector-rotation factors are computed FIRST from the
    GEOGRAPHIC carriers (nodes / element centres) — they need the geographic
    tangent basis — then all stored coordinates are rotated as points (the
    rotated mesh is the same discrete object in another chart; centres are
    NOT recomputed from rotated nodes — the circular-mean centroid is not
    rotation-equivariant).

    Returns ``(nodes_lon_r, nodes_lat_r, elem_center_lon_r,
    elem_center_lat_r, vec_rot_factors)``.
    """
    from ..core.rotation import euler_matrix, rotate_lonlat, rotate_vectors
    R = euler_matrix(*rotation_angles)
    # vector factors FIRST (need geographic carriers): rotating unit-east
    # gives (c, s) of the local tangent-plane rotation; the full map is
    # [[c, -s], [s, c]] (unit-north -> (-s, c)).
    c_node, s_node = rotate_vectors(
        np.ones_like(nodes_lon), np.zeros_like(nodes_lon),
        nodes_lon, nodes_lat, R)
    c_elem, s_elem = rotate_vectors(
        np.ones_like(elem_center_lon), np.zeros_like(elem_center_lon),
        elem_center_lon, elem_center_lat, R)
    vec_rot_factors = {"node": (c_node, s_node),
                       "element": (c_elem, s_elem)}
    # then rotate ALL stored coordinates as points
    nodes_lon_r, nodes_lat_r = rotate_lonlat(nodes_lon, nodes_lat, R)
    elem_center_lon_r, elem_center_lat_r = rotate_lonlat(
        elem_center_lon, elem_center_lat, R)
    print(f"    rotated frame: alpha/beta/gamma = "
          f"{tuple(rotation_angles)} (pole-safe)")
    return (nodes_lon_r, nodes_lat_r, elem_center_lon_r, elem_center_lat_r,
            vec_rot_factors)


def _warn_near_pole_fesom2(lat_geo, rotation_angles):
    """Emit the near-pole warning (rotation OFF) and the exact-pole guard.

    ``lat_geo`` are the GEOGRAPHIC node latitudes (before any rotation).
    The metric-degeneracy warning fires only when rotation is OFF; the
    exact-pole guard fires (rotation ON) when a node sits at |lat| = 90,
    where the data's own u/v decomposition is ill-defined.
    """
    import warnings
    max_abs_lat = float(np.abs(lat_geo).max())
    if rotation_angles is None and max_abs_lat > POLE_WARN_LAT:
        warnings.warn(
            f"mesh reaches |lat| = {max_abs_lat:.2f} deg - lon/lat tracking "
            "degenerates near the pole; set mesh.rotation: fesom2 to track "
            "in the model's rotated frame.",
            RuntimeWarning, stacklevel=2)
    if rotation_angles is not None and (
            np.abs(np.abs(lat_geo) - 90.0) < 1e-9).any():
        warnings.warn(
            "mesh contains a node at the geographic pole - the file's own "
            "u/v decomposition is ill-defined there",
            RuntimeWarning, stacklevel=2)


def _resolve_uv_names(ds, mode_cfg):
    """Return the (u_name, v_name) FESOM2 ``read_velocities`` will read.

    Honours ``mode_cfg.var_u`` / ``var_v`` when set (so node-projected
    velocities ``unod`` / ``vnod`` can be selected), else the defaults
    ``"u"`` / ``"v"``.  Same logic in 2D and 3D mode.  Shared with
    :meth:`Fesom2Reader.probe_uv_layout` so probe and read can never
    disagree on which variables are classified.
    """
    cfg_var_u = getattr(mode_cfg, "var_u", None) if mode_cfg else None
    cfg_var_v = getattr(mode_cfg, "var_v", None) if mode_cfg else None
    return (cfg_var_u or "u"), (cfg_var_v or "v")


_VERT_DIMS = ("nz1", "nz", "depth", "level")


def _az_var_is_mid_layer(ds, var_az):
    """True when ``var_az`` sits on mid-layers instead of interfaces.

    Canonical FESOM2 naming: the ``nz1`` dim.  Non-canonical vertical
    dim names (``depth``/``level``) are classified by PRE-TRIM size
    against the dataset's interface dim ``nz`` when present — a
    ``max_levels`` trim would otherwise let a (K-1) variable slip past
    the trimmed-shape check half-a-layer shifted (review 2026-08-15).
    """
    dims = ds[var_az].dims
    if "nz1" in dims:
        return True
    vdim = next((d for d in ("depth", "level") if d in dims), None)
    if vdim is not None and "nz" in ds.dims:
        return ds.sizes[vdim] == ds.sizes["nz"] - 1
    return False


def _orient_vert_first(da):
    """Return ``da`` with its vertical dim as axis 0 (``(K, N)`` layout).

    A 3D node/element variable can be stored either ``(vert, spatial)`` or
    ``(spatial, vert)`` after the time slice.  The kernels assume vertical
    first, so transpose by dim NAME when the vertical dim is not already
    axis 0.  No-op for 2D (no vertical dim) arrays.
    """
    vdim = next((d for d in _VERT_DIMS if d in da.dims), None)
    if vdim is None or da.dims.index(vdim) == 0:
        return da
    spatial = next(d for d in da.dims if d != vdim)
    return da.transpose(vdim, spatial)


# ---------------------------------------------------------------------------
# ASCII mesh readers
# ---------------------------------------------------------------------------

def _read_nod2d(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read node coordinates from ``nod2d.out``.

    Format::
        <n_nodes>
        id  lon  lat  flag
        ...
    """
    lines = path.read_text().strip().split("\n")
    data = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 3:
            data.append((float(parts[1]), float(parts[2])))
    arr = np.array(data, dtype=np.float64)
    return arr[:, 0], arr[:, 1]


def _read_elem2d(path: Path) -> np.ndarray:
    """Read triangular element connectivity from ``elem2d.out``.

    Format::
        <n_elements>
        n1  n2  n3
        ...

    Returns 0-based connectivity, shape ``(E, 3)``.
    """
    lines = path.read_text().strip().split("\n")
    data = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 3:
            data.append([int(x) for x in parts[:3]])
    return np.array(data, dtype=np.int32) - 1     # 0-based


def _compute_elem_centres(
    lon: np.ndarray,
    lat: np.ndarray,
    elem_nodes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute element centres using circular mean for longitude.

    Circular mean handles dateline crossings correctly.
    """
    tri_lon = lon[elem_nodes]         # (E, 3)
    tri_lat = lat[elem_nodes]         # (E, 3)
    # Circular mean for longitude
    lon_rad = np.deg2rad(tri_lon)
    lon_c = np.rad2deg(np.arctan2(
        np.sin(lon_rad).mean(axis=1),
        np.cos(lon_rad).mean(axis=1),
    ))
    lat_c = tri_lat.mean(axis=1)
    return lon_c, lat_c


def _read_node_depth(mesh_dir: Path, n_nodes: int) -> np.ndarray:
    """Read node depth from ``depth.out`` if available."""
    candidate = mesh_dir / "depth.out"
    if candidate.exists():
        return np.loadtxt(candidate, dtype=np.float64).ravel()[:n_nodes]
    return np.zeros(n_nodes, dtype=np.float64)


def _read_sigma_levels(mesh_dir: Path) -> np.ndarray:
    """Read vertical sigma/depth levels from aux3d.out or similar."""
    for fname in ["aux3d.out", "depth_levels.txt", "nlvls.out"]:
        candidate = mesh_dir / fname
        if candidate.exists():
            data = np.loadtxt(candidate, dtype=np.float64)
            if data.size > 0:
                return data.ravel()
    return np.array([], dtype=np.float64)


# ---------------------------------------------------------------------------
# NetCDF mesh reader (fesom.mesh.diag.nc)
# ---------------------------------------------------------------------------

def _resolve_mesh_vars(ds):
    """Return (lon_deg, lat_deg, elem_nodes_0based_Ex3, zlevels_abs, n_nodes)
    from a FESOM2 mesh-diag dataset, spanning the known naming conventions:
      - core2 (old): packed ``nodes (2,N)`` in RADIANS, ``elem (3,E)``, ``zbar``
      - .comp.nc:    ``lon``/``lat`` (deg), ``elements (3,E)``, ``nz``
      - UGRID:       ``lon``/``lat`` (deg), ``face_nodes (E,3)``, ``nz``/``zbar``
    Reads by VARIABLE NAME, never by dim name (nl/nl1 vs nz/nz1 irrelevant).
    """
    import numpy as np
    v = ds.variables
    # --- coordinates ---
    if "lon" in v and "lat" in v:
        lon = np.asarray(v["lon"][:], dtype=np.float64)
        lat = np.asarray(v["lat"][:], dtype=np.float64)
    elif "nodes" in v:
        nvar = v["nodes"]
        nodes = np.asarray(nvar[:], dtype=np.float64)         # (2, N)
        if nodes.shape[0] != 2:
            nodes = nodes.T
        lon, lat = nodes[0].copy(), nodes[1].copy()
        # Decide radians vs degrees.  Trust an explicit ``units`` attr first
        # (``radian*`` -> radians, ``degree*`` -> degrees); else require BOTH
        # |lon| <= 2*pi AND |lat| <= pi/2 (lon-only is insufficient — a small
        # degree-domain could sit under 2*pi yet be degrees).
        units = str(getattr(nvar, "units", "") or "").lower()
        if units.startswith("radian"):
            is_radians = True
        elif units.startswith("degree"):
            is_radians = False
        else:
            is_radians = (np.nanmax(np.abs(lon)) <= 2 * np.pi + 1e-6
                          and np.nanmax(np.abs(lat)) <= 0.5 * np.pi + 1e-6)
        if is_radians:
            lon = np.degrees(lon)
            lat = np.degrees(lat)
    else:
        raise ValueError(
            "mesh diag: no coordinates found (expected 'lon'/'lat' or 'nodes')")
    n_nodes = lon.shape[0]
    # --- connectivity ---
    conn_name = next((k for k in ("elements", "elem", "face_nodes") if k in v),
                     None)
    if conn_name is None:
        raise ValueError(
            "mesh diag: no connectivity found "
            "(expected 'elements', 'elem' or 'face_nodes')")
    conn_var = v[conn_name]
    conn = np.asarray(conn_var[:], dtype=np.int64)
    # orient to (E, 3): the axis of length 3 is the vertex axis
    if conn.shape[0] == 3 and conn.shape[1] != 3:
        conn = conn.T
    # Honour a UGRID ``start_index`` attr (FESOM2 writes 1, but 0 is legal);
    # default to 1 when absent (all our conventions are 1-based).
    try:
        start_index = int(getattr(conn_var, "start_index", 1))
    except (TypeError, ValueError):
        start_index = 1
    elem_nodes = (conn - start_index).astype(np.int32)        # base -> 0-based
    # --- z-levels ---
    z_name = next((k for k in ("nz", "zbar") if k in v), None)
    if z_name is not None:
        zlevels = np.abs(np.asarray(v[z_name][:], dtype=np.float64))
    else:
        zlevels = np.array([], dtype=np.float64)
    return lon, lat, elem_nodes, zlevels, n_nodes


def _nod_in_elem2d_is_degenerate(n2e: np.ndarray, n2e_count: np.ndarray) -> bool:
    """Decide whether a precomputed ``nod_in_elem2D`` table must be rebuilt.

    The table (already oriented ``(n_nodes, max_conn)`` and shifted to 0-based
    with padding ``-1``) must agree with its companion count array
    ``nod_in_elem2D_num``. Some FESOM2 mesh diags (the core2
    ``fesom.mesh.diag.nc`` among them) ship it *unpopulated* (all zeros ->
    all ``-1`` after the shift) while the count array stays valid; that
    silently breaks the containing-element search (every particle goes
    OUTSIDE, nothing advects).

    Two independent signals trip a rebuild:

    - **mostly empty** — more than half the nodes have no first incident
      element. Catches the fully-unpopulated table even when the count array
      is *also* zero (so the consistency check below cannot see it).
    - **inconsistent** — at least one node declares ``count > 0`` incident
      elements yet a slot *inside that count* is padding (``-1``), OR declares
      more incident elements than the table has columns to hold
      (``count > max_conn`` — a truncated/corrupt row). A healthy
      triangulation can never do either (the first ``count`` slots are real
      element ids and fit), so even a single occurrence — partial corruption
      well below the 50 % bar — forces a rebuild rather than trusting bad rows.

    Parameters
    ----------
    n2e : ndarray, shape (n_nodes, max_conn), int
        Node->element table, 0-based, ``-1`` padded.
    n2e_count : ndarray, shape (n_nodes,), int
        Per-node count of incident elements (``nod_in_elem2D_num``).

    Returns
    -------
    bool
        ``True`` if the table is unpopulated or inconsistent and should be
        rebuilt from the element connectivity.

    Notes
    -----
    This detects *structural* degeneracy (unpopulated rows, padding inside the
    declared count, and out-of-range counts). It deliberately does NOT
    cross-validate that every stored, non-negative element id is in range and
    actually contains the node — a table of wrong-but-plausible ids would pass.
    Full id validation against ``elem_nodes`` is out of scope here; the
    real-world failure mode is the all-zero / truncated table this catches.
    """
    if n2e.shape[0] == 0:
        return False
    max_conn = n2e.shape[1]
    if max_conn == 0:
        return True                                      # no adjacency stored at all
    mostly_empty = (n2e[:, 0] < 0).mean() > 0.5
    # Slots strictly inside each node's declared count must all be >= 0 ...
    slot_idx = np.arange(max_conn)[None, :]              # (1, max_conn)
    within_count = slot_idx < n2e_count[:, None]         # (n_nodes, max_conn)
    empty_inside_count = (within_count & (n2e < 0)).any()
    # ... and the count itself must be in [0, max_conn]. A negative count or one
    # that exceeds the table width is corrupt and, if trusted, drives
    # downstream out-of-bounds / wrong adjacency reads (node_to_elems[i, :cnt]).
    bad_count = (n2e_count < 0).any() or (n2e_count > max_conn).any()
    return bool(mostly_empty or empty_inside_count or bad_count)


def _load_mesh_nc(nc_path: Path, r_earth: float = 6_367_500.0,
                  rotation_angles=None) -> Mesh:
    """Load FESOM2 mesh from a NetCDF mesh diagnostic file.

    Expected variables (as produced by FESOM2):
        lon(nod2), lat(nod2)              — node coordinates (degrees)
        elements(n3, elem)                — connectivity (1-based)
        nod_in_elem2D(N, nod2)            — precomputed node→element map (1-based, 0=pad)
        nod_in_elem2D_num(nod2)           — count of elements per node
        zbar_n_bottom(nod2)               — nodal bottom depth (optional)
        nz(nz)                            — depth levels (optional)

    Parameters
    ----------
    nc_path : Path
        Path to the mesh NetCDF file.

    Returns
    -------
    Mesh
    """
    import netCDF4 as nc4

    t0 = _time.perf_counter()
    print(f"  Reading mesh from {nc_path.name} ...")

    ds = nc4.Dataset(str(nc_path), "r")

    # --- Coordinates / connectivity / z-levels (convention-agnostic) ---
    lon, lat, elem_nodes, sigma, n_nodes = _resolve_mesh_vars(ds)
    print(f"    {n_nodes:,} nodes  ({_time.perf_counter() - t0:.1f}s)")
    n_elems = elem_nodes.shape[0]
    print(f"    {n_elems:,} elements  ({_time.perf_counter() - t0:.1f}s)")

    # --- Precomputed node-to-element adjacency ---
    # stored (N, nod) or (nod, N); orient to (nod, N), 1-based→0-based, 0→-1
    n2e = np.asarray(ds.variables["nod_in_elem2D"][:], dtype=np.int32)
    if n2e.shape[0] != n_nodes:           # stored (N, nod) -> (nod, N)
        n2e = n2e.T
    n2e = n2e - 1  # Fortran 1-based→0-based; padding 0→-1
    n2e_count = np.asarray(ds.variables["nod_in_elem2D_num"][:], dtype=np.int32)
    # Some mesh-diag files (e.g. this core2 fesom.mesh.diag.nc) ship an
    # *unpopulated* or partially-corrupt nod_in_elem2D while nod_in_elem2D_num
    # is valid. After the 1-based->0-based shift the bad rows become -1, which
    # silently breaks the containing-element search (every particle -> OUTSIDE,
    # nothing advects). Detect that (see _nod_in_elem2d_is_degenerate) and
    # rebuild from the (valid) element connectivity instead.
    if _nod_in_elem2d_is_degenerate(n2e, n2e_count):
        print(f"    nod_in_elem2D is empty/inconsistent — rebuilding node_to_elems from elements ...")
        n2e, n2e_count = build_node_to_elems(n_nodes, elem_nodes)
        print(f"    node_to_elems rebuilt (max_conn={n2e.shape[1]})  ({_time.perf_counter() - t0:.1f}s)")
    else:
        print(f"    node_to_elems precomputed (max_conn={n2e.shape[1]})  ({_time.perf_counter() - t0:.1f}s)")

    # --- Depth ---
    if "zbar_n_bottom" in ds.variables:
        depth = np.abs(np.asarray(ds.variables["zbar_n_bottom"][:], dtype=np.float64))
    else:
        depth = np.zeros(n_nodes, dtype=np.float64)

    # Model-provided element areas (FESOM2 mesh diag) take PRECEDENCE over
    # the spherical-excess fallback: bit-parity with the model build and
    # pole-correct by construction (FESOM2's elem_area is the rotated-frame
    # planar cos-formula, unreproducible without the Euler angles).  Accept
    # only when the squeezed array is exactly 1-D of length n_elems; a
    # genuine extra axis (e.g. a time dim) falls through to the fallback.
    # ``copy=True``: the cache must OWN its buffer (returned by reference).
    elem_areas = None
    if "elem_area" in ds.variables:
        cand = np.squeeze(np.asarray(ds.variables["elem_area"][:],
                                     dtype=np.float64))
        if cand.ndim == 1 and cand.shape[0] == n_elems:
            elem_areas = np.array(cand, dtype=np.float64, copy=True)

    if elem_areas is not None:
        print("    element areas: from file (elem_area)")
    else:
        print("    element areas: not in file -> spherical fallback on demand")

    ds.close()

    # --- Element centres (geographic, the rotation carriers) ---
    print(f"    Computing element centres ...")
    lon_c, lat_c = _compute_elem_centres(lon, lat, elem_nodes)
    print(f"    done  ({_time.perf_counter() - t0:.1f}s)")

    # Near-pole warning on the GEOGRAPHIC nodes, before any rotation.
    _warn_near_pole_fesom2(lat, rotation_angles)

    # Rotate into the model frame (factors from geographic carriers first,
    # then coordinates as points).  Search structures are built AFTER, on
    # the rotated coordinates.
    vec_rot_factors = None
    if rotation_angles is not None:
        (lon, lat, lon_c, lat_c, vec_rot_factors) = _apply_mesh_rotation(
            lon, lat, lon_c, lat_c, rotation_angles)

    mesh = Mesh(
        name="fesom2",
        nodes_lon=lon,
        nodes_lat=lat,
        depth=depth,
        elem_nodes=elem_nodes,
        elem_center_lon=lon_c,
        elem_center_lat=lat_c,
        node_to_elems=n2e,
        node_to_elems_count=n2e_count,
        sigma=sigma,
        vertical_coord="zlevel",
        r_earth=r_earth,
        rotation_angles=rotation_angles,
        metadata={"meshpath": str(nc_path), "source": "netcdf"},
    )
    if elem_areas is not None:
        mesh._elem_areas = elem_areas
    mesh._vec_rot_factors = vec_rot_factors

    # --- spatial index ---
    print(f"    Building spatial index for {n_nodes:,} nodes ...")
    build_search(mesh)
    print(f"    Spatial index ready  ({_time.perf_counter() - t0:.1f}s)")

    return mesh


# ---------------------------------------------------------------------------
# Zarr mesh reader
# ---------------------------------------------------------------------------

def _load_mesh_zarr(zarr_path: Path, r_earth: float = 6_367_500.0,
                    rotation_angles=None) -> Mesh:
    """Load FESOM2 mesh from a Zarr store.

    Reads the same variables as ``_load_mesh_nc`` but via xarray with the
    Zarr engine.  This is the format produced by ``extract_fesom_region.py``.

    Parameters
    ----------
    zarr_path : Path
        Path to the ``.zarr`` directory store.

    Returns
    -------
    Mesh
    """
    t0 = _time.perf_counter()
    print(f"  Reading mesh from {zarr_path.name} (Zarr) ...")

    ds = xr.open_dataset(str(zarr_path), engine="zarr")

    # --- Coordinates ---
    lon = np.asarray(ds["lon"].values, dtype=np.float64)
    lat = np.asarray(ds["lat"].values, dtype=np.float64)
    n_nodes = lon.shape[0]
    print(f"    {n_nodes:,} nodes  ({_time.perf_counter() - t0:.1f}s)")

    # --- Element connectivity: (n3, E) → (E, 3), 1-based → 0-based ---
    elem_nodes = np.asarray(ds["elements"].values, dtype=np.int32).T  # (E, 3)
    elem_nodes -= 1  # 0-based
    n_elems = elem_nodes.shape[0]
    print(f"    {n_elems:,} elements  ({_time.perf_counter() - t0:.1f}s)")

    # --- Precomputed node-to-element adjacency ---
    # Some mesh-diag files (e.g. the core2 fesom.mesh.diag.nc) ship an
    # *unpopulated* or partially-corrupt nod_in_elem2D while nod_in_elem2D_num
    # is valid. After the 1-based->0-based shift the bad rows become -1, which
    # silently breaks the containing-element search (every particle -> OUTSIDE,
    # nothing advects). Detect that (see _nod_in_elem2d_is_degenerate) and
    # rebuild from the (valid) element connectivity instead.
    if "nod_in_elem2D" in ds and "nod_in_elem2D_num" in ds:
        n2e = np.asarray(ds["nod_in_elem2D"].values, dtype=np.int32).T
        n2e -= 1  # 1-based→0-based; padding 0→-1
        n2e_count = np.asarray(ds["nod_in_elem2D_num"].values, dtype=np.int32)
        if _nod_in_elem2d_is_degenerate(n2e, n2e_count):
            print(f"    nod_in_elem2D is empty/inconsistent — rebuilding node_to_elems from elements ...")
            n2e, n2e_count = build_node_to_elems(n_nodes, elem_nodes)
            print(f"    done  ({_time.perf_counter() - t0:.1f}s)")
        else:
            print(f"    node_to_elems precomputed (max_conn={n2e.shape[1]})  ({_time.perf_counter() - t0:.1f}s)")
    else:
        print(f"    Building node_to_elems from scratch ...")
        n2e, n2e_count = build_node_to_elems(n_nodes, elem_nodes)
        print(f"    done  ({_time.perf_counter() - t0:.1f}s)")

    # --- Depth ---
    if "zbar_n_bottom" in ds:
        depth = np.abs(np.asarray(ds["zbar_n_bottom"].values, dtype=np.float64))
    else:
        depth = np.zeros(n_nodes, dtype=np.float64)

    # --- Depth levels ---
    if "nz" in ds:
        sigma = np.abs(np.asarray(ds["nz"].values, dtype=np.float64))
    else:
        sigma = np.array([], dtype=np.float64)

    # --- Element centres ---
    if "elem_center_lon" in ds and "elem_center_lat" in ds:
        lon_c = np.asarray(ds["elem_center_lon"].values, dtype=np.float64)
        lat_c = np.asarray(ds["elem_center_lat"].values, dtype=np.float64)
    else:
        print(f"    Computing element centres ...")
        lon_c, lat_c = _compute_elem_centres(lon, lat, elem_nodes)

    # Model-provided element areas take PRECEDENCE over the spherical-
    # excess fallback (bit-parity with the model build, pole-correct).
    # Accept only when the squeezed var is exactly 1-D of length n_elems;
    # a genuine extra axis falls through to the fallback.  ``copy=True``:
    # the cache must OWN its buffer (returned by reference to consumers).
    elem_areas = None
    if "elem_area" in ds:
        cand = ds["elem_area"].squeeze()
        if cand.ndim == 1 and cand.shape[0] == n_elems:
            elem_areas = np.array(cand.values, dtype=np.float64, copy=True)

    if elem_areas is not None:
        print("    element areas: from file (elem_area)")
    else:
        print("    element areas: not in file -> spherical fallback on demand")

    ds.close()
    print(f"    Zarr mesh loaded  ({_time.perf_counter() - t0:.1f}s)")

    # Near-pole warning on the GEOGRAPHIC nodes, before any rotation.
    _warn_near_pole_fesom2(lat, rotation_angles)

    # Rotate into the model frame (factors from geographic carriers first,
    # then coordinates as points — element centres are rotated as POINTS
    # whether they were read from the store or computed, never recomputed
    # from rotated nodes).  Search structures are built AFTER, rotated.
    vec_rot_factors = None
    if rotation_angles is not None:
        (lon, lat, lon_c, lat_c, vec_rot_factors) = _apply_mesh_rotation(
            lon, lat, lon_c, lat_c, rotation_angles)

    mesh = Mesh(
        name="fesom2",
        nodes_lon=lon,
        nodes_lat=lat,
        depth=depth,
        elem_nodes=elem_nodes,
        elem_center_lon=lon_c,
        elem_center_lat=lat_c,
        node_to_elems=n2e,
        node_to_elems_count=n2e_count,
        sigma=sigma,
        vertical_coord="zlevel",
        r_earth=r_earth,
        rotation_angles=rotation_angles,
        metadata={"meshpath": str(zarr_path), "source": "zarr"},
    )
    if elem_areas is not None:
        mesh._elem_areas = elem_areas
    mesh._vec_rot_factors = vec_rot_factors

    print(f"    Building spatial index for {n_nodes:,} nodes ...")
    build_search(mesh)
    print(f"    Spatial index ready  ({_time.perf_counter() - t0:.1f}s)")

    return mesh


# ---------------------------------------------------------------------------
# Multi-year directory helpers (one file per variable per year)
# ---------------------------------------------------------------------------

#: Eta variable names tried (in order) when ``mode.var_eta`` is NOT set, in
#: the exact multi-year layout — mirrors the read-side alias list.
_ETA_ALIASES = ("ssh", "eta", "zos")
#: W variable names tried when ``mode.var_w`` is NOT set — mirrors the
#: read-side alias list in ``read_velocities``.
_W_ALIASES = ("w", "wo", "wvel")


def _fesom2_var_files(src, name):
    """Sorted yearly files for an EXACT variable token: <name>.fesom.<YYYY>.nc.

    Only files whose token between ``.fesom.`` and the extension parses as an
    integer year are returned (so ``u.fesom.mean.nc`` / ``u.fesom.2020.extra.nc``
    do not masquerade as yearly files and trip the exact-layout path).
    """
    files = [f for f in sorted(src.glob(f"{name}.fesom.*.nc"))
             if _parse_year(f) is not None]
    if not files:
        files = [f for f in sorted(src.glob(f"{name}.fesom.*.zarr"))
                 if _parse_year(f) is not None]
    return files


def _wanted_var_names(mode_cfg, diffusion=None):
    """Resolve the variable SPECS to assemble from the config.

    Returns a list of ``(candidates, required)`` tuples where ``candidates``
    is the ordered list of acceptable variable names (a single name, or an
    alias list when the name is unset) and ``required`` flags whether a
    zero-file match must raise:

      - ``var_u`` / ``var_v``       — always required.
      - W: explicit ``var_w``       — required in 3D; else the alias list
        ``w``/``wo``/``wvel`` (still required in 3D — at least one must exist).
      - eta: explicit ``var_eta``   — required; else the alias list ``ssh``/
        ``eta``/``zos`` — optional (no raise if none present).
      - explicit ``sample_vars``    — each required.
      - ``diffusion.var_az``        — required, when the vertical Az kick
        is active (``enabled`` + ``az_enabled``) in a 3-D run.
    """
    g = lambda a, d: (getattr(mode_cfg, a, None) or d) if mode_cfg else d
    specs = [([g("var_u", "u")], True), ([g("var_v", "v")], True)]
    is_3d = bool(getattr(mode_cfg, "Fdrift3D", False)) if mode_cfg else False
    if is_3d:
        cfg_w = getattr(mode_cfg, "var_w", None) if mode_cfg else None
        specs.append(([cfg_w] if cfg_w else list(_W_ALIASES), True))
    cfg_eta = getattr(mode_cfg, "var_eta", None) if mode_cfg else None
    if cfg_eta:
        specs.append(([cfg_eta], True))
    else:
        specs.append((list(_ETA_ALIASES), False))
    sv = getattr(mode_cfg, "sample_vars", None) if mode_cfg else None
    if sv:
        for name in sv:
            if name:
                specs.append(([name], True))
    az_on = (diffusion is not None
             and getattr(diffusion, "enabled", False)
             and getattr(diffusion, "az_enabled", False))
    var_az = getattr(diffusion, "var_az", None) if az_on else None
    if is_3d and var_az:
        specs.append(([var_az], True))
    return specs


def _parse_year(path):
    """Extract the integer year token from ``<name>.fesom.<YYYY>.<ext>``."""
    stem = Path(path).name
    parts = stem.split(".fesom.")
    if len(parts) != 2:
        return None
    tail = parts[1]
    # tail must be EXACTLY "<YYYY>.nc" or "<YYYY>.zarr" — reject
    # "<YYYY>.extra.nc", "mean.nc", "2020.20.nc", etc. so a stray file cannot
    # masquerade as a yearly file and trip the exact-layout path.
    for ext in (".nc", ".zarr"):
        if tail.endswith(ext):
            ytok = tail[: -len(ext)]
            return int(ytok) if (ytok.isdigit() and len(ytok) == 4) else None
    return None


def _warn_year_gap(name, files):
    """Warn (do NOT raise) if the year tokens for a variable are not
    consecutive — temporal interpolation would silently bridge the gap."""
    years = sorted({y for y in (_parse_year(f) for f in files) if y is not None})
    if len(years) < 2:
        return
    missing = [y for y in range(years[0], years[-1] + 1) if y not in years]
    if missing:
        warnings.warn(
            f"multi-year FESOM2: variable {name!r} has non-consecutive years "
            f"{years} (missing {missing}); temporal interpolation would bridge "
            "the gap silently.",
            RuntimeWarning, stacklevel=3)


def _open_var_mf(files, name):
    """Lazy open_mfdataset for one variable across its yearly files (works for
    a single-file list too — stays dask-backed)."""
    import xarray as xr
    eng = _detect_engine(files[0])
    try:
        ds = xr.open_mfdataset(
            [str(f) for f in files], combine="by_coords", engine=eng,
        )
    except ValueError as exc:
        raise ValueError(
            f"multi-year FESOM2: variable {name!r} could not be combined "
            f"across {[Path(f).name for f in files]} on a single ordered time "
            f"axis (duplicate/overlapping timestamps?): {exc}") from exc
    return ds[name].to_dataset(name=name)


def _is_exact_layout(src, mode_cfg):
    """True when ``src`` is an exact per-variable/per-year directory: EITHER
    configured velocity var (``var_u`` or ``var_v``) has >=1 matching
    ``<name>.fesom.<YYYY>.nc`` file.

    OR (not AND): a directory carrying ``u.fesom.*`` but missing ``v.fesom.*``
    is still an exact-layout dir with a broken/missing v — it must enter exact
    mode so the required-var check raises a clear error instead of silently
    falling through to the broad ``u*``/``v*`` glob (which would grab e.g.
    ``vice``).  A ``u.nc``/``v.nc`` directory or a single combined file matches
    neither glob -> ``False`` -> broad-glob path (unchanged)."""
    g = lambda a, d: (getattr(mode_cfg, a, None) or d) if mode_cfg else d
    u_name, v_name = g("var_u", "u"), g("var_v", "v")
    return (len(_fesom2_var_files(src, u_name)) > 0
            or len(_fesom2_var_files(src, v_name)) > 0)


def _validate_merged_time(ds):
    """Validate the merged ``time`` axis: strictly increasing and unique.

    Uses an adjacent ``t[1:] > t[:-1]`` comparison so it works uniformly for
    datetime64, numeric (signed or unsigned), and cftime/object axes — a
    ``np.diff`` would yield ``timedelta``/``datetime.timedelta`` that cannot be
    compared against a scalar ``0`` for cftime, and could wrap for unsigned
    integers.
    """
    if "time" not in ds.dims:
        return
    t = ds["time"].values
    if t.size < 2:
        return
    if not bool(np.all(t[1:] > t[:-1])):
        raise ValueError(
            "multi-year FESOM2: merged time axis is not strictly increasing "
            "(duplicate, overlapping, or misordered timestamps across the "
            "yearly files).")


def _multifile_open(src, mode_cfg, diffusion=None):
    """Lazy per-variable open_mfdataset + merge across years.

    Returns a merged xr.Dataset (dask-backed) when ``src`` is an exact
    per-variable/per-year layout (``var_u``/``var_v`` each match
    ``<name>.fesom.*.nc``), else ``None`` (fall through to the broad-glob
    path).  Required variables with zero matching files raise a clear
    ``ValueError`` naming the var + glob.  ``diffusion`` (when the Az
    kick is active) adds ``var_az`` to the assembled family — cycle1
    stores ``Av.fesom.<year>.nc`` per year like every other variable.
    """
    import xarray as xr
    if not _is_exact_layout(src, mode_cfg):
        return None
    parts, seen = [], set()
    for candidates, required in _wanted_var_names(mode_cfg, diffusion):
        opened = None
        for name in candidates:
            files = _fesom2_var_files(src, name)
            if not files:
                continue
            opened = name
            if name in seen:        # already assembled (e.g. listed twice)
                break
            seen.add(name)
            _warn_year_gap(name, files)
            parts.append(_open_var_mf(files, name))
            break
        if opened is None and required:
            tried = " / ".join(f"{c}.fesom.*.nc" for c in candidates)
            raise ValueError(
                f"multi-year FESOM2: required variable {candidates[0]!r} not "
                f"found (no files matched {tried} under {src}).")
    if not parts:
        return None
    # join="exact" makes a mixed-cadence family (monthly u vs daily w) fail
    # loudly: xarray raises AlignmentError on newer versions and a plain
    # ValueError on older ones (AlignmentError did not exist before ~2023.8).
    # Catch by the shared ValueError base (AlignmentError subclasses it) and
    # relabel with the cadence hint — independent of the class name being
    # importable from the installed xarray.
    try:
        merged = xr.merge(parts, join="exact")
    except ValueError as exc:
        raise ValueError(
            "multi-year FESOM2 open failed (mixed time cadence across the "
            f"selected variables? — they must share a time axis): {exc}"
        ) from exc
    _validate_merged_time(merged)
    return merged


# ---------------------------------------------------------------------------
# Reader class
# ---------------------------------------------------------------------------

@register("fesom2")
class Fesom2Reader:
    """Reader for FESOM2 meshes and velocity output."""

    @staticmethod
    def load_mesh(mesh_cfg) -> Mesh:
        """Load FESOM2 mesh from a directory, NetCDF, or Zarr file.

        Accepts:
          - A directory containing ``nod2d.out`` and ``elem2d.out`` (ASCII).
          - A single ``.nc`` file (``fesom.mesh.diag.nc`` or similar).
          - A ``.zarr`` directory store (as produced by ``extract_fesom_region``).

        Parameters
        ----------
        mesh_cfg : MeshConfig (or any object with a ``.meshpath`` attribute)

        Returns
        -------
        Mesh
        """
        mesh_path = Path(mesh_cfg.meshpath).expanduser()

        # Earth radius: explicit config override, else the FESOM2 family
        # default (the FESOM namelist value 6_367_500 m).  Used by the
        # spherical-excess area fallback; model-provided areas ignore it.
        r_earth = (getattr(mesh_cfg, "r_earth", None) or 6_367_500.0)

        # Coordinate-frame rotation (FESOM2-only): None | "fesom2" | [a,b,g].
        rotation_angles = _resolve_rotation_angles(mesh_cfg)

        # Zarr mesh store
        if _detect_engine(mesh_path) == "zarr":
            return _load_mesh_zarr(mesh_path, r_earth=r_earth,
                                   rotation_angles=rotation_angles)

        # NetCDF mesh file
        if mesh_path.suffix.lower() == ".nc" and mesh_path.is_file():
            return _load_mesh_nc(mesh_path, r_earth=r_earth,
                                 rotation_angles=rotation_angles)

        # ASCII mesh directory (legacy)
        if not mesh_path.is_dir():
            raise FileNotFoundError(
                f"FESOM2 mesh path not found: {mesh_path}  "
                f"(expected .nc, .zarr, or directory with nod2d.out/elem2d.out)"
            )
        nod_file = mesh_path / "nod2d.out"
        elem_file = mesh_path / "elem2d.out"
        if not nod_file.exists():
            raise FileNotFoundError(f"Missing {nod_file}")
        if not elem_file.exists():
            raise FileNotFoundError(f"Missing {elem_file}")

        lon, lat = _read_nod2d(nod_file)
        elem_nodes = _read_elem2d(elem_file)
        lon_c, lat_c = _compute_elem_centres(lon, lat, elem_nodes)
        depth = _read_node_depth(mesh_path, lon.shape[0])
        sigma = _read_sigma_levels(mesh_path)

        n2e, n2e_count = build_node_to_elems(lon.shape[0], elem_nodes)

        # Near-pole warning on the GEOGRAPHIC nodes, before any rotation.
        _warn_near_pole_fesom2(lat, rotation_angles)

        # Rotate into the model frame (factors from geographic carriers
        # first, then coordinates as points).  build_search runs AFTER, on
        # the rotated coordinates.
        vec_rot_factors = None
        if rotation_angles is not None:
            (lon, lat, lon_c, lat_c, vec_rot_factors) = _apply_mesh_rotation(
                lon, lat, lon_c, lat_c, rotation_angles)

        mesh = Mesh(
            name="fesom2",
            nodes_lon=lon,
            nodes_lat=lat,
            depth=depth,
            elem_nodes=elem_nodes,
            elem_center_lon=lon_c,
            elem_center_lat=lat_c,
            node_to_elems=n2e,
            node_to_elems_count=n2e_count,
            sigma=sigma,
            vertical_coord="zlevel",
            r_earth=r_earth,
            rotation_angles=rotation_angles,
            metadata={"meshpath": str(mesh_path)},
        )
        mesh._vec_rot_factors = vec_rot_factors
        # ASCII meshes never carry model areas; spherical-excess is used.
        print("    element areas: not in file -> spherical fallback on demand")
        build_search(mesh)
        return mesh

    @staticmethod
    def open_dataset(path: str, v_path=None, w_path=None, mode_cfg=None,
                     diffusion=None) -> xr.Dataset:
        """Open FESOM2 velocity dataset.

        Accepts:
          - A single NetCDF or Zarr file/store with u, v [, ssh] variables.
          - Separate u/v/w files via ``v_path`` and ``w_path``.
          - A multi-year directory in the ``<var>.fesom.<year>.nc`` layout
            (one file per variable per year): when ``mode_cfg`` is supplied,
            exact per-variable globs (from ``mode.var_*``/``sample_vars``) are
            opened lazily via ``xr.open_mfdataset`` and merged across years.
          - A directory containing ``u*.nc``/``v*.nc`` or ``u*.zarr``/``v*.zarr``
            (plus optional ``w*``/``ssh*``).  Any other per-variable file in
            the directory (``temp*``, ``salt*``, ...) is merged too, so
            ``mode.sample_vars`` can sample it; files with ``mesh`` in the
            name and variables on non-FESOM2 dims are skipped.

        Parameters
        ----------
        path : str
            Path to file, Zarr store, or directory.
        v_path : str or None
            Separate v-velocity file.
        w_path : str or None
            Separate w-velocity file.
        mode_cfg : ModeConfig or None
            When given and ``path`` is a directory, enables the multi-year
            per-variable layout: the requested variable family (resolved from
            ``var_u``/``var_v``/``var_w``/``var_eta``/``sample_vars``) is
            assembled from ``<var>.fesom.<year>.nc`` files.  Single-year
            directories fall through to the broad-glob behaviour.
        diffusion : DiffusionConfig or None
            When the vertical Az kick is active (``enabled`` +
            ``az_enabled`` + ``var_az``), the multi-year exact layout also
            assembles the ``var_az`` family (e.g. ``Av.fesom.<year>.nc``).
            Ignored otherwise; single files/broad-glob dirs are unaffected
            (Az is then read from the opened dataset directly).

        Returns
        -------
        xr.Dataset
        """
        src = Path(path).expanduser()
        engine = _detect_engine(src)
        # Without mode_cfg the reader cannot know that a run is 2-D: a W
        # that does not fit is then refused rather than skipped.
        w_needed = mode_cfg is None or bool(getattr(mode_cfg, "Fdrift3D", False))

        if src.is_dir() and engine != "zarr" and (v_path or w_path):
            warnings.warn(
                "open_dataset: data.ncinput_v / data.ncinput_w are ignored when "
                "data.ncinput is a directory (the directory's own v*/w* files "
                "are used); point data.ncinput at the u file to attach them",
                RuntimeWarning, stacklevel=2)

        # Multi-year per-variable layout: <var>.fesom.<year>.nc, exact globs
        # from the config; lazy open_mfdataset + merge.  _multifile_open raises
        # clear ValueErrors itself (missing required var, mixed cadence,
        # non-monotonic time); returns None for a non-exact dir (broad-glob
        # fallthrough below).
        if src.is_dir() and mode_cfg is not None:
            merged = _multifile_open(src, mode_cfg, diffusion)
            if merged is not None:
                return merged
        # Mixed-layout guard: per-year Az files in a directory whose
        # velocities did NOT enter the exact layout would fall to the
        # broad glob and silently use a single year of Az.
        if src.is_dir() and diffusion is not None:
            _az_on = (getattr(diffusion, "enabled", False)
                      and getattr(diffusion, "az_enabled", False))
            _va = getattr(diffusion, "var_az", None) if _az_on else None
            if _va and len(_fesom2_var_files(src, _va)) >= 1:
                # Even ONE yearly-pattern Az file here is anomalous: the
                # broad glob would outer-align it against the velocity
                # time axis and the NaN padding would silently zero the
                # kick outside that year (round-2 review).
                raise ValueError(
                    f"mixed layout: {_va}.fesom.<year> file(s) present "
                    "but the velocity variables are not in the exact "
                    "per-variable layout — use the exact layout for the "
                    "whole family or pre-concatenate the Az variable"
                )

        # Single file or Zarr store (not a directory of split files)
        if src.is_file() or engine == "zarr":
            ds_u = xr.open_dataset(src, engine=engine)
            # Standardise names
            renames = {}
            for alias, std in [("ua", "u"), ("u_vel", "u"),
                               ("va", "v"), ("v_vel", "v")]:
                if alias in ds_u and std not in ds_u:
                    renames[alias] = std
            if renames:
                ds_u = ds_u.rename(renames)

            # Merge separate v file if provided
            if v_path is not None and "v" not in ds_u:
                vp = Path(v_path).expanduser()
                ds_v = xr.open_dataset(vp, engine=_detect_engine(vp))
                v_renames = {}
                for alias, std in [("va", "v"), ("v_vel", "v")]:
                    if alias in ds_v and std not in ds_v:
                        v_renames[alias] = std
                if v_renames:
                    ds_v = ds_v.rename(v_renames)
                v_key = "v" if "v" in ds_v else list(ds_v.data_vars)[0]
                try:
                    ds_u["v"] = _align_onto_time_axis(
                        ds_u, ds_v[v_key], f"v file {vp}", f"the u file {src}",
                        require_time=True)
                except ValueError:
                    ds_v.close()
                    ds_u.close()
                    raise

            # Merge separate w file if provided
            if w_path is not None and "w" not in ds_u:
                wp = Path(w_path).expanduser()
                ds_w = xr.open_dataset(wp, engine=_detect_engine(wp))
                w_key = "w" if "w" in ds_w else list(ds_w.data_vars)[0]
                # A W covering only part of u's records (a reconstruct_w
                # time_range sub-run) attaches with NaN elsewhere;
                # read_velocities refuses such a record when it is needed.
                try:
                    ds_u["w"] = _align_onto_time_axis(
                        ds_u, ds_w[w_key], f"w file {wp}", f"the u file {src}",
                        allow_missing=True, require_time=True,
                        hint=_W_FILE_HINT)
                except ValueError as exc:
                    ds_w.close()
                    if w_needed:
                        ds_u.close()
                        raise
                    # a 2-D run never reads W (e.g. ncinput_w kept from a
                    # 3-D config): skip it, as the directory layout does
                    warnings.warn(f"open_dataset: skipping the w file {wp.name}: "
                                  f"{exc}", RuntimeWarning, stacklevel=2)

            return ds_u

        if not src.is_dir():
            raise FileNotFoundError(f"Path not found: {src}")

        # Directory: look for split files (NetCDF or Zarr)
        u_files = sorted(src.glob("u*.nc")) or sorted(src.glob("u*.zarr"))
        v_files = sorted(src.glob("v*.nc")) or sorted(src.glob("v*.zarr"))
        if not u_files or not v_files:
            raise FileNotFoundError(
                f"Directory {src} must contain u*/v* files (.nc or .zarr)."
            )
        opened = []

        def _open(p):
            d = xr.open_dataset(p, engine=_detect_engine(p))
            opened.append(d)
            return d

        try:
            ds_u = _open(u_files[0])
            ds_v = _open(v_files[0])

            # Pick the velocity variable (may have different names)
            u_key = "u" if "u" in ds_u else list(ds_u.data_vars)[0]
            v_key = "v" if "v" in ds_v else list(ds_v.data_vars)[0]

            u_da = ds_u[u_key]
            # Every member is attached by its time LABELS when the dataset is
            # assembled (an outer join): a file on another record axis would
            # stretch the axis and pad the velocities with NaN.  Each member is
            # re-positioned onto u's records first (_align_onto_time_axis); v
            # must cover them, W may cover part of them (refused per record at
            # read time) and is skipped with a warning in 2-D, ssh and the
            # optional extras are skipped with a warning when they do not fit.
            u_name = u_files[0].name
            datasets = {"u": u_da, "v": _align_onto_time_axis(
                u_da, ds_v[v_key], f"v file {v_files[0].name}",
                f"the u file {u_name}", require_time=True)}

            def _optional(name, da, fname, **kw):
                try:
                    datasets[name] = _align_onto_time_axis(
                        u_da, da, f"{fname}", f"the u file {u_name}", **kw)
                except ValueError as exc:
                    if name == "w" and w_needed:
                        raise
                    warnings.warn(f"open_dataset: skipping {name!r} from "
                                  f"{fname}: {exc}", RuntimeWarning, stacklevel=3)

            # Check for W
            w_files = sorted(src.glob("w*.nc")) or sorted(src.glob("w*.zarr"))
            if w_files:
                ds_w = _open(w_files[0])
                w_key = "w" if "w" in ds_w else list(ds_w.data_vars)[0]
                _optional("w", ds_w[w_key], w_files[0].name,
                          allow_missing=True, require_time=True, hint=_W_FILE_HINT)

            # Check for SSH
            ssh_files = sorted(src.glob("ssh*.nc")) or sorted(src.glob("ssh*.zarr"))
            if ssh_files:
                ds_ssh = _open(ssh_files[0])
                ssh_key = "ssh" if "ssh" in ds_ssh else list(ds_ssh.data_vars)[0]
                _optional("ssh", ds_ssh[ssh_key], ssh_files[0].name,
                          allow_missing=True, warn_missing=True)

            # Merge any remaining per-variable files (temp*, salt*, ...) so
            # trajectory sampling (``mode.sample_vars``) can reach them — raw
            # FESOM2 output and extract_fesom_region both store one variable
            # per file.  Two guards: mesh-diagnostic stores are skipped by
            # name, and variables on alien grids (dims outside the known
            # FESOM2 set) are skipped silently — they cannot be sampled and
            # would only risk dimension-alignment errors.
            matched = {f for group in (u_files, v_files, w_files, ssh_files)
                       for f in group}
            extra_files = [
                f for f in sorted(src.glob("*.nc")) + sorted(src.glob("*.zarr"))
                if f not in matched and "mesh" not in f.name.lower()
            ]
            known_dims = {"time", "nz1", "nz", "nod2", "elem",
                          "node", "sigma", "depth", "level"}
            for f in extra_files:
                try:
                    ds_x = _open(f)
                except Exception as exc:  # unreadable → skip, but say so
                    warnings.warn(f"open_dataset: skipping unreadable {f.name}: "
                                  f"{exc}", RuntimeWarning, stacklevel=2)
                    continue
                for var in ds_x.data_vars:
                    if var not in datasets and set(ds_x[var].dims) <= known_dims:
                        _optional(var, ds_x[var], f.name, allow_missing=True,
                                  warn_missing=True)

            return xr.Dataset(datasets)
        except BaseException:
            for d in opened:          # no file left open on refusal
                d.close()
            raise

    @staticmethod
    def read_scalar_fields(ds, timestep, mesh, varnames, mode_cfg=None):
        """Read node-based scalar fields for trajectory sampling (TODO 2.4).

        Returns {name: (N,) | (K-1, N) | (K, N) float64}; the vertical dim
        is trimmed consistently with mesh.sigma (already max_levels-trimmed
        by run_tracking).  Raises ValueError for missing or non-node-based
        variables.
        """
        is_3d = mode_cfg is not None and getattr(mode_cfg, "Fdrift3D", False)
        N = mesh.n_nodes
        K_t = mesh.sigma.size            # trimmed interface count (0 in 2D runs)
        # Direct/public callers (track_particles, raw reader calls) may pass an
        # untrimmed mesh.sigma with max_levels set; read_velocities trims u/v/w
        # by max_levels directly, so trim scalars the same way to keep the
        # returned VelocityField consistent.
        ml = getattr(mode_cfg, "max_levels", None) if mode_cfg is not None else None
        if is_3d and ml is not None and K_t > 0:
            K_t = min(K_t, int(ml))
        out = {}
        vdims = ("nz1", "nz", "depth", "level")
        for name in varnames:
            if name not in ds:
                raise ValueError(
                    f"sample variable '{name}' not found in dataset")
            da = ds[name]
            if "time" in da.dims:
                da = da.isel(time=timestep)
            vdim = next((d for d in vdims if d in da.dims), None)
            if vdim is not None and not is_3d:
                depth_idx = 0
                if mode_cfg is not None and getattr(
                        mode_cfg, "sigma_layer", None) is not None:
                    depth_idx = mode_cfg.sigma_layer
                da = da.isel({vdim: depth_idx})
                vdim = None
            if vdim is not None and K_t > 0:
                n_src = da.sizes[vdim]
                if vdim == "nz":
                    keep = min(K_t, n_src)              # interface field
                elif vdim == "nz1":
                    keep = min(K_t - 1, n_src)          # layer field
                else:                                    # depth/level: by size
                    keep = min(K_t if n_src >= K_t else K_t - 1, n_src)
                da = da.isel({vdim: slice(0, keep)})
            # Orient by dim NAME (known before .values), not by shape: a square
            # field (N == n_layers) stored (nod2, nz1) would defeat a
            # shape-heuristic transpose and silently return axis-swapped data.
            vdim_axis = da.dims.index(vdim) if vdim is not None else None
            arr = np.asarray(da.values, dtype=np.float64)
            if vdim_axis == 1:
                arr = np.ascontiguousarray(arr.T)        # (N, K) -> (K, N)
            if arr.shape[-1] != N:
                raise ValueError(
                    f"sample variable '{name}' is not node-based: shape "
                    f"{arr.shape}, expected last axis {N} "
                    "(element-based fields are not supported)")
            out[name] = arr
        return out

    @staticmethod
    def read_velocities(ds: xr.Dataset, timestep: int, mesh: Mesh,
                        mode_cfg=None, diffusion=None,
                        sample_vars=None, velocity_float32=False) -> VelocityField:
        """Read velocity fields for one timestep.

        Parameters
        ----------
        ds : xr.Dataset
        timestep : int
            Time index.
        mesh : Mesh
        mode_cfg : ModeConfig or None
            If None, defaults to 2D mode.
        sample_vars : list of str or None
            Node-based scalar variable names to attach to the returned
            ``VelocityField.scalars`` for trajectory sampling.  ``None`` or
            an empty list leaves ``scalars=None`` (default behaviour).

        Returns
        -------
        VelocityField
        """
        u_name, v_name = _resolve_uv_names(ds, mode_cfg)
        u_var = ds[u_name].isel(time=timestep)
        v_var = ds[v_name].isel(time=timestep)

        # Classify u/v mesh location (element centres vs nodes).  u and v
        # must agree — a mismatch means a misconfigured var_u/var_v pair.
        uv_layout = _detect_field_layout(u_var, mesh)
        v_layout = _detect_field_layout(v_var, mesh)
        if uv_layout != v_layout:
            raise ValueError(
                f"u/v layout disagreement: {u_name!r} is {uv_layout} but "
                f"{v_name!r} is {v_layout}; both must be on the same mesh "
                "location (element or node)."
            )

        is_3d = mode_cfg is not None and mode_cfg.Fdrift3D
        max_levels = getattr(mode_cfg, "max_levels", None) if mode_cfg else None
        sigma_keep = None
        if is_3d and max_levels is not None:
            if max_levels < 2:
                raise ValueError(
                    f"max_levels must be >= 2 in 3D mode, got {max_levels}"
                )
            sigma_keep = int(max_levels)
            if mesh.sigma.size > 0:
                sigma_keep = min(sigma_keep, mesh.sigma.size)

        # Handle depth dimension
        if not is_3d:
            # 2D: select a single depth level
            depth_idx = 0
            if mode_cfg is not None and mode_cfg.sigma_layer is not None:
                depth_idx = mode_cfg.sigma_layer
            for dim in ("nz1", "nz", "depth", "level"):
                if dim in u_var.dims:
                    u_var = u_var.isel({dim: depth_idx})
                if dim in v_var.dims:
                    v_var = v_var.isel({dim: depth_idx})
        elif sigma_keep is not None:
            # 3D with interface-count limit: keep K interfaces => K-1 layers
            n_layer_keep = max(sigma_keep - 1, 0)
            for dim in ("nz1", "nz", "depth", "level"):
                if dim in u_var.dims:
                    u_var = u_var.isel({dim: slice(0, n_layer_keep)})
                if dim in v_var.dims:
                    v_var = v_var.isel({dim: slice(0, n_layer_keep)})

        # 3D: the kernels expect (K-1, N) — vertical axis first.  A node
        # variable stored (time, nod2, nz1) survives detection but comes
        # out (N, K-1).  Orient by dim NAME (known before .values), not by
        # shape, mirroring read_scalar_fields: a square field would defeat
        # a shape heuristic and silently return axis-swapped data.
        if is_3d:
            u_var = _orient_vert_first(u_var)
            v_var = _orient_vert_first(v_var)

        # u/v/w may be kept in float32 (native FESOM2 dtype) to halve snapshot
        # memory (TODO 5.3); the per-particle kernels accumulate in float64, so
        # a non-rotated run is bit-identical to the float64 path.  Default
        # float64.
        vd = np.float32 if velocity_float32 else np.float64
        # C-order explicitly: for spatial-first storage _orient_vert_first is
        # a transpose VIEW, and asarray with a matching dtype makes no copy —
        # an F-ordered field would silently drop the parallel whole-field
        # time blends to their serial numpy fallback (2026-08 review).
        u = np.ascontiguousarray(np.asarray(u_var.values, dtype=vd))
        v = np.ascontiguousarray(np.asarray(v_var.values, dtype=vd))

        # Rotate u/v into the mesh frame when the mesh carries a rotation.
        # The tangent-plane image of a proper rotation is a pure 2D rotation:
        # (c, s) come from rotating unit-east at the carrier (element centres
        # for element layout, nodes for node layout); unit-north maps to
        # (-s, c) (pinned by tests against rotate_vectors on random fields).
        # The carrier factors are (X,); for 3D (K-1, X) they broadcast across
        # the layer axis (spatial axis is last).  Scalars (w/eta/Kh) are
        # frame-invariant and pass through untouched.
        if getattr(mesh, "rotation_angles", None) is not None:
            c, s = mesh._vec_rot_factors[uv_layout]
            # The rotation (float64 factors) promotes u/v back to float64; cast
            # to the target dtype so float32 storage is preserved (rotated runs
            # then round at ~float32 eps — negligible, documented).
            u, v = (c * u - s * v).astype(vd, copy=False), \
                   (s * u + c * v).astype(vd, copy=False)

        # SSH / eta — honor a configured mode.var_eta first (mirrors var_w
        # below and the FESOM-C reader), then the default name list.  Keeps
        # the use_eta fail-fast guard and the actual read in agreement: a
        # custom eta name that satisfies the guard is also the one read here.
        eta = np.zeros(mesh.n_nodes, dtype=np.float64)
        cfg_var_eta = getattr(mode_cfg, "var_eta", None) if mode_cfg else None
        eta_keys = ([cfg_var_eta] if cfg_var_eta else []) + ["ssh", "eta", "zos"]
        for key in eta_keys:
            if key in ds:
                eta = np.asarray(
                    ds[key].isel(time=timestep).values, dtype=np.float64
                )
                break

        # W velocity (3D mode only)
        w = None
        if is_3d:
            cfg_var_w = getattr(mode_cfg, "var_w", None)
            for w_name in ([cfg_var_w] if cfg_var_w else ["w", "wo", "wvel"]):
                if w_name in ds:
                    w_var = ds[w_name].isel(time=timestep)
                    # Stagger guard (mirrors the az validator below): W is
                    # interface-based; a LAYER field (dim nz1, K-1 levels)
                    # silently accepted here becomes wrong vertical
                    # velocities everywhere downstream.
                    if "nz1" in w_var.dims:
                        raise ValueError(
                            f"vertical velocity {w_name!r} lives on the "
                            "nz1 mid-layers — W must be interface-based "
                            "(dim nz, K interfaces)")
                    # Orient by dim name like u/v/az: a spatial-first store
                    # (time, nod2, nz) otherwise returns w axis-SWAPPED
                    # (N, K) — silently corrupting every consumer that
                    # trusts w.shape[0] == K (2026-08 review).
                    w_var = _orient_vert_first(w_var)
                    if sigma_keep is not None:
                        for dim in ("nz", "depth", "level"):
                            if dim in w_var.dims:
                                w_var = w_var.isel({dim: slice(0, sigma_keep)})
                    w = np.ascontiguousarray(
                        np.asarray(w_var.values, dtype=vd))
                    # cheap first: a W record with any value is the rule;
                    # the full scan runs only when the first block is empty
                    if (w.size and not np.isfinite(w.ravel()[:4096]).any()
                            and not np.isfinite(w).any()):
                        # a W attached for part of the records only (a
                        # utils/reconstruct_w.py time_range sub-run)
                        raise ValueError(
                            f"the vertical velocity {w_name!r} has no value "
                            "at the requested record: the attached W file "
                            "does not cover it. Reconstruct the run's "
                            "records, or keep the run inside the W file's "
                            "time range with a margin: a step reads the next "
                            "record as well (PCHIP the next two)")
                    # FESOM2 writes w positive UP (oce_ale.F90,
                    # vert_vel_ale); the tracker integrates a positive-
                    # DOWN depth.  zstar: w is the velocity through the
                    # moving interfaces; integrated on the fixed reference
                    # levels it advances the particle's zstar depth with a
                    # relative rate error eta/H; see docs/vertical.md.
                    if resolve_w_sign(w_var, mode_cfg, "up") < 0.0:
                        # a NEW array: on an in-memory dataset ``w`` may be
                        # a view of it, and an in-place flip would negate
                        # the stored field (every re-read alternates)
                        w = np.negative(w)
                    # Level-count check only when the mesh carries a
                    # vertical table (unit fixtures / direct callers may
                    # read with an empty mesh.sigma — no expectation then).
                    k_expect_w = (sigma_keep if sigma_keep is not None
                                  else mesh.sigma.shape[0])
                    if (w.ndim != 2 or w.shape[1] != mesh.n_nodes
                            or (k_expect_w > 0
                                and w.shape[0] != k_expect_w)):
                        raise ValueError(
                            f"vertical velocity {w_name!r}: expected an "
                            f"interface-based node field of shape "
                            f"({k_expect_w or 'K'}, {mesh.n_nodes}), got "
                            f"{w.shape} — W must live on the nz "
                            "interfaces (K levels), same layout as the "
                            "mesh vertical table")
                    break

        # Vertical diffusivity Az (TODO 2.2b) — node-based at interfaces,
        # the same (K, N) layout as w; trimmed by max_levels like w and
        # zero-filled where the file stores fill values (below-bottom).
        # Read ONLY when the vertical kick is actually on: a var_az left
        # in a config whose diffusion / az_enabled is off must cost no
        # I/O and must not fail on a missing variable (A/B configs).
        az = None
        az_on = (diffusion is not None
                 and getattr(diffusion, "enabled", False)
                 and getattr(diffusion, "az_enabled", False))
        var_az = getattr(diffusion, "var_az", None) if az_on else None
        if is_3d and var_az:
            if var_az not in ds:
                raise ValueError(
                    f"diffusion.var_az={var_az!r} not found in the dataset "
                    f"(variables: {sorted(ds.data_vars)})"
                )
            if _az_var_is_mid_layer(ds, var_az):
                # Reject BEFORE any trimming: with max_levels set, a
                # (K-1) mid-layer variable would pass the trimmed-shape
                # check below half-a-layer shifted (review 2026-08-15).
                # Catches the canonical nz1 name and, via the pre-trim
                # size vs the file's nz dim, depth/level-named variants.
                raise ValueError(
                    f"diffusion.var_az={var_az!r} lives on the "
                    "mid-layers — Az must be interface-based (dim nz, "
                    "same layout as w)"
                )
            az_var = _orient_vert_first(ds[var_az].isel(time=timestep))
            if sigma_keep is not None:
                for dim in ("nz", "depth", "level"):
                    if dim in az_var.dims:
                        az_var = az_var.isel({dim: slice(0, sigma_keep)})
            # C-order for the parallel time blends (see u/v above).
            az = np.ascontiguousarray(
                np.asarray(az_var.values, dtype=np.float64))
            k_expect = (w.shape[0] if w is not None
                        else (sigma_keep if sigma_keep is not None
                              else mesh.sigma.shape[0]))
            if (az.ndim != 2 or az.shape[0] != k_expect
                    or az.shape[1] != mesh.n_nodes):
                raise ValueError(
                    f"diffusion.var_az={var_az!r}: expected an "
                    f"interface-based node field of shape "
                    f"({k_expect}, {mesh.n_nodes}) (same layout as w), "
                    f"got {az.shape} — Az must live on the nz interfaces, "
                    "not the nz1 mid-layers"
                )
            az = np.nan_to_num(az, nan=0.0, posinf=0.0, neginf=0.0)

        kh = _read_kh_fields(ds, timestep, mesh, diffusion)
        kh_z, kh_m, kh_layout = kh if kh is not None else (None, None, None)
        scalars = None
        if sample_vars:
            scalars = Fesom2Reader.read_scalar_fields(
                ds, timestep, mesh, sample_vars, mode_cfg)
        return VelocityField(
            u=u, v=v, eta=eta, w=w,
            kh_zonal=kh_z, kh_meridional=kh_m, kh_layout=kh_layout,
            uv_layout=uv_layout,
            scalars=scalars,
            az=az,
        )

    @staticmethod
    def probe_uv_layout(ds, mode_cfg, mesh) -> str:
        """Classify the u/v layout the run will actually read — node or
        element — from variable METADATA only (dims/sizes, no data load).
        Shares name resolution with read_velocities.

        Parameters
        ----------
        ds : xr.Dataset
            Open dataset (variables need not be loaded).
        mode_cfg : ModeConfig or None
            Run mode; selects the same u/v variable names that
            ``read_velocities`` will use.
        mesh : Mesh
            Mesh for node/element count disambiguation.

        Returns
        -------
        str
            ``"node"`` or ``"element"``.
        """
        u_name, _v_name = _resolve_uv_names(ds, mode_cfg)
        return _classify_layout_meta(ds[u_name], mesh)
