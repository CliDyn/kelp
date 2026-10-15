"""FESOM-C model reader.

Reads mesh and velocity data from FVCOM-style NetCDF files.
"""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import xarray as xr

from ..core.config import MeshConfig, ModeConfig
from ..core.io import VelocityField
from ..core.mesh import (Mesh, build_node_to_elems, build_search,
                         validate_quad_convexity)
from . import (
    register,
    _detect_engine,
    _read_kh_fields,
    _detect_field_layout,
    _classify_layout_meta,
    resolve_w_sign,
    _align_onto_time_axis,
)

#: Mesh latitudes beyond this (degrees) trigger the near-pole warning (the
#: lon/lat step metric degenerates near the pole; FESOM-C offers no rotation).
POLE_WARN_LAT = 88.0


def _resolve_uv_names(ds, mode_cfg):
    """Return the (u_name, v_name) FESOM-C ``read_velocities`` will read.

    Reproduces the reader's per-branch name logic exactly so the probe and
    the read can never disagree:

    - 3D (``mode_cfg.Fdrift3D`` true): ``var_u`` / ``var_v`` else ``"u"`` /
      ``"v"``.
    - 2D: ``var_u`` / ``var_v`` else ``"u2d"`` if present else ``"u"`` (and
      the ``"v2d"`` / ``"v"`` analogue).
    """
    cfg_var_u = getattr(mode_cfg, "var_u", None) if mode_cfg else None
    cfg_var_v = getattr(mode_cfg, "var_v", None) if mode_cfg else None
    if getattr(mode_cfg, "Fdrift3D", False):
        return (cfg_var_u or "u"), (cfg_var_v or "v")
    u_name = cfg_var_u or ("u2d" if "u2d" in ds else "u")
    v_name = cfg_var_v or ("v2d" if "v2d" in ds else "v")
    return u_name, v_name


def _sanitize_connectivity(nv):
    """Convert 1-based padded connectivity to 0-based with -1 fill."""
    conn = np.asarray(nv, dtype=np.int64)
    conn = np.where(conn <= 0, -1, conn - 1)
    clean = -np.ones_like(conn, dtype=np.int32)
    for i in range(conn.shape[0]):
        seen = []
        for j in range(conn.shape[1]):
            node = int(conn[i, j])
            if node < 0 or node in seen:
                continue
            seen.append(node)
        clean[i, :len(seen)] = seen
    return clean


def _reduce_to_2d(arr, mesh, sigma_layer=None, sigma_mean=False):
    """Reduce a potentially 3D velocity array to 2D (one value per element).

    Parameters
    ----------
    arr : np.ndarray
        Velocity data, shape ``(nele,)`` or ``(nele, sigmam1)`` or ``(sigmam1, nele)``.
    mesh : Mesh
        Mesh with ``sigma`` levels for computing layer thicknesses.
    sigma_layer : int or None
        If set, extract this sigma layer index (0 = surface).
    sigma_mean : bool
        If True, compute sigma-weighted vertical mean.

    Returns
    -------
    np.ndarray, shape ``(nele,)``
    """
    if arr.ndim == 1:
        return np.asarray(arr, dtype=np.float64)

    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D velocity array, got shape {arr.shape}")

    # Detect which axis is sigma: its size should be len(sigma) - 1
    n_sigma_layers = len(mesh.sigma) - 1 if len(mesh.sigma) > 1 else 0

    if n_sigma_layers > 0 and arr.shape[0] == n_sigma_layers and arr.shape[1] != n_sigma_layers:
        sigma_axis = 0  # (sigmam1, nele)
    elif n_sigma_layers > 0 and arr.shape[1] == n_sigma_layers and arr.shape[0] != n_sigma_layers:
        sigma_axis = 1  # (nele, sigmam1)
    else:
        # Ambiguous or no sigma info — assume last axis is sigma
        sigma_axis = 1 if arr.shape[0] == mesh.n_elements else 0

    if sigma_mean:
        dz = np.diff(mesh.sigma)
        weights = np.abs(dz) / np.abs(dz).sum()
        if sigma_axis == 0:
            return np.asarray(weights @ arr, dtype=np.float64)
        else:
            return np.asarray(arr @ weights, dtype=np.float64)

    # Layer selection (default: surface = layer 0)
    layer_idx = sigma_layer if sigma_layer is not None else 0
    if sigma_axis == 0:
        return np.asarray(arr[layer_idx], dtype=np.float64)
    else:
        return np.asarray(arr[:, layer_idx], dtype=np.float64)


def _reduce_scalar_to_2d(arr_kn, mesh, sigma_layer=None, sigma_mean=False):
    """Node-based (K-1, N) -> (N,) vertical reduction.

    ``_reduce_to_2d`` is element-hardcoded (detects the sigma axis via
    ``shape[0] == mesh.n_elements``) and must NOT be used for node fields.
    Order: layer pick, then thickness-weighted mean, else surface layer
    (``_reduce_to_2d`` checks the mean first; the configuration refuses
    ``sigma_layer`` together with ``sigma_mean``, so the two never meet).
    """
    if arr_kn.ndim == 1:
        return np.asarray(arr_kn, dtype=np.float64)
    if sigma_layer is not None:
        return np.asarray(arr_kn[sigma_layer], dtype=np.float64)
    if sigma_mean:
        thick = np.abs(np.diff(mesh.sigma))
        return np.average(arr_kn, axis=0, weights=thick)
    return np.asarray(arr_kn[0], dtype=np.float64)


def _reduce_node_layered_to_2d(arr, dims, mesh, sigma_layer=None,
                               sigma_mean=False):
    """Reduce a node-layout velocity array to a single value per node.

    The time axis has already been ``isel``'d away, so ``arr`` is ``(N,)``
    (already 2D-reduced — returned unchanged) or 2D with one vertical and
    one node axis.  ``_reduce_scalar_to_2d`` assumes the vertical axis is
    FIRST (``(K-1, N)``); a node-layout var may instead arrive
    ``(node, sigma...)``, so orient by the (post-time-isel) dim names and
    transpose when the vertical dim is last.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    vert_dims = ("sigma_lay", "sigmam1", "sigma", "nz1", "nz",
                 "depth", "level")
    # ``dims`` still includes "time"; drop it to match the post-isel array.
    post = [d for d in dims if d != "time"]
    if len(post) == 2 and post[1] in vert_dims and post[0] not in vert_dims:
        arr = np.ascontiguousarray(arr.T)        # (N, K-1) -> (K-1, N)
    return _reduce_scalar_to_2d(arr, mesh, sigma_layer, sigma_mean)


def _normalize_layer_field(arr, n_layers, n_items, name, dims=None):
    """Normalize a 2D layer-centered field to shape ``(n_layers, n_items)``.

    When xarray dim names are available, use them to determine axis order.
    In both cases, validate the resulting shape against ``(n_layers, n_items)``
    to catch mis-staggered fields (e.g. ``sigma`` dim on a layer variable).
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D in 3D mode, got shape {arr.shape}")
    sigma_dims = {"sigmam1", "sigma", "nz", "nz1", "depth", "level"}
    if dims is not None and len(dims) == 2:
        if dims[0] in sigma_dims and dims[1] not in sigma_dims:
            result = arr
        elif dims[1] in sigma_dims and dims[0] not in sigma_dims:
            result = np.asarray(arr.T, dtype=np.float64)
        else:
            result = None
        if result is not None:
            if result.shape != (n_layers, n_items):
                raise ValueError(
                    f"{name}: dim-name layout gives shape {result.shape} "
                    f"but expected ({n_layers}, {n_items}); check that the "
                    f"variable has the correct vertical stagger (sigmam1 for "
                    f"layers, sigma for interfaces)"
                )
            # C-order explicitly: ``arr.T`` is an F-ordered VIEW and would
            # drop the parallel whole-field time blends to their serial
            # numpy fallback downstream (2026-08 review).  No-op when C.
            return np.ascontiguousarray(result)
    if arr.shape == (n_layers, n_items):
        return np.ascontiguousarray(arr)
    if arr.shape == (n_items, n_layers):
        return np.ascontiguousarray(arr.T)
    raise ValueError(
        f"{name} has unexpected shape {arr.shape}; expected "
        f"({n_layers}, {n_items}) or ({n_items}, {n_layers})"
    )


def _normalize_interface_field(arr, n_sigma, n_items, name, dims=None):
    """Normalize a 2D interface-centered field to shape ``(n_sigma, n_items)``.

    Same logic as ``_normalize_layer_field`` but expects interface count.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D in 3D mode, got shape {arr.shape}")
    sigma_dims = {"sigma", "sigmam1", "nz", "nz1", "depth", "level"}
    if dims is not None and len(dims) == 2:
        if dims[0] in sigma_dims and dims[1] not in sigma_dims:
            result = arr
        elif dims[1] in sigma_dims and dims[0] not in sigma_dims:
            result = np.asarray(arr.T, dtype=np.float64)
        else:
            result = None
        if result is not None:
            if result.shape != (n_sigma, n_items):
                raise ValueError(
                    f"{name}: dim-name layout gives shape {result.shape} "
                    f"but expected ({n_sigma}, {n_items}); check that the "
                    f"variable has the correct vertical stagger (sigma for "
                    f"interfaces, sigmam1 for layers)"
                )
            # C-order explicitly (see _normalize_layer_field).
            return np.ascontiguousarray(result)
    if arr.shape == (n_sigma, n_items):
        return np.ascontiguousarray(arr)
    if arr.shape == (n_items, n_sigma):
        return np.ascontiguousarray(arr.T)
    raise ValueError(
        f"{name} has unexpected shape {arr.shape}; expected "
        f"({n_sigma}, {n_items}) or ({n_items}, {n_sigma})"
    )



@register("fesomc")
class FesomCReader:
    @staticmethod
    def load_mesh(mesh_cfg):
        # Coordinate-frame rotation is not supported by the FESOM-C reader (FESOM2/HEALPix support it) by design; silently
        # accepting it here would imply an unimplemented vector-rotation
        # path (FESOM-C is a coastal model — no near-pole use case).
        if getattr(mesh_cfg, "rotation", None) is not None:
            raise ValueError(
                "mesh.rotation is not supported by the FESOM-C reader (FESOM2 and HEALPix support it) (FESOM-C is a coastal model; "
                "its reader has no vector-rotation path)")

        path = Path(mesh_cfg.meshpath).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"FESOM-C mesh file not found: {path}")

        with xr.open_dataset(path, engine=_detect_engine(path)) as ds:
            lon = np.asarray(ds["lon"].values, dtype=np.float64)
            lat = np.asarray(ds["lat"].values, dtype=np.float64)

            # Near-pole warning (no rotation offered for FESOM-C).
            max_abs_lat = float(np.abs(lat).max())
            if max_abs_lat > POLE_WARN_LAT:
                warnings.warn(
                    f"mesh reaches |lat| = {max_abs_lat:.2f} deg - the "
                    "lon/lat step metric degenerates near the pole; results "
                    "there are unreliable.",
                    RuntimeWarning, stacklevel=2)
            depth = np.asarray(ds["depth"].values, dtype=np.float64) if "depth" in ds else np.zeros_like(lon)
            conn = _sanitize_connectivity(ds["nv"].values)
            # Load-time mesh-contract check (2026-07): the guard-free
            # product-form Wachspress weights assume convex quads — fail
            # loud on a malformed file instead of NaN weights mid-run.
            validate_quad_convexity(conn, lon, lat,
                                    source=f"FESOM-C mesh {path}")

            if "lon_elem" in ds and "lat_elem" in ds:
                lon_elem = np.asarray(ds["lon_elem"].values, dtype=np.float64)
                lat_elem = np.asarray(ds["lat_elem"].values, dtype=np.float64)
            else:
                def _center_lon(row):
                    # Seam gate (idiom of mesh._project_to_local_xy): a raw
                    # arithmetic mean puts the center of an antimeridian-
                    # straddling cell ~180 deg away (e.g. -59.8 for
                    # [179.5, -179.5, -179.5]), corrupting every consumer of
                    # elem_center_lon.  Anchor on the first vertex and
                    # average WRAPPED deltas only when the raw span exceeds
                    # 180 deg — off-seam cells keep the raw mean bitwise.
                    # The value is correct modulo 360; since 2026-07 every
                    # consumer wraps its deltas (the default leastsq cores
                    # gained a gated antimeridian correction), so the
                    # branch the value lands in no longer matters.
                    valid = row[row >= 0]
                    if valid.size == 0:
                        return 0.0
                    lo = lon[valid]
                    if lo.max() - lo.min() <= 180.0:
                        return lo.mean()
                    ref = lo[0]
                    d = ((lo - ref + 180.0) % 360.0) - 180.0
                    return ((ref + d.mean() + 180.0) % 360.0) - 180.0

                lon_elem = np.array([_center_lon(row) for row in conn],
                                    dtype=np.float64)
                lat_elem = np.array([lat[row[row >= 0]].mean() if np.any(row >= 0) else 0.0 for row in conn], dtype=np.float64)

            if "sigma_lev" in ds:
                sigma = np.asarray(ds["sigma_lev"].values, dtype=np.float64)
            elif "sigma" in ds and ds["sigma"].ndim == 1:
                sigma = np.asarray(ds["sigma"].values, dtype=np.float64)
            else:
                sigma = np.array([0.0], dtype=np.float64)

            n_elem = conn.shape[0]
            n_nodes = lon.shape[0]
            # Model-provided areas take PRECEDENCE over the spherical-excess
            # fallback (bit-parity with the model build, pole-correct).
            # Production FESOM-C NetCDFs may carry element areas and/or node
            # control-volume areas; accept a candidate only when it reduces
            # (after squeezing size-1 dims) to a 1-D array of the expected
            # length.  A var with a genuine extra axis (e.g.
            # ``elem_area(time, nele)`` with time>1) is silently SKIPPED
            # (areas are optional, never raise) so it falls through to the
            # spherical-excess fallback instead of poisoning numba typing
            # downstream with a 2-D buffer.  ``copy=True``: the lazy cache
            # must OWN its buffer (it is returned by reference to consumers).
            elem_var = node_var = None
            elem_areas = None
            for cand in ("elem_area", "area_elem", "art2", "elem_areas"):
                if cand in ds:
                    da = ds[cand].squeeze()      # xarray drops size-1 dims
                    if da.ndim == 1 and da.shape[0] == n_elem:
                        elem_areas = np.array(
                            da.values, dtype=np.float64, copy=True)
                        elem_var = cand
                        break
            node_areas = None
            for cand in ("area", "art1", "nod_area", "node_area"):
                if cand in ds:
                    da = ds[cand].squeeze()
                    if da.ndim == 1 and da.shape[0] == n_nodes:
                        node_areas = np.array(
                            da.values, dtype=np.float64, copy=True)
                        node_var = cand
                        break

        # FESOM-C family Earth radius: explicit config override, else the
        # legacy Fortran value (matches core.geometry.R_EARTH).  Used by the
        # spherical-excess area fallback; model-provided areas ignore it.
        r_earth = (getattr(mesh_cfg, "r_earth", None) or 6_400_000.0)

        # Area-source visibility (file areas are the FESOM-C default; the
        # user wants this surfaced).  Matches the fesom2 reader's style.
        if elem_areas is not None:
            line = f"    element areas: from file ({elem_var})"
        else:
            line = ("    element areas: not in file -> "
                    "spherical fallback on demand")
        if node_areas is not None:
            line += f"; node areas: from file ({node_var})"
        print(line)

        n2e, n2e_count = build_node_to_elems(lon.shape[0], conn)
        mesh = Mesh(
            name="fesomc", nodes_lon=lon, nodes_lat=lat, depth=depth,
            elem_nodes=conn, elem_center_lon=lon_elem, elem_center_lat=lat_elem,
            node_to_elems=n2e, node_to_elems_count=n2e_count, sigma=sigma,
            vertical_coord="sigma", r_earth=r_earth,
            metadata={"path": str(path), "sigma_transform": "linear_fesomc"},
        )
        if elem_areas is not None:
            mesh._elem_areas = elem_areas
        if node_areas is not None:
            mesh._node_areas = node_areas
        build_search(mesh)
        return mesh

    @staticmethod
    def open_dataset(path, v_path=None, w_path=None, mode_cfg=None):
        """Open FESOM-C velocity dataset.

        Parameters
        ----------
        path : str
            Path to NetCDF file (single combined file, or u-only file).
        v_path : str or None
            If given, path to a separate v NetCDF file. The two datasets
            are merged so that ``read_velocities`` sees all variables.
        mode_cfg : ModeConfig or None
            Accepted for a uniform reader interface; ignored here (FESOM-C
            output is a single combined file, no multi-year per-variable
            layout).

        Returns
        -------
        xr.Dataset
        """
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

        base = xr.open_dataset(p, engine=_detect_engine(p))
        datasets = [base]

        # The merge aligns the files on their time LABELS: a file on another
        # record axis would be padded with NaN (or fail on the dtype), so
        # each attached file must share the velocity file's records.
        for label, extra in (("v", v_path), ("w", w_path)):
            if extra is None:
                continue
            xp = Path(extra).expanduser()
            if not xp.exists():
                raise FileNotFoundError(f"{label.upper()} file not found: {xp}")
            ds_x = xr.open_dataset(xp, engine=_detect_engine(xp))
            try:
                ds_x = _align_onto_time_axis(base, ds_x, f"{label} file {xp}",
                                             f"the velocity file {p}",
                                             require_time=True)
            except ValueError:
                for d in datasets + [ds_x]:
                    d.close()
                raise
            datasets.append(ds_x)

        if len(datasets) == 1:
            return datasets[0]

        return xr.merge(datasets, compat="override")

    @staticmethod
    def read_scalar_fields(ds, timestep, mesh, varnames, mode_cfg=None):
        """Read node-based scalar fields for trajectory sampling (TODO 2.4).

        Returns ``{name: (N,) | (K-1, N) | (K, N) float64}``.  Layer fields
        are trimmed to ``mesh.sigma.size - 1`` and interface fields to
        ``mesh.sigma.size`` (``mesh.sigma`` is already max_levels-trimmed by
        ``run_tracking``).  In 2D mode (``not Fdrift3D``) a layered field is
        vertically reduced via :func:`_reduce_scalar_to_2d` (layer pick →
        thickness-weighted mean → surface, mirroring ``_reduce_to_2d``).

        Raises ``ValueError`` for missing or non-node-based variables.
        """
        is_3d = mode_cfg is not None and getattr(mode_cfg, "Fdrift3D", False)
        N = mesh.n_nodes
        K_t = mesh.sigma.size            # trimmed interface count
        # Direct/public callers may pass an untrimmed mesh.sigma with
        # max_levels set; read_velocities slices u/v/w by max_levels directly,
        # so trim scalars the same way to keep the VelocityField consistent.
        ml = getattr(mode_cfg, "max_levels", None) if mode_cfg is not None else None
        if is_3d and ml is not None and K_t > 0:
            K_t = min(K_t, int(ml))
        out = {}
        # Layer dims first (sigma_lay / sigmam1 are FESOM-C layer-centred),
        # then interface (sigma) and generic vertical names.
        layer_dims = ("sigma_lay", "sigmam1", "nz1")
        iface_dims = ("sigma", "nz")
        vdims = layer_dims + iface_dims + ("depth", "level")
        for name in varnames:
            if name == "w_cart":
                # mids stored on the interface dimension (last level unset):
                # the scalar path would misplace it by half a layer, and it
                # is not the vertical velocity anyway (review 2026-09-23)
                raise ValueError(
                    "sample variable 'w_cart' is refused: FESOM-C stores it "
                    "at the layer mids on the interface dimension and it "
                    "lacks the dia-sigma velocity; the trajectory output "
                    "records the tracker's vertical rate as vel_w.")
            if name not in ds:
                raise ValueError(
                    f"sample variable '{name}' not found in dataset")
            da = ds[name]
            if "time" in da.dims:
                da = da.isel(time=timestep)
            vdim = next((d for d in vdims if d in da.dims), None)
            arr = np.asarray(da.values, dtype=np.float64)
            # Orient so the vertical axis is first: (N, K) -> (K, N).
            if vdim is not None and arr.ndim == 2:
                if da.dims.index(vdim) == 1:
                    arr = np.ascontiguousarray(arr.T)
            # Trim the vertical axis consistently with mesh.sigma.
            if vdim is not None and arr.ndim == 2 and K_t > 0:
                if vdim in iface_dims:
                    keep = min(K_t, arr.shape[0])            # interface field
                elif vdim in layer_dims:
                    keep = min(K_t - 1, arr.shape[0])        # layer field
                else:                                         # depth/level: size
                    keep = min(K_t if arr.shape[0] >= K_t else K_t - 1,
                               arr.shape[0])
                arr = arr[:keep]
            # 2D mode: collapse the vertical axis to a single node field.
            if not is_3d and arr.ndim == 2:
                sigma_layer = getattr(mode_cfg, "sigma_layer", None)
                sigma_mean = getattr(mode_cfg, "sigma_mean", False)
                arr = _reduce_scalar_to_2d(arr, mesh, sigma_layer, sigma_mean)
            if arr.shape[-1] != N:
                raise ValueError(
                    f"sample variable '{name}' is not node-based: shape "
                    f"{arr.shape}, expected last axis {N} "
                    "(element-based fields are not supported)")
            out[name] = np.ascontiguousarray(arr)
        return out

    @staticmethod
    def read_velocities(ds, timestep, mesh, mode_cfg, diffusion=None,
                        sample_vars=None, velocity_float32=False):
        # u/v/w dtype: float32 (native) halves snapshot memory (TODO 5.3);
        # default float64 (byte-identical — kernels accumulate in float64).
        # Caveat: 2D fields built via sigma_mean / Vpuls are computed in float64
        # then downcast here, so under float32 they store a rounded intermediate
        # (not bit-identical); FESOM2 direct reads + FESOM-C direct 2D are exact.
        vd = np.float32 if velocity_float32 else np.float64
        # Resolve optional config fields (backward-compat with SimpleNamespace)
        cfg_var_w = getattr(mode_cfg, "var_w", None)
        cfg_var_eta = getattr(mode_cfg, "var_eta", None)
        cfg_sigma_layer = getattr(mode_cfg, "sigma_layer", None)
        cfg_sigma_mean = getattr(mode_cfg, "sigma_mean", False)
        cfg_max_levels = getattr(mode_cfg, "max_levels", None)

        # -- eta --
        eta_name = cfg_var_eta or "eta"
        if eta_name in ds:
            eta = np.asarray(ds[eta_name].isel(time=timestep).values, dtype=np.float64)
        else:
            eta = np.zeros(mesh.n_nodes, dtype=np.float64)
            if mode_cfg.Fdrift3D and not mesh.metadata.get("_warned_missing_eta", False):
                warnings.warn(
                    "FESOM-C 3D sigma tracking: eta is missing, using eta=0 "
                    "(fixed-column geometry) for this run.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                mesh.metadata["_warned_missing_eta"] = True

        # -- 3D mode --
        if mode_cfg.Fdrift3D:
            u_name, v_name = _resolve_uv_names(ds, mode_cfg)
            w_name = cfg_var_w or "w"
            if w_name == "w_cart" or (w_name not in ds and "w_cart" in ds):
                raise ValueError(
                    "FESOM-C 3-D: 'w_cart' cannot drive the tracker — the "
                    "model's vert_vel_cart (fv_3D_vel.F90) builds it with "
                    "z_mid*(Wvel_t - Wvel_b) instead of z_t*Wvel_t - "
                    "z_b*Wvel_b, so it lacks the dia-sigma part of the "
                    "vertical velocity.  Write the sigma-coordinate vertical "
                    "velocity 'w' (Wvel, on the sigma interfaces) and track "
                    "with it (mode.var_w: w, the default).")

            if u_name not in ds:
                raise KeyError(f"FESOM-C 3D variable '{u_name}' not found")
            if v_name not in ds:
                raise KeyError(f"FESOM-C 3D variable '{v_name}' not found")
            if w_name not in ds:
                raise KeyError(
                    f"FESOM-C 3D vertical velocity variable '{w_name}' not found"
                )

            if cfg_max_levels is not None and cfg_max_levels < 2:
                raise ValueError(
                    f"max_levels must be >= 2 in 3D mode, got {cfg_max_levels}"
                )

            u_da = ds[u_name].isel(time=timestep)
            v_da = ds[v_name].isel(time=timestep)
            w_da = ds[w_name].isel(time=timestep)

            # Classify u/v mesh location (element centres vs nodes) so the
            # layer-field normalization validates against the right item
            # count.  u and v must agree.
            uv_layout = _detect_field_layout(u_da, mesh)
            v_layout = _detect_field_layout(v_da, mesh)
            if uv_layout != v_layout:
                raise ValueError(
                    f"u/v layout disagreement: {u_name!r} is {uv_layout} but "
                    f"{v_name!r} is {v_layout}; both must be on the same mesh "
                    "location (element or node)."
                )
            uv_n_items = mesh.n_nodes if uv_layout == "node" else mesh.n_elements

            # Derive full sigma interface count from the dataset, not the
            # potentially-trimmed mesh.sigma (tracking.py may have sliced
            # mesh.sigma to max_levels before this reader call).
            n_sigma_full = None
            if "sigma_lev" in ds:
                n_sigma_full = ds["sigma_lev"].shape[0]
            else:
                # Fallback: locate the W variable's vertical axis by name
                # (don't use max(shape) — n_nodes >> n_sigma typically,
                # so max would return the node count).
                for d in w_da.dims:
                    if d == "sigma":
                        n_sigma_full = w_da.sizes[d]
                        break
                if n_sigma_full is None:
                    # Last resort: try u/v's layer dim and add 1
                    for d in u_da.dims:
                        if d == "sigmam1":
                            n_sigma_full = u_da.sizes[d] + 1
                            break
            if n_sigma_full is None:
                raise ValueError(
                    "FESOM-C 3D: cannot infer sigma interface count. "
                    "Dataset must contain 'sigma_lev', or 'w_cart' must have "
                    "a 'sigma' dim, or 'u/v' must have a 'sigmam1' dim."
                )
            n_layers_full = max(n_sigma_full - 1, 0)

            # Normalize against FULL sigma count first — this validates
            # stagger correctness (e.g. rejects w on sigmam1).  Slicing
            # to max_levels happens after validation.
            u = _normalize_layer_field(
                u_da.values, n_layers_full, uv_n_items, u_name, dims=u_da.dims
            )
            v = _normalize_layer_field(
                v_da.values, n_layers_full, uv_n_items, v_name, dims=v_da.dims
            )
            w = _normalize_interface_field(
                w_da.values, n_sigma_full, mesh.n_nodes, w_name, dims=w_da.dims
            )
            # FESOM-C's w = Wvel: the flux through the sigma interfaces,
            # positive UP (fv_3D_vel.F90:vert_vel_sigma).  The tracker's W
            # on a sigma mesh is that flux positive DOWN; the integrators
            # advance the particle's sigma fraction s = z / H with
            # ds/dt = W / H and place it at z = s H (docs/vertical.md §3),
            # which carries the moving surface and the sloping bottom.
            w = resolve_w_sign(w_da, mode_cfg, "up") * np.asarray(
                w, dtype=np.float64)

            # Slice to max_levels after normalization
            if cfg_max_levels is not None:
                n_sigma_keep = min(int(cfg_max_levels), n_sigma_full)
                n_layers_keep = max(n_sigma_keep - 1, 0)
                u = u[:n_layers_keep]
                v = v[:n_layers_keep]
                w = w[:n_sigma_keep]

            kh = _read_kh_fields(ds, timestep, mesh, diffusion)
            kh_z, kh_m, kh_layout = kh if kh is not None else (None, None, None)
            scalars = None
            if sample_vars:
                scalars = FesomCReader.read_scalar_fields(
                    ds, timestep, mesh, sample_vars, mode_cfg)
            return VelocityField(
                u=u.astype(vd, copy=False), v=v.astype(vd, copy=False),
                eta=eta,
                w=(w.astype(vd, copy=False) if w is not None else None),
                kh_zonal=kh_z, kh_meridional=kh_m, kh_layout=kh_layout,
                uv_layout=uv_layout,
                scalars=scalars,
            )

        # -- 2D mode: resolve variable names --
        u_name, v_name = _resolve_uv_names(ds, mode_cfg)

        u_da = ds[u_name].isel(time=timestep)
        v_da = ds[v_name].isel(time=timestep)

        # Classify u/v mesh location (element centres vs nodes).  u and v
        # must agree; a node-layout layered var needs the node-aware
        # vertical reduction (_reduce_to_2d is element-hardcoded).
        uv_layout = _detect_field_layout(u_da, mesh)
        v_layout = _detect_field_layout(v_da, mesh)
        if uv_layout != v_layout:
            raise ValueError(
                f"u/v layout disagreement: {u_name!r} is {uv_layout} but "
                f"{v_name!r} is {v_layout}; both must be on the same mesh "
                "location (element or node)."
            )

        u_raw = np.asarray(u_da.values, dtype=np.float64)
        v_raw = np.asarray(v_da.values, dtype=np.float64)

        # -- 3D→2D reduction if needed --
        if uv_layout == "node":
            u = _reduce_node_layered_to_2d(
                u_raw, u_da.dims, mesh, cfg_sigma_layer, cfg_sigma_mean)
            v = _reduce_node_layered_to_2d(
                v_raw, v_da.dims, mesh, cfg_sigma_layer, cfg_sigma_mean)
        else:
            u = _reduce_to_2d(u_raw, mesh, cfg_sigma_layer, cfg_sigma_mean)
            v = _reduce_to_2d(v_raw, mesh, cfg_sigma_layer, cfg_sigma_mean)

        # -- pulsation velocity --
        if mode_cfg.Vpuls:
            if "u2d_puls" in ds:
                u = u + np.asarray(ds["u2d_puls"].isel(time=timestep).values, dtype=np.float64)
            if "v2d_puls" in ds:
                v = v + np.asarray(ds["v2d_puls"].isel(time=timestep).values, dtype=np.float64)

        kh = _read_kh_fields(ds, timestep, mesh, diffusion)
        kh_z, kh_m, kh_layout = kh if kh is not None else (None, None, None)
        scalars = None
        if sample_vars:
            scalars = FesomCReader.read_scalar_fields(
                ds, timestep, mesh, sample_vars, mode_cfg)
        return VelocityField(
            u=u.astype(vd, copy=False), v=v.astype(vd, copy=False), eta=eta,
            kh_zonal=kh_z, kh_meridional=kh_m, kh_layout=kh_layout,
            uv_layout=uv_layout,
            scalars=scalars,
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
