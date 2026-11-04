#!/usr/bin/env python3
"""Offline FESOM2 regional subsetting utility.

This utility is intentionally self-contained under ``utils/`` so it can be
developed independently from the main kelp runtime. It extracts a regional
subset from a large FESOM2 mesh/velocity dataset and writes the result to
NetCDF or Zarr with metadata and provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import xarray as xr
import yaml


EARTH_RADIUS_KM = 6371.0
U_ALIASES = ("u", "ua", "u_vel")
V_ALIASES = ("v", "va", "v_vel")
W_ALIASES = ("w", "wo", "wvel")
SSH_ALIASES = ("ssh", "eta", "zos")
TEMP_ALIASES = ("temp", "temperature", "thetao", "t", "T")
SALT_ALIASES = ("salt", "salinity", "so", "s", "S")


@dataclass(frozen=True)
class ExtraVarSpec:
    """One extra variable to extract (e.g. temperature, salinity)."""

    path: str                        # file path (NetCDF or Zarr)
    var: Optional[str] = None        # variable name; auto-detected if None
    aliases: Tuple[str, ...] = ()    # fallback aliases for auto-detection
    role: Optional[str] = None       # output name; defaults to var name


@dataclass(frozen=True)
class SourceConfig:
    """Input data source configuration."""

    mesh: str
    input: Optional[str] = None
    u_path: Optional[str] = None
    v_path: Optional[str] = None
    w_path: Optional[str] = None
    ssh_path: Optional[str] = None
    extra_vars: Tuple[ExtraVarSpec, ...] = ()


@dataclass(frozen=True)
class RegionSpec:
    """Requested region for subsetting."""

    mode: str = "bbox"
    bbox: Optional[Tuple[float, float, float, float]] = None
    center: Optional[Tuple[float, float]] = None
    radius_km: Optional[float] = None
    polygon_path: Optional[str] = None
    polygon_format: Optional[str] = None
    halo_layers: int = 0


@dataclass(frozen=True)
class TimeConfig:
    """Time slicing for velocity data."""

    start: Optional[int] = None
    stop: Optional[int] = None
    step: int = 1


@dataclass(frozen=True)
class OutputConfig:
    """Output file layout configuration."""

    dir: str
    format: str = "netcdf"
    layout: str = "split"
    copy_source_attrs: bool = True
    write_config_copy: bool = True
    embed_config_text: bool = True
    write_manifest: bool = True
    zarr_chunks: Optional[Dict[str, Any]] = None  # per-dim chunk sizes for zarr


@dataclass(frozen=True)
class MetadataConfig:
    """User-provided metadata for the subset outputs."""

    subset_name: str = ""
    description: str = ""
    created_by: str = "extract_fesom_region"
    notes: str = ""


@dataclass(frozen=True)
class SubsetConfig:
    """Top-level utility configuration."""

    source: SourceConfig
    region: RegionSpec
    time: TimeConfig
    output: OutputConfig
    metadata: MetadataConfig = field(default_factory=MetadataConfig)


@dataclass
class MeshSource:
    """Loaded source mesh with normalized arrays."""

    path: Path
    ds: xr.Dataset
    lon: np.ndarray
    lat: np.ndarray
    elem_nodes: np.ndarray         # (E, 3), 0-based
    node_to_elems: np.ndarray      # (N, max_conn), 0-based, -1 padded
    node_to_elems_count: np.ndarray
    depth: np.ndarray
    sigma: np.ndarray
    elem_center_lon: np.ndarray
    elem_center_lat: np.ndarray

    @property
    def n_nodes(self) -> int:
        return int(self.lon.shape[0])

    @property
    def n_elements(self) -> int:
        return int(self.elem_nodes.shape[0])


@dataclass
class VarSource:
    """One input velocity-like variable and its source dataset."""

    role: str
    path: Path
    dataset: xr.Dataset
    var_name: str
    data: xr.DataArray


@dataclass(frozen=True)
class SubsetSelection:
    """Selected nodes/elements in source indexing."""

    elem_ids: np.ndarray
    node_ids: np.ndarray
    elem_global_to_local: np.ndarray
    node_global_to_local: np.ndarray


@dataclass
class SubsetArtifacts:
    """Written output files/stores."""

    mesh_path: Path
    velocity_paths: Dict[str, Path]
    manifest_path: Optional[Path] = None
    config_copy_path: Optional[Path] = None


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    p = argparse.ArgumentParser(
        description="Extract a regional FESOM2 subset from large mesh/velocity inputs."
    )
    p.add_argument("--config", help="Path to YAML config for this utility.")
    p.add_argument("--mesh", help="Path to source FESOM2 mesh NetCDF.")
    p.add_argument(
        "--input",
        help="Combined velocity file or a directory with u/v/w/ssh NetCDF files.",
    )
    p.add_argument("--u-path", help="Optional path to U velocity file.")
    p.add_argument("--v-path", help="Optional path to V velocity file.")
    p.add_argument("--w-path", help="Optional path to W velocity file.")
    p.add_argument("--ssh-path", help="Optional path to SSH file.")
    p.add_argument("--output-dir", help="Output directory.")
    p.add_argument(
        "--output-format",
        choices=("netcdf", "zarr"),
        help="Subset output format.",
    )
    p.add_argument(
        "--output-layout",
        choices=("split", "combined"),
        help="Velocity output layout.",
    )
    p.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MIN_LON", "MAX_LON", "MIN_LAT", "MAX_LAT"),
        help="Bounding box region.",
    )
    p.add_argument(
        "--center",
        nargs=2,
        type=float,
        metavar=("LON", "LAT"),
        help="Region center for radial selection.",
    )
    p.add_argument("--radius-km", type=float, help="Radius for radial selection.")
    p.add_argument("--polygon-path", help="Path to polygon definition.")
    p.add_argument(
        "--polygon-format",
        choices=("geojson", "yaml", "ascii"),
        help="Polygon input format.",
    )
    p.add_argument(
        "--region-mode",
        choices=("bbox", "center_radius", "polygon"),
        help="Region selection mode.",
    )
    p.add_argument(
        "--halo-layers",
        type=int,
        help="Number of neighboring element layers to add.",
    )
    p.add_argument("--time-start", type=int, help="First timestep to copy.")
    p.add_argument("--time-stop", type=int, help="Stop timestep (exclusive).")
    p.add_argument("--time-step", type=int, help="Temporal stride.")
    return p


def _deepcopy_jsonable(data: Mapping[str, Any]) -> dict:
    return json.loads(json.dumps(data))


def _read_yaml(path: Path) -> dict:
    try:
        return yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Config file not found: {path}") from exc


def _path_value(value: Optional[str], base_dir: Optional[Path]) -> Optional[str]:
    if value in (None, "", "null"):
        return None
    p = Path(value).expanduser()
    if not p.is_absolute() and base_dir is not None:
        p = (base_dir / p).resolve()
    return str(p)


def _coerce_bbox(value: Any) -> Optional[Tuple[float, float, float, float]]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return (
            float(value["min_lon"]),
            float(value["max_lon"]),
            float(value["min_lat"]),
            float(value["max_lat"]),
        )
    if isinstance(value, Sequence) and len(value) == 4:
        return tuple(float(v) for v in value)  # type: ignore[return-value]
    raise ValueError(f"Invalid bbox specification: {value!r}")


def _coerce_center(value: Any) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return (float(value["lon"]), float(value["lat"]))
    if isinstance(value, Sequence) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    raise ValueError(f"Invalid center specification: {value!r}")


def _merge_cli_overrides(raw_cfg: dict, args: argparse.Namespace) -> dict:
    cfg = _deepcopy_jsonable(raw_cfg)
    cfg.setdefault("source", {})
    cfg.setdefault("region", {})
    cfg.setdefault("time", {})
    cfg.setdefault("output", {})
    cfg.setdefault("metadata", {})

    def set_if(section: str, key: str, value: Any) -> None:
        if value is not None:
            cfg.setdefault(section, {})
            cfg[section][key] = value

    set_if("source", "mesh", args.mesh)
    set_if("source", "input", args.input)
    set_if("source", "u_path", args.u_path)
    set_if("source", "v_path", args.v_path)
    set_if("source", "w_path", args.w_path)
    set_if("source", "ssh_path", args.ssh_path)

    set_if("output", "dir", args.output_dir)
    set_if("output", "format", args.output_format)
    set_if("output", "layout", args.output_layout)

    set_if("region", "mode", args.region_mode)
    set_if("region", "bbox", args.bbox)
    set_if("region", "center", args.center)
    set_if("region", "radius_km", args.radius_km)
    set_if("region", "polygon_path", args.polygon_path)
    set_if("region", "polygon_format", args.polygon_format)
    set_if("region", "halo_layers", args.halo_layers)

    set_if("time", "start", args.time_start)
    set_if("time", "stop", args.time_stop)
    set_if("time", "step", args.time_step)
    return cfg


def load_subset_config(args: argparse.Namespace) -> tuple[SubsetConfig, dict, Optional[Path], str]:
    """Load and validate config from YAML plus CLI overrides."""
    config_path = Path(args.config).expanduser().resolve() if args.config else None
    raw = _read_yaml(config_path) if config_path else {}
    merged = _merge_cli_overrides(raw, args)
    base_dir = config_path.parent if config_path else Path.cwd()

    if "source" not in merged or "output" not in merged:
        raise ValueError("Config must define at least 'source' and 'output' sections.")

    src_raw = merged.get("source", {})
    out_raw = merged.get("output", {})
    region_raw = merged.get("region", {})
    time_raw = merged.get("time", {})
    meta_raw = merged.get("metadata", {})

    # --- well-known aliases for extra_vars shorthand ---
    _WELL_KNOWN_ALIASES: Dict[str, Tuple[str, ...]] = {
        "temp": TEMP_ALIASES,
        "temperature": TEMP_ALIASES,
        "thetao": TEMP_ALIASES,
        "salt": SALT_ALIASES,
        "salinity": SALT_ALIASES,
        "so": SALT_ALIASES,
    }

    # Parse extra_vars list
    extra_specs: list[ExtraVarSpec] = []
    for ev in src_raw.get("extra_vars", []):
        if isinstance(ev, str):
            # shorthand: just a path
            extra_specs.append(ExtraVarSpec(path=_path_value(ev, base_dir) or ev))
        elif isinstance(ev, Mapping):
            ev_path = _path_value(ev.get("path"), base_dir) or ""
            ev_var = ev.get("var")
            ev_role = ev.get("role")
            # Build aliases: explicit list, or well-known, or empty
            if "aliases" in ev:
                ev_aliases = tuple(ev["aliases"])
            elif ev_var and ev_var.lower() in _WELL_KNOWN_ALIASES:
                ev_aliases = _WELL_KNOWN_ALIASES[ev_var.lower()]
            elif ev_role and ev_role.lower() in _WELL_KNOWN_ALIASES:
                ev_aliases = _WELL_KNOWN_ALIASES[ev_role.lower()]
            else:
                ev_aliases = (ev_var,) if ev_var else ()
            extra_specs.append(ExtraVarSpec(
                path=ev_path, var=ev_var, aliases=ev_aliases, role=ev_role,
            ))
        else:
            raise ValueError(f"Invalid extra_vars entry: {ev!r}")

    source = SourceConfig(
        mesh=_path_value(src_raw.get("mesh"), base_dir) or "",
        input=_path_value(src_raw.get("input"), base_dir),
        u_path=_path_value(src_raw.get("u_path"), base_dir),
        v_path=_path_value(src_raw.get("v_path"), base_dir),
        w_path=_path_value(src_raw.get("w_path"), base_dir),
        ssh_path=_path_value(src_raw.get("ssh_path"), base_dir),
        extra_vars=tuple(extra_specs),
    )
    region = RegionSpec(
        mode=str(region_raw.get("mode", "bbox")),
        bbox=_coerce_bbox(region_raw.get("bbox")),
        center=_coerce_center(region_raw.get("center")),
        radius_km=float(region_raw["radius_km"]) if region_raw.get("radius_km") is not None else None,
        polygon_path=_path_value(region_raw.get("polygon_path"), base_dir),
        polygon_format=str(region_raw["polygon_format"]) if region_raw.get("polygon_format") else None,
        halo_layers=int(region_raw.get("halo_layers", 0) or 0),
    )
    time_cfg = TimeConfig(
        start=int(time_raw["start"]) if time_raw.get("start") is not None else None,
        stop=int(time_raw["stop"]) if time_raw.get("stop") is not None else None,
        step=int(time_raw.get("step", 1) or 1),
    )
    raw_chunks = out_raw.get("zarr_chunks")
    zarr_chunks = dict(raw_chunks) if isinstance(raw_chunks, Mapping) else None
    output = OutputConfig(
        dir=_path_value(out_raw.get("dir"), base_dir) or "",
        format=str(out_raw.get("format", "netcdf")),
        layout=str(out_raw.get("layout", "split")),
        copy_source_attrs=bool(out_raw.get("copy_source_attrs", True)),
        write_config_copy=bool(out_raw.get("write_config_copy", True)),
        embed_config_text=bool(out_raw.get("embed_config_text", True)),
        write_manifest=bool(out_raw.get("write_manifest", True)),
        zarr_chunks=zarr_chunks,
    )
    metadata = MetadataConfig(
        subset_name=str(meta_raw.get("subset_name", "")),
        description=str(meta_raw.get("description", "")),
        created_by=str(meta_raw.get("created_by", "extract_fesom_region")),
        notes=str(meta_raw.get("notes", "")),
    )
    cfg = SubsetConfig(
        source=source,
        region=region,
        time=time_cfg,
        output=output,
        metadata=metadata,
    )
    validate_config(cfg)

    normalized_dict = asdict(cfg)
    config_text = yaml.safe_dump(normalized_dict, sort_keys=False, allow_unicode=False)
    return cfg, normalized_dict, config_path, config_text


def validate_config(cfg: SubsetConfig) -> None:
    """Validate the normalized config."""
    if not cfg.source.mesh:
        raise ValueError("source.mesh must be provided.")
    if not cfg.output.dir:
        raise ValueError("output.dir must be provided.")
    if cfg.output.format not in ("netcdf", "zarr"):
        raise ValueError(f"Unsupported output.format: {cfg.output.format}")
    if cfg.output.layout not in ("split", "combined"):
        raise ValueError(f"Unsupported output.layout: {cfg.output.layout}")
    if cfg.time.step <= 0:
        raise ValueError("time.step must be positive.")
    if cfg.region.halo_layers < 0:
        raise ValueError("region.halo_layers must be >= 0.")
    if cfg.region.mode not in ("bbox", "center_radius", "polygon"):
        raise ValueError(f"Unsupported region.mode: {cfg.region.mode}")
    if cfg.region.mode == "bbox" and cfg.region.bbox is None:
        raise ValueError("region.mode=bbox requires region.bbox.")
    if cfg.region.mode == "center_radius":
        if cfg.region.center is None or cfg.region.radius_km is None:
            raise ValueError("region.mode=center_radius requires center and radius_km.")
    if cfg.region.mode == "polygon":
        if not cfg.region.polygon_path:
            raise ValueError("region.mode=polygon requires polygon_path.")
        if (cfg.region.polygon_format or "geojson") not in ("geojson", "yaml", "ascii"):
            raise ValueError("polygon_format must be geojson, yaml, or ascii.")
    if not cfg.source.input and not (cfg.source.u_path and cfg.source.v_path):
        raise ValueError("Provide source.input or both source.u_path and source.v_path.")


def _first_existing(ds: xr.Dataset, names: Sequence[str]) -> Optional[str]:
    for name in names:
        if name in ds.variables:
            return name
    return None


def _pick_data_var(ds: xr.Dataset, aliases: Sequence[str], required: bool) -> Optional[str]:
    for name in aliases:
        if name in ds.data_vars:
            return name
    if len(ds.data_vars) == 1:
        return next(iter(ds.data_vars))
    if required:
        raise KeyError(
            f"Could not detect variable among aliases {list(aliases)} in {list(ds.data_vars)}"
        )
    return None


def _normalize_connectivity(arr: np.ndarray, start_index: int = 1) -> np.ndarray:
    """Orient triangular connectivity to ``(E, 3)`` and shift to 0-based.

    ``start_index`` is the base of the stored ids: FESOM2 writes 1; UGRID may
    declare an explicit ``start_index`` attr (0 or 1). Mirrors the runtime
    reader (``_resolve_mesh_vars``) — the base is taken from the attr, NOT
    guessed from ``min()`` (a 1-based mesh that never references node 0 would
    be mis-shifted, and a non-{0,1} base would be mis-indexed).
    """
    arr = np.asarray(arr)
    if arr.ndim != 2 or 3 not in arr.shape:
        raise ValueError(
            f"Expected triangular connectivity with a dimension of size 3, got {arr.shape}")
    elems = arr.T if (arr.shape[0] == 3 and arr.shape[1] != 3) else arr
    return (np.asarray(elems, dtype=np.int64) - int(start_index)).astype(np.int32)


def build_node_to_elems(n_nodes: int, elem_nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build node-to-element adjacency from local connectivity."""
    counts = np.zeros(n_nodes, dtype=np.int32)
    for elem in elem_nodes:
        seen: set[int] = set()
        for node in elem:
            n = int(node)
            if n < 0 or n in seen:
                continue
            seen.add(n)
            counts[n] += 1

    max_conn = int(counts.max()) if n_nodes else 0
    node_to_elems = np.full((n_nodes, max_conn), -1, dtype=np.int32)
    fill = np.zeros(n_nodes, dtype=np.int32)
    for e, elem in enumerate(elem_nodes):
        seen: set[int] = set()
        for node in elem:
            n = int(node)
            if n < 0 or n in seen:
                continue
            seen.add(n)
            node_to_elems[n, fill[n]] = e
            fill[n] += 1
    return node_to_elems, counts


def _compute_elem_centres(
    lon: np.ndarray,
    lat: np.ndarray,
    elem_nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tri_lon = lon[elem_nodes]
    tri_lat = lat[elem_nodes]
    lon_rad = np.deg2rad(tri_lon)
    lon_c = np.rad2deg(np.arctan2(np.sin(lon_rad).mean(axis=1), np.cos(lon_rad).mean(axis=1)))
    lat_c = tri_lat.mean(axis=1)
    return lon_c, lat_c


def _nod_in_elem2d_is_degenerate(n2e: np.ndarray, n2e_count: np.ndarray) -> bool:
    """Whether a precomputed nod_in_elem2D table must be rebuilt.

    ``n2e`` is oriented ``(n_nodes, max_conn)``, 0-based, ``-1`` padded. Some
    FESOM2 mesh diags (the core2 ``fesom.mesh.diag.nc`` among them) ship it
    unpopulated (all zeros -> all -1 after the 1-based shift) while the count
    array stays valid. Two signals trip a rebuild: *mostly empty* (>50 % of
    nodes have no first incident element) and *inconsistent* (a slot inside a
    node's declared count is padding, or a count exceeds the table width).
    Mirrors kelp.readers.fesom2._nod_in_elem2d_is_degenerate.
    """
    if n2e.shape[0] == 0:
        return False
    max_conn = n2e.shape[1]
    if max_conn == 0:
        return True
    mostly_empty = (n2e[:, 0] < 0).mean() > 0.5
    within_count = np.arange(max_conn)[None, :] < n2e_count[:, None]
    empty_inside_count = bool((within_count & (n2e < 0)).any())
    bad_count = bool((n2e_count < 0).any() or (n2e_count > max_conn).any())
    return bool(mostly_empty or empty_inside_count or bad_count)


def _resolve_mesh_coords_conn(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(lon_deg, lat_deg, elem_nodes_0based) across FESOM2 diag conventions.

    Handles ``.comp``/UGRID (``lon``/``lat`` + ``elements``/``face_nodes``) and
    the older core2 form (packed ``nodes (2,N)`` in RADIANS + ``elem (3,E)``).
    """
    if _first_existing(ds, ("lon",)) and _first_existing(ds, ("lat",)):
        lon = np.asarray(ds["lon"].values, dtype=np.float64)
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
    elif _first_existing(ds, ("nodes",)):
        nodes = np.asarray(ds["nodes"].values, dtype=np.float64)      # (2, N)
        if nodes.shape[0] != 2:
            nodes = nodes.T
        lon, lat = nodes[0].copy(), nodes[1].copy()
        units = str(ds["nodes"].attrs.get("units", "") or "").lower()
        is_radians = units.startswith("radian") or (
            not units.startswith("degree")
            and np.nanmax(np.abs(lon)) <= 2 * np.pi + 1e-6
            and np.nanmax(np.abs(lat)) <= 0.5 * np.pi + 1e-6
        )
        if is_radians:
            lon, lat = np.degrees(lon), np.degrees(lat)
    else:
        raise KeyError("Mesh NetCDF must contain 'lon'/'lat' or packed 'nodes'.")
    conn_name = _first_existing(ds, ("elements", "elem", "face_nodes"))
    if conn_name is None:
        raise KeyError("Mesh NetCDF must contain 'elements', 'elem', or 'face_nodes'.")
    conn_var = ds[conn_name]
    try:
        start_index = int(conn_var.attrs.get("start_index", 1))
    except (TypeError, ValueError):
        start_index = 1
    return lon, lat, _normalize_connectivity(conn_var.values, start_index)


def _resolve_node_to_elems(ds: xr.Dataset, n_nodes: int,
                           elem_nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Node->element table: use the precomputed one, but rebuild from the
    connectivity when it is absent or degenerate (an all-zero nod_in_elem2D
    would otherwise corrupt halo expansion)."""
    n2e_name = _first_existing(ds, ("nod_in_elem2D",))
    cnt_name = _first_existing(ds, ("nod_in_elem2D_num",))
    if n2e_name is None or cnt_name is None:
        return build_node_to_elems(n_nodes, elem_nodes)
    raw = np.asarray(ds[n2e_name].values)
    if raw.ndim == 2 and raw.shape[0] == n_nodes:
        oriented = raw
    elif raw.ndim == 2 and raw.shape[1] == n_nodes:
        oriented = raw.T
    else:
        return build_node_to_elems(n_nodes, elem_nodes)
    counts = np.asarray(ds[cnt_name].values, dtype=np.int32)
    # The count vector must be 1-D of length n_nodes. A malformed one (e.g. a
    # singleton) would otherwise silently broadcast inside the degeneracy check
    # and be returned as an invalid node_to_elems_count — don't trust it,
    # rebuild from the connectivity (the old _normalize_node_to_elems raised on
    # this; rebuilding is the graceful equivalent, consistent with this fn).
    if counts.ndim != 1 or counts.shape[0] != n_nodes:
        return build_node_to_elems(n_nodes, elem_nodes)
    n2e = (np.asarray(oriented, dtype=np.int64) - 1).astype(np.int32)  # 1-based->0, 0->-1
    if _nod_in_elem2d_is_degenerate(n2e, counts):
        return build_node_to_elems(n_nodes, elem_nodes)
    return n2e, counts


def load_mesh_source(path: str) -> MeshSource:
    """Open and normalize a source FESOM2 mesh NetCDF (convention-agnostic)."""
    mesh_path = Path(path).expanduser().resolve()
    ds = xr.open_dataset(mesh_path)

    lon, lat, elem_nodes = _resolve_mesh_coords_conn(ds)
    n_nodes = lon.shape[0]

    depth_name = _first_existing(ds, ("zbar_n_bottom", "depth"))
    depth = (
        np.abs(np.asarray(ds[depth_name].values, dtype=np.float64))
        if depth_name is not None else np.zeros(n_nodes, dtype=np.float64)
    )
    sigma_name = _first_existing(ds, ("nz", "zbar", "sigma"))
    sigma = (
        np.abs(np.asarray(ds[sigma_name].values, dtype=np.float64))
        if sigma_name is not None else np.array([], dtype=np.float64)
    )

    node_to_elems, node_to_elems_count = _resolve_node_to_elems(ds, n_nodes, elem_nodes)

    elem_center_lon, elem_center_lat = _compute_elem_centres(lon, lat, elem_nodes)
    return MeshSource(
        path=mesh_path,
        ds=ds,
        lon=lon,
        lat=lat,
        elem_nodes=elem_nodes,
        node_to_elems=node_to_elems,
        node_to_elems_count=node_to_elems_count,
        depth=depth,
        sigma=sigma,
        elem_center_lon=elem_center_lon,
        elem_center_lat=elem_center_lat,
    )


def _open_var_file(path: Path, role: str, aliases: Sequence[str], required: bool) -> Optional[VarSource]:
    ds = xr.open_dataset(path)
    var_name = _pick_data_var(ds, aliases, required=required)
    if var_name is None:
        ds.close()
        return None
    return VarSource(role=role, path=path, dataset=ds, var_name=var_name, data=ds[var_name])


def load_velocity_sources(cfg: SourceConfig) -> Dict[str, VarSource]:
    """Open input velocity sources."""
    sources: Dict[str, VarSource] = {}

    def add_if(path_str: Optional[str], role: str, aliases: Sequence[str], required: bool) -> None:
        if path_str is None:
            return
        source = _open_var_file(Path(path_str).expanduser().resolve(), role, aliases, required)
        if source is not None:
            sources[role] = source

    if cfg.input:
        input_path = Path(cfg.input).expanduser().resolve()
        if input_path.is_dir():
            files = {
                "u": sorted(input_path.glob("u*.nc")),
                "v": sorted(input_path.glob("v*.nc")),
                "w": sorted(input_path.glob("w*.nc")),
                "ssh": sorted(input_path.glob("ssh*.nc")),
            }
            if not files["u"] or not files["v"]:
                raise FileNotFoundError(
                    f"Directory {input_path} must contain u*.nc and v*.nc files."
                )
            add_if(str(files["u"][0]), "u", U_ALIASES, True)
            add_if(str(files["v"][0]), "v", V_ALIASES, True)
            add_if(str(files["w"][0]), "w", W_ALIASES, False) if files["w"] else None
            add_if(str(files["ssh"][0]), "ssh", SSH_ALIASES, False) if files["ssh"] else None
        else:
            ds = xr.open_dataset(input_path)
            try:
                for role, aliases, required in (
                    ("u", U_ALIASES, True),
                    ("v", V_ALIASES, True),
                    ("w", W_ALIASES, False),
                    ("ssh", SSH_ALIASES, False),
                ):
                    var_name = _pick_data_var(ds, aliases, required=required)
                    if var_name is not None:
                        sources[role] = VarSource(
                            role=role,
                            path=input_path,
                            dataset=ds,
                            var_name=var_name,
                            data=ds[var_name],
                        )
                if "v" not in sources and cfg.v_path:
                    add_if(cfg.v_path, "v", V_ALIASES, True)
                if "w" not in sources and cfg.w_path:
                    add_if(cfg.w_path, "w", W_ALIASES, False)
                if "ssh" not in sources and cfg.ssh_path:
                    add_if(cfg.ssh_path, "ssh", SSH_ALIASES, False)
            except Exception:
                if not any(src.dataset is ds for src in sources.values()):
                    ds.close()
                raise
    else:
        add_if(cfg.u_path, "u", U_ALIASES, True)
        add_if(cfg.v_path, "v", V_ALIASES, True)
        add_if(cfg.w_path, "w", W_ALIASES, False)
        add_if(cfg.ssh_path, "ssh", SSH_ALIASES, False)

    if "u" not in sources or "v" not in sources:
        raise ValueError("Could not open both u and v velocity sources.")
    return sources


def load_extra_var_sources(specs: Sequence[ExtraVarSpec]) -> Dict[str, VarSource]:
    """Open extra variable sources (temp, salt, or any arbitrary variable)."""
    sources: Dict[str, VarSource] = {}
    for spec in specs:
        path = Path(spec.path).expanduser().resolve()
        ds = xr.open_dataset(path)
        if spec.var:
            # Explicit variable name
            if spec.var not in ds.data_vars:
                raise KeyError(f"Variable {spec.var!r} not found in {path}; available: {list(ds.data_vars)}")
            var_name = spec.var
        elif spec.aliases:
            var_name = _pick_data_var(ds, spec.aliases, required=True)
        elif len(ds.data_vars) == 1:
            var_name = next(iter(ds.data_vars))
        else:
            raise ValueError(
                f"Cannot auto-detect variable in {path} with {len(ds.data_vars)} "
                f"variables ({list(ds.data_vars)}); specify 'var' or 'aliases'."
            )
        role = spec.role or var_name
        sources[role] = VarSource(
            role=role, path=path, dataset=ds, var_name=var_name, data=ds[var_name],
        )
    return sources


def close_velocity_sources(sources: Mapping[str, VarSource]) -> None:
    """Close all unique xarray datasets behind velocity sources."""
    seen: set[int] = set()
    for src in sources.values():
        ds_id = id(src.dataset)
        if ds_id in seen:
            continue
        src.dataset.close()
        seen.add(ds_id)


def _region_bbox(spec: RegionSpec, polygon_lon: Optional[np.ndarray], polygon_lat: Optional[np.ndarray]) -> tuple[float, float, float, float]:
    if spec.mode == "bbox":
        assert spec.bbox is not None
        return spec.bbox
    if spec.mode == "center_radius":
        assert spec.center is not None and spec.radius_km is not None
        lon0, lat0 = spec.center
        # The spherical cap of angular radius theta: r / (111 cos lat0) was
        # too narrow in longitude and missed a cap over a pole.
        theta = math.degrees(spec.radius_km / EARTH_RADIUS_KM)
        min_lat, max_lat = max(lat0 - theta, -90.0), min(lat0 + theta, 90.0)
        if lat0 + theta >= 90.0 or lat0 - theta <= -90.0:
            return (lon0 - 180.0, lon0 + 180.0, min_lat, max_lat)
        ratio = math.sin(math.radians(theta)) / math.cos(math.radians(lat0))
        dlon = math.degrees(math.asin(min(ratio, 1.0)))
        return (lon0 - dlon, lon0 + dlon, min_lat, max_lat)
    if polygon_lon is None or polygon_lat is None:
        raise ValueError("polygon bbox requested without polygon coordinates.")
    return (
        float(polygon_lon.min()),
        float(polygon_lon.max()),
        float(polygon_lat.min()),
        float(polygon_lat.max()),
    )


def _contains_bbox(lon: np.ndarray, lat: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    min_lon, max_lon, min_lat, max_lat = bbox
    return (lon >= min_lon) & (lon <= max_lon) & (lat >= min_lat) & (lat <= max_lat)


def _haversine_km(lon1: np.ndarray, lat1: np.ndarray, lon2: float, lat2: float) -> np.ndarray:
    lon1r = np.deg2rad(lon1)
    lat1r = np.deg2rad(lat1)
    lon2r = math.radians(lon2)
    lat2r = math.radians(lat2)
    dlon = lon1r - lon2r
    dlat = lat1r - lat2r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * math.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return EARTH_RADIUS_KM * c


def _contains_circle(
    lon: np.ndarray,
    lat: np.ndarray,
    center: tuple[float, float],
    radius_km: float,
) -> np.ndarray:
    lon0, lat0 = center
    return _haversine_km(lon, lat, lon0, lat0) <= radius_km


def _points_in_polygon(lon: np.ndarray, lat: np.ndarray, poly_lon: np.ndarray, poly_lat: np.ndarray) -> np.ndarray:
    inside = np.zeros(lon.shape[0], dtype=bool)
    x1 = poly_lon[-1]
    y1 = poly_lat[-1]
    for x2, y2 in zip(poly_lon, poly_lat):
        crosses = ((y2 > lat) != (y1 > lat))
        denom = (y1 - y2) + 1e-300
        x_intersect = (x1 - x2) * (lat - y2) / denom + x2
        inside ^= crosses & (lon < x_intersect)
        x1, y1 = x2, y2
    return inside


def _sign(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    return (px - bx) * (ay - by) - (ax - bx) * (py - by)


def _point_in_triangle(px: float, py: float, tri_lon: np.ndarray, tri_lat: np.ndarray) -> bool:
    d1 = _sign(px, py, tri_lon[0], tri_lat[0], tri_lon[1], tri_lat[1])
    d2 = _sign(px, py, tri_lon[1], tri_lat[1], tri_lon[2], tri_lat[2])
    d3 = _sign(px, py, tri_lon[2], tri_lat[2], tri_lon[0], tri_lat[0])
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def _orientation(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> int:
    val = (by - ay) * (cx - bx) - (bx - ax) * (cy - by)
    if abs(val) < 1e-15:
        return 0
    return 1 if val > 0 else 2


def _on_segment(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> bool:
    return min(ax, cx) - 1e-15 <= bx <= max(ax, cx) + 1e-15 and min(ay, cy) - 1e-15 <= by <= max(ay, cy) + 1e-15


def _segments_intersect(
    ax: float, ay: float, bx: float, by: float,
    cx: float, cy: float, dx: float, dy: float,
) -> bool:
    o1 = _orientation(ax, ay, bx, by, cx, cy)
    o2 = _orientation(ax, ay, bx, by, dx, dy)
    o3 = _orientation(cx, cy, dx, dy, ax, ay)
    o4 = _orientation(cx, cy, dx, dy, bx, by)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(ax, ay, cx, cy, bx, by):
        return True
    if o2 == 0 and _on_segment(ax, ay, dx, dy, bx, by):
        return True
    if o3 == 0 and _on_segment(cx, cy, ax, ay, dx, dy):
        return True
    if o4 == 0 and _on_segment(cx, cy, bx, by, dx, dy):
        return True
    return False


def _load_polygon(path: str, polygon_format: Optional[str]) -> tuple[np.ndarray, np.ndarray]:
    p = Path(path).expanduser().resolve()
    fmt = (polygon_format or p.suffix.lstrip(".") or "geojson").lower()

    if fmt in ("geojson", "json"):
        raw = json.loads(p.read_text())
        geom = raw
        if raw.get("type") == "Feature":
            geom = raw["geometry"]
        elif raw.get("type") == "FeatureCollection":
            features = raw.get("features", [])
            if not features:
                raise ValueError(f"No features found in {p}")
            geom = features[0]["geometry"]
        if geom.get("type") != "Polygon":
            raise ValueError(f"Only GeoJSON Polygon is supported in v1, got {geom.get('type')}")
        coords = geom["coordinates"][0]
    elif fmt in ("yaml", "yml"):
        raw = yaml.safe_load(p.read_text())
        if isinstance(raw, Mapping) and "vertices" in raw:
            coords = raw["vertices"]
        else:
            coords = raw
    elif fmt == "ascii":
        coords = []
        for line in p.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid polygon line in {p}: {line!r}")
            coords.append((float(parts[0]), float(parts[1])))
    else:
        raise ValueError(f"Unsupported polygon format: {fmt}")

    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Polygon coordinates must have shape (N, 2), got {arr.shape}")
    if arr.shape[0] < 3:
        raise ValueError("Polygon must have at least 3 vertices.")
    if np.allclose(arr[0], arr[-1]):
        arr = arr[:-1]
    return arr[:, 0], arr[:, 1]


def _candidate_mask_from_bbox(
    elem_lon_min: np.ndarray,
    elem_lon_max: np.ndarray,
    elem_lat_min: np.ndarray,
    elem_lat_max: np.ndarray,
    region_bbox: tuple[float, float, float, float],
) -> np.ndarray:
    min_lon, max_lon, min_lat, max_lat = region_bbox
    return (
        (elem_lon_max >= min_lon) &
        (elem_lon_min <= max_lon) &
        (elem_lat_max >= min_lat) &
        (elem_lat_min <= max_lat)
    )


def _unwrap_near(lon: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """``lon`` shifted by whole turns to lie within 180 degrees of ``ref``.

    A longitude already within 180 degrees comes back bit for bit, so a
    vertex on a region edge keeps its verdict.
    """
    return lon + 360.0 * np.round((ref - lon) / 360.0)


def select_region(mesh: MeshSource, spec: RegionSpec) -> np.ndarray:
    """Return the selected element mask on the source mesh.

    An element is selected when it intersects the closed region: one of its
    vertices or its centre lies in the region, or (bbox and polygon, in the
    lon/lat plane) a region vertex lies in the triangle or a region edge
    crosses a triangle edge, or (center_radius, on the sphere) the circle
    centre lies in the triangle or a triangle edge passes within the radius.
    Each triangle's longitudes are unwrapped relative to its first vertex,
    so a triangle across the +-180 seam stays a narrow triangle; it is
    tested at its own longitudes and shifted by one turn either way, so it
    meets a region given on either side of the seam.  ``halo_layers``
    (applied by the caller) adds element rings around the selection.
    """
    polygon_lon = polygon_lat = None
    if spec.mode == "polygon":
        polygon_lon, polygon_lat = _load_polygon(spec.polygon_path or "", spec.polygon_format)

    tri_lat = mesh.lat[mesh.elem_nodes]
    # Seam-aware: raw longitudes would give a seam triangle a 360-degree box.
    tri_lon = mesh.lon[mesh.elem_nodes]
    tri_lon = _unwrap_near(tri_lon, tri_lon[:, :1])
    elem_lon_min = tri_lon.min(axis=1)
    elem_lon_max = tri_lon.max(axis=1)
    elem_lat_min = tri_lat.min(axis=1)
    elem_lat_max = tri_lat.max(axis=1)
    region_bbox = _region_bbox(spec, polygon_lon, polygon_lat)
    mask = np.zeros(mesh.n_elements, dtype=bool)
    for turn in (0.0, -360.0, 360.0):
        candidate_mask = _candidate_mask_from_bbox(
            elem_lon_min + turn,
            elem_lon_max + turn,
            elem_lat_min,
            elem_lat_max,
            region_bbox,
        ) & ~mask
        candidates = np.flatnonzero(candidate_mask)
        if candidates.size == 0:
            continue
        cand_lon = tri_lon[candidates] + turn
        selected = _select_candidates(
            spec, cand_lon, tri_lat[candidates],
            # the (seam-aware) centre, in the same turn as its triangle
            _unwrap_near(mesh.elem_center_lon[candidates], cand_lon[:, 0]),
            mesh.elem_center_lat[candidates],
            polygon_lon, polygon_lat,
        )
        mask[candidates[selected]] = True
    return mask


def _select_candidates(
    spec: RegionSpec,
    tri_lon: np.ndarray,
    tri_lat: np.ndarray,
    center_lon: np.ndarray,
    center_lat: np.ndarray,
    polygon_lon: Optional[np.ndarray],
    polygon_lat: Optional[np.ndarray],
) -> np.ndarray:
    """Which candidate elements meet the region (see :func:`select_region`);
    ``tri_lon``/``tri_lat`` hold the candidates' vertices, shape ``(n, 3)``."""
    n_candidates = tri_lon.shape[0]
    if spec.mode == "bbox":
        assert spec.bbox is not None
        center_in = _contains_bbox(center_lon, center_lat, spec.bbox)
        vertex_in = _contains_bbox(
            tri_lon.reshape(-1),
            tri_lat.reshape(-1),
            spec.bbox,
        ).reshape(n_candidates, 3).any(axis=1)
        selected = center_in | vertex_in
        # a box inside one triangle, or cutting a triangle's corner, holds
        # none of its vertices: test the box as a polygon there
        min_lon, max_lon, min_lat, max_lat = spec.bbox
        rest = np.flatnonzero(~selected)
        selected[rest] = _polygon_reaches_triangles(
            tri_lon[rest], tri_lat[rest],
            np.array([min_lon, max_lon, max_lon, min_lon]),
            np.array([min_lat, min_lat, max_lat, max_lat]),
        )
    elif spec.mode == "center_radius":
        assert spec.center is not None and spec.radius_km is not None
        center_in = _contains_circle(center_lon, center_lat, spec.center, spec.radius_km)
        vertex_in = _contains_circle(
            tri_lon.reshape(-1),
            tri_lat.reshape(-1),
            spec.center,
            spec.radius_km,
        ).reshape(n_candidates, 3).any(axis=1)
        selected = center_in | vertex_in
        rest = np.flatnonzero(~selected)
        selected[rest] = _circle_reaches_triangles(
            tri_lon[rest], tri_lat[rest], spec.center, spec.radius_km)
    else:
        assert polygon_lon is not None and polygon_lat is not None
        center_in = _points_in_polygon(center_lon, center_lat, polygon_lon, polygon_lat)
        vertex_in = _points_in_polygon(
            tri_lon.reshape(-1),
            tri_lat.reshape(-1),
            polygon_lon,
            polygon_lat,
        ).reshape(n_candidates, 3).any(axis=1)
        selected = center_in | vertex_in
        rest = np.flatnonzero(~selected)
        selected[rest] = _polygon_reaches_triangles(
            tri_lon[rest], tri_lat[rest], polygon_lon, polygon_lat)
    return selected


def _polygon_reaches_triangles(
    tri_lon: np.ndarray,
    tri_lat: np.ndarray,
    polygon_lon: np.ndarray,
    polygon_lat: np.ndarray,
) -> np.ndarray:
    """Per triangle (vertices ``(n, 3)``): a polygon vertex lies in the
    triangle, or a polygon edge crosses or touches a triangle edge (the
    lon/lat plane)."""
    n_candidates = tri_lon.shape[0]
    reached = np.zeros(n_candidates, dtype=bool)
    poly_points = list(zip(polygon_lon, polygon_lat))
    poly_edges = list(zip(poly_points, poly_points[1:] + poly_points[:1]))
    for out_i in range(n_candidates):
        tri_x = tri_lon[out_i]
        tri_y = tri_lat[out_i]
        if any(_point_in_triangle(px, py, tri_x, tri_y) for px, py in poly_points):
            reached[out_i] = True
            continue
        tri_edges = [
            ((tri_x[0], tri_y[0]), (tri_x[1], tri_y[1])),
            ((tri_x[1], tri_y[1]), (tri_x[2], tri_y[2])),
            ((tri_x[2], tri_y[2]), (tri_x[0], tri_y[0])),
        ]
        reached[out_i] = any(
            _segments_intersect(ax, ay, bx, by, cx, cy, dx, dy)
            for (ax, ay), (bx, by) in tri_edges
            for (cx, cy), (dx, dy) in poly_edges
        )
    return reached


def _circle_reaches_triangles(
    tri_lon: np.ndarray,
    tri_lat: np.ndarray,
    center: tuple[float, float],
    radius_km: float,
) -> np.ndarray:
    """Per triangle (vertices ``(n, 3)``, great-circle edges): the circle
    centre lies in the triangle, or an edge passes within ``radius_km``.

    In the gnomonic projection centred on the circle centre great circles
    are straight lines and a point at angular distance theta lies at radius
    tan(theta), so comparing the distance from the origin to a projected
    edge with tan(r / R) is exact on the sphere.  A triangle with a vertex
    in the far hemisphere (cos theta <= 0) has no projection; for r below
    90 degrees a mesh triangle that reaches the circle has none there.
    """
    theta_r = radius_km / EARTH_RADIUS_KM
    reached = np.zeros(tri_lon.shape[0], dtype=bool)
    if theta_r >= 0.5 * math.pi or tri_lon.shape[0] == 0:
        # the outside of such a circle is a cap below 90 degrees, which
        # holds every geodesic triangle whose vertices it holds: the vertex
        # test alone is exact
        return reached
    lon0, lat0 = math.radians(center[0]), math.radians(center[1])
    dlam = np.deg2rad(tri_lon) - lon0
    phi = np.deg2rad(tri_lat)
    cos_c = (math.sin(lat0) * np.sin(phi)
             + math.cos(lat0) * np.cos(phi) * np.cos(dlam))
    near = (cos_c > 0.0).all(axis=1)
    cos_c = np.where(cos_c > 0.0, cos_c, 1.0)
    ax = np.cos(phi) * np.sin(dlam) / cos_c
    ay = (math.cos(lat0) * np.sin(phi)
          - math.sin(lat0) * np.cos(phi) * np.cos(dlam)) / cos_c
    bx, by = np.roll(ax, -1, axis=1), np.roll(ay, -1, axis=1)   # edges a -> b
    ex, ey = bx - ax, by - ay
    len2 = ex * ex + ey * ey
    t = np.clip(-(ax * ex + ay * ey) / np.where(len2 > 0.0, len2, 1.0), 0.0, 1.0)
    edge_in = (np.hypot(ax + t * ex, ay + t * ey) <= math.tan(theta_r)).any(axis=1)
    # the origin is inside (or on) the triangle when a x b has one sign on
    # all three edges
    cross = ax * by - ay * bx
    centre_in = (cross >= 0.0).all(axis=1) | (cross <= 0.0).all(axis=1)
    reached[:] = near & (edge_in | centre_in)
    return reached


def expand_halo(mesh: MeshSource, selected_mask: np.ndarray, halo_layers: int) -> np.ndarray:
    """Expand selection by neighboring element layers."""
    mask = np.asarray(selected_mask, dtype=bool).copy()
    if halo_layers <= 0 or not mask.any():
        return mask

    for _ in range(halo_layers):
        selected_elems = np.flatnonzero(mask)
        selected_nodes = np.unique(mesh.elem_nodes[selected_elems].reshape(-1))
        selected_nodes = selected_nodes[selected_nodes >= 0]
        adjacent = mesh.node_to_elems[selected_nodes]
        adjacent = adjacent[adjacent >= 0]
        if adjacent.size == 0:
            break
        mask[np.unique(adjacent)] = True
    return mask


def build_subset_selection(mesh: MeshSource, selected_mask: np.ndarray) -> SubsetSelection:
    """Build source->subset index mapping."""
    elem_ids = np.flatnonzero(selected_mask).astype(np.int32)
    if elem_ids.size == 0:
        raise ValueError("Region selection produced zero elements.")
    node_ids = np.unique(mesh.elem_nodes[elem_ids].reshape(-1))
    node_ids = node_ids[node_ids >= 0].astype(np.int32)
    if node_ids.size == 0:
        raise ValueError("Region selection produced zero nodes.")

    elem_global_to_local = np.full(mesh.n_elements, -1, dtype=np.int32)
    node_global_to_local = np.full(mesh.n_nodes, -1, dtype=np.int32)
    elem_global_to_local[elem_ids] = np.arange(elem_ids.size, dtype=np.int32)
    node_global_to_local[node_ids] = np.arange(node_ids.size, dtype=np.int32)
    return SubsetSelection(
        elem_ids=elem_ids,
        node_ids=node_ids,
        elem_global_to_local=elem_global_to_local,
        node_global_to_local=node_global_to_local,
    )


def _netcdf_safe_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in attrs.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, np.integer, np.floating)):
            safe[str(key)] = value.item() if isinstance(value, (np.integer, np.floating)) else value
        elif isinstance(value, (list, tuple, dict, np.ndarray, bool)):
            safe[str(key)] = json.dumps(value if not isinstance(value, np.ndarray) else value.tolist())
        else:
            safe[str(key)] = str(value)
    return safe


def build_provenance_attrs(
    cfg: SubsetConfig,
    config_path: Optional[Path],
    config_text: str,
    config_digest: str,
    mesh: MeshSource,
    selection: SubsetSelection,
    time_indices: Optional[np.ndarray],
) -> dict[str, Any]:
    """Build attrs shared across outputs."""
    region = cfg.region
    time_start = int(time_indices[0]) if time_indices is not None and time_indices.size else None
    time_stop = int(time_indices[-1] + cfg.time.step) if time_indices is not None and time_indices.size else None
    return _netcdf_safe_attrs({
        "subset_tool": "utils.extract_fesom_region",
        "subset_tool_version": "v1",
        "subset_created_utc": datetime.now(timezone.utc).isoformat(),
        "subset_created_by": cfg.metadata.created_by,
        "subset_name": cfg.metadata.subset_name,
        "subset_description": cfg.metadata.description,
        "subset_notes": cfg.metadata.notes,
        "subset_source_mesh": str(mesh.path),
        "subset_source_input": cfg.source.input,
        "subset_source_u_path": cfg.source.u_path,
        "subset_source_v_path": cfg.source.v_path,
        "subset_source_w_path": cfg.source.w_path,
        "subset_source_ssh_path": cfg.source.ssh_path,
        "subset_region_mode": region.mode,
        "subset_region_bbox": list(region.bbox) if region.bbox is not None else None,
        "subset_region_center": list(region.center) if region.center is not None else None,
        "subset_region_radius_km": region.radius_km,
        "subset_region_polygon_path": region.polygon_path,
        "subset_region_polygon_format": region.polygon_format,
        "subset_halo_layers": region.halo_layers,
        "subset_time_start_idx": cfg.time.start,
        "subset_time_stop_idx": cfg.time.stop,
        "subset_time_step": cfg.time.step,
        "subset_time_selected_start": time_start,
        "subset_time_selected_stop_exclusive": time_stop,
        "subset_output_format": cfg.output.format,
        "subset_output_layout": cfg.output.layout,
        "subset_source_n_nodes": mesh.n_nodes,
        "subset_source_n_elements": mesh.n_elements,
        "subset_output_n_nodes": int(selection.node_ids.size),
        "subset_output_n_elements": int(selection.elem_ids.size),
        "subset_config_path": str(config_path) if config_path else None,
        "subset_config_sha256": config_digest,
        "subset_config_text": config_text if cfg.output.embed_config_text else None,
    })


def build_subset_mesh_dataset(
    mesh: MeshSource,
    selection: SubsetSelection,
    attrs: Mapping[str, Any],
    copy_source_attrs: bool,
) -> xr.Dataset:
    """Build subset mesh dataset in FESOM2-compatible NetCDF shape."""
    node_ids = selection.node_ids
    elem_ids = selection.elem_ids
    subset_lon = mesh.lon[node_ids]
    subset_lat = mesh.lat[node_ids]
    subset_depth = mesh.depth[node_ids]
    subset_elems = selection.node_global_to_local[mesh.elem_nodes[elem_ids]]
    subset_n2e, subset_n2e_count = build_node_to_elems(node_ids.size, subset_elems)
    subset_lon_c, subset_lat_c = _compute_elem_centres(subset_lon, subset_lat, subset_elems)

    max_conn = subset_n2e.shape[1] if subset_n2e.ndim == 2 else 0
    ds = xr.Dataset(
        data_vars={
            "lon": (("nod2",), subset_lon),
            "lat": (("nod2",), subset_lat),
            "elements": (("n3", "elem"), (subset_elems.T + 1).astype(np.int32)),
            "nod_in_elem2D": (("N", "nod2"), (subset_n2e.T + 1).astype(np.int32)),
            "nod_in_elem2D_num": (("nod2",), subset_n2e_count.astype(np.int32)),
            "zbar_n_bottom": (("nod2",), subset_depth),
            "global_node_id": (("nod2",), node_ids.astype(np.int32)),
            "global_elem_id": (("elem",), elem_ids.astype(np.int32)),
            "elem_center_lon": (("elem",), subset_lon_c),
            "elem_center_lat": (("elem",), subset_lat_c),
        },
        coords={
            "nod2": np.arange(node_ids.size, dtype=np.int32),
            "elem": np.arange(elem_ids.size, dtype=np.int32),
            "n3": np.arange(3, dtype=np.int32),
            "N": np.arange(max_conn, dtype=np.int32),
        },
    )
    if mesh.sigma.size:
        ds["nz"] = (("nz",), mesh.sigma.copy())
    # The element levels and bed (the FESOM2 reader declares each element's
    # bottom from them) and the node level counts, values and signs as in
    # the source.
    for name, dim, ids, n_source in (
            ("nlevels", "elem", elem_ids, mesh.n_elements),
            ("ulevels", "elem", elem_ids, mesh.n_elements),
            ("zbar_e_bottom", "elem", elem_ids, mesh.n_elements),
            ("nlevels_nod2D", "nod2", node_ids, mesh.n_nodes)):
        if name not in mesh.ds.variables:
            continue
        source = mesh.ds[name]
        if source.shape != (n_source,):
            raise ValueError(
                f"Mesh {mesh.path}: variable {name!r} has shape "
                f"{source.shape}, expected ({n_source},).")
        ds[name] = ((dim,), np.asarray(source.values)[ids], dict(source.attrs))
    # The model's element areas: the FESOM2 reader prefers them to its own
    # spherical areas, so the subset must carry them to match the source.
    # Accepted as the reader accepts them (squeezed, one value per element).
    if "elem_area" in mesh.ds.variables:
        area = mesh.ds["elem_area"].squeeze()
        if area.ndim == 1 and area.shape[0] == mesh.n_elements:
            ds["elem_area"] = (("elem",), np.asarray(area.values)[elem_ids],
                               dict(area.attrs))

    source_attrs = dict(mesh.ds.attrs) if copy_source_attrs else {}
    source_attrs.update(attrs)
    ds.attrs = _netcdf_safe_attrs(source_attrs)
    return ds


def _detect_time_dim(da: xr.DataArray) -> Optional[str]:
    if "time" in da.dims:
        return "time"
    return None



def resolve_time_indices(primary: xr.DataArray, cfg: TimeConfig) -> Optional[np.ndarray]:
    """Resolve selected time indices from a primary variable."""
    time_dim = _detect_time_dim(primary)
    if time_dim is None:
        if any(v is not None for v in (cfg.start, cfg.stop)) or cfg.step != 1:
            raise ValueError("Time slicing requested, but source variable has no time dimension.")
        return None
    n_time = primary.sizes[time_dim]
    start = cfg.start if cfg.start is not None else 0
    stop = cfg.stop if cfg.stop is not None else n_time
    if start < 0 or stop < 0 or start > n_time or stop > n_time or start > stop:
        raise ValueError(f"Invalid time window start={start}, stop={stop}, n_time={n_time}")
    return np.arange(start, stop, cfg.step, dtype=np.int32)


@dataclass
class _LazySubset:
    """Deferred spatial subset — holds source array + index info for streaming."""

    role: str
    source_da: xr.DataArray          # time-sliced only (lazy)
    spatial_dim: str
    spatial_idx: np.ndarray
    time_dim: Optional[str]
    out_attrs: dict
    out_name: str
    global_attrs: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple:
        """Shape after spatial subsetting (without materialising)."""
        dims = list(self.source_da.dims)
        sizes = dict(self.source_da.sizes)
        sizes[self.spatial_dim] = len(self.spatial_idx)
        return tuple(sizes[d] for d in dims)


def _detect_spatial_dim_auto(
    da: xr.DataArray,
    mesh: MeshSource,
    exclude: Iterable[str] = (),
) -> Tuple[str, np.ndarray, str]:
    """Auto-detect spatial dimension and return (dim_name, subset_idx, basis).

    Tries element-based first (u/v style), then node-based (w/ssh/temp/salt).
    Returns ``basis`` as ``"elem"`` or ``"node"`` for diagnostics.
    """
    excluded = set(exclude)
    candidates = [(d, da.sizes[d]) for d in da.dims if d not in excluded]

    # Try elements first
    for d, sz in candidates:
        if sz == mesh.n_elements:
            return d, mesh.n_elements, "elem"
    # Then nodes
    for d, sz in candidates:
        if sz == mesh.n_nodes:
            return d, mesh.n_nodes, "node"

    raise ValueError(
        f"Could not match any spatial dimension in variable {da.name!r} "
        f"(dims={da.dims}) to mesh nodes ({mesh.n_nodes:,}) or "
        f"elements ({mesh.n_elements:,})."
    )


def subset_var_array(
    src: VarSource,
    mesh: MeshSource,
    selection: SubsetSelection,
    time_indices: Optional[np.ndarray],
) -> _LazySubset:
    """Prepare a deferred spatial subset (no data read yet).

    Auto-detects whether the variable lives on elements or nodes by
    matching dimension sizes against the mesh.
    """
    da = src.data
    time_dim = _detect_time_dim(da)
    if time_dim is not None and time_indices is not None:
        da = da.isel({time_dim: time_indices})

    spatial_dim, _, basis = _detect_spatial_dim_auto(
        da, mesh, exclude=(time_dim,) if time_dim else ()
    )
    spatial_idx = selection.elem_ids if basis == "elem" else selection.node_ids

    attrs = dict(da.attrs)
    attrs["subset_source_variable"] = src.var_name
    attrs["subset_source_path"] = str(src.path)

    return _LazySubset(
        role=src.role,
        source_da=da,
        spatial_dim=spatial_dim,
        spatial_idx=spatial_idx,
        time_dim=time_dim,
        out_attrs=attrs,
        out_name=src.role,
    )


def _zarr_encoding(
    ds: xr.Dataset,
    chunk_overrides: Optional[Dict[str, Any]] = None,
) -> dict[str, dict[str, Any]]:
    """Build per-variable zarr encoding with chunk sizes.

    Default strategy (optimised for kelp's whole-timestep reads):
      - time: 1  (one timestep per chunk — each read is one chunk)
      - depth dims (nz, nz1): full  (always read all levels)
      - spatial dims: full  (always read all nodes/elements)

    For very large subsets the spatial "full" chunk may be too big for
    zarr's compressor buffer.  Override via ``chunk_overrides`` dict
    keyed by dimension name, e.g. ``{"spatial": 100000}`` or exact
    dim names ``{"nod2": 50000, "elem": 80000}``.

    The special key ``"spatial"`` applies to any dim that is not time
    and not a depth dim.
    """
    overrides = chunk_overrides or {}
    encoding: dict[str, dict[str, Any]] = {}
    depth_dims = {"nz", "nz1", "n3", "N", "depth", "level"}

    for name, da in ds.data_vars.items():
        chunks = []
        for dim in da.dims:
            size = ds.sizes[dim]
            if dim in overrides:
                # Exact dim name override
                chunks.append(min(int(overrides[dim]), size))
            elif dim == "time":
                chunks.append(min(int(overrides.get("time", 1)), size))
            elif dim in depth_dims:
                chunks.append(size)  # always full depth
            elif "spatial" in overrides:
                chunks.append(min(int(overrides["spatial"]), size))
            else:
                chunks.append(size)  # default: full spatial
        encoding[name] = {"chunks": tuple(chunks)}
    return encoding


def _sanitize_dataset_attrs(ds: xr.Dataset) -> xr.Dataset:
    """Return a shallow copy with NetCDF/Zarr-safe attrs."""
    clean = ds.copy(deep=False)
    clean.attrs = _netcdf_safe_attrs(clean.attrs)
    for name in clean.data_vars:
        clean[name].attrs = _netcdf_safe_attrs(clean[name].attrs)
    for name in clean.coords:
        clean[name].attrs = _netcdf_safe_attrs(clean[name].attrs)
    return clean


def _write_dataset(
    ds: xr.Dataset,
    path: Path,
    fmt: str,
    chunk_overrides: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a plain xr.Dataset to disk in one shot (used for mesh)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = _sanitize_dataset_attrs(ds)
    if fmt == "netcdf":
        ds.to_netcdf(path)
    elif fmt == "zarr":
        if path.exists():
            shutil.rmtree(path)
        ds.to_zarr(path, mode="w", encoding=_zarr_encoding(ds, chunk_overrides), consolidated=False)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def _time_encoding(step_ds: xr.Dataset, tdim: Optional[str],
                   src_time: Optional[xr.DataArray] = None) -> Dict[str, dict]:
    """``{tdim: {units: "seconds since <first>", dtype: float64, calendar}}``
    for a datetime/cftime time coordinate, else ``{}``.  Fixed at the FIRST
    write: from one value xarray would pick e.g. integer days, and later
    sub-daily steps appended into that axis would be truncated (NetCDF) or
    re-encoded in the wrong unit (Zarr: 6 h became 6 d)."""
    if not tdim or tdim not in step_ds.coords:
        return {}
    tc = step_ds[tdim]
    if not (np.issubdtype(tc.dtype, np.datetime64) or tc.dtype == object):
        return {}
    first = tc.values[0]
    if np.issubdtype(tc.dtype, np.datetime64):
        t0 = np.datetime_as_string(first, unit="s").replace("T", " ")
        cal = None
    else:
        t0 = (f"{first.year:04d}-{first.month:02d}-{first.day:02d} "
              f"{first.hour:02d}:{first.minute:02d}:{first.second:02d}")
        cal = getattr(first, "calendar", None)
    src_enc = dict(getattr(src_time, "encoding", {}) or {}) if src_time is not None else {}
    enc = {"units": f"seconds since {t0}", "dtype": "float64"}
    cal = src_enc.get("calendar") or cal
    if cal:
        enc["calendar"] = cal
    return {tdim: enc}


def _append_time_step_netcdf(path: Path, step_ds: xr.Dataset, tdim: str) -> None:
    """Append one time step (``step_ds``, length 1 along ``tdim``) to the
    NetCDF file ``path`` along its unlimited ``tdim`` dimension.  Every
    variable carrying ``tdim`` is written at the next index; a datetime
    coordinate is encoded with the file's own ``units``/``calendar``."""
    import netCDF4 as nc4
    with nc4.Dataset(path, "a") as nc:
        k = len(nc.dimensions[tdim])
        for name in list(step_ds.coords) + list(step_ds.data_vars):
            da = step_ds[name]
            if tdim not in da.dims or name not in nc.variables:
                continue
            var = nc.variables[name]
            vals = da.values
            if np.issubdtype(vals.dtype, np.datetime64) or vals.dtype == object:
                units = var.getncattr("units")
                cal = var.getncattr("calendar") if "calendar" in var.ncattrs() else "standard"
                vals, _, _ = xr.coding.times.encode_cf_datetime(vals, units, cal)
                vals = np.asarray(vals, dtype=var.dtype)
            axis = var.dimensions.index(tdim)
            idx = [slice(None)] * var.ndim
            idx[axis] = slice(k, k + 1)
            order = [da.dims.index(d) for d in var.dimensions]
            arr = np.transpose(np.asarray(vals), order)
            if (np.issubdtype(arr.dtype, np.floating)
                    and ("_FillValue" in var.ncattrs() or var.dtype.kind in "iu")):
                # a decoded NaN must become the variable's fill, as xarray
                # wrote it at the first step — cast to a packed int it read
                # back as a finite value (round 2 review)
                arr = np.ma.masked_invalid(arr)
            var[tuple(idx)] = arr


def _write_lazy_subset(
    lazy: _LazySubset,
    path: Path,
    fmt: str,
    global_attrs: dict,
    progress: bool = True,
    chunk_overrides: Optional[Dict[str, Any]] = None,
) -> None:
    """Stream a _LazySubset to disk one timestep at a time.

    For each timestep:
      1. Read the **full** time slice from source (one contiguous HDF5
         hyperslab — fast).
      2. Subset spatially in memory with numpy fancy indexing (instant).
      3. Write the small result to disk.

    This avoids the pathologically slow non-contiguous fancy-index reads
    that NetCDF4/HDF5 produces when xarray applies spatial isel lazily.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    src = lazy.source_da
    sdim = lazy.spatial_dim
    sidx = lazy.spatial_idx
    tdim = lazy.time_dim

    has_time = tdim is not None and src.sizes.get(tdim, 0) > 1

    if not has_time:
        # No time dim or single step — read full, subset, write.
        full = src.compute()
        subset = full.isel({sdim: sidx})
        subset = subset.rename(lazy.out_name)
        subset.attrs = lazy.out_attrs
        ds = subset.to_dataset(name=lazy.out_name)
        ds.attrs = _netcdf_safe_attrs(global_attrs)
        ds = _sanitize_dataset_attrs(ds)
        if fmt == "netcdf":
            ds.to_netcdf(path)
        elif fmt == "zarr":
            if path.exists():
                shutil.rmtree(path)
            ds.to_zarr(path, mode="w", encoding=_zarr_encoding(ds, chunk_overrides), consolidated=False)
        return

    n_times = src.sizes[tdim]
    zarr_enc = None  # computed once from first chunk
    for i in range(n_times):
        # 1) Read full time slice (contiguous hyperslab — one HDF5 read).
        full_slice = src.isel({tdim: [i]}).compute()
        # 2) Subset spatially in memory (numpy indexing — instant).
        chunk_da = full_slice.isel({sdim: sidx})
        chunk_da = chunk_da.rename(lazy.out_name)
        if i == 0:
            chunk_da.attrs = lazy.out_attrs
        chunk_ds = chunk_da.to_dataset(name=lazy.out_name)
        if i == 0:
            chunk_ds.attrs = _netcdf_safe_attrs(global_attrs)

        if fmt == "netcdf":
            if i == 0:
                chunk_ds = _sanitize_dataset_attrs(chunk_ds)
                # fix the time encoding now: from ONE value xarray would pick
                # e.g. integer days, and later sub-daily steps appended into
                # that axis would be truncated
                enc = _time_encoding(chunk_ds, tdim, src[tdim])
                chunk_ds.to_netcdf(path, unlimited_dims=[tdim], encoding=enc)
            else:
                # xarray's mode="a" OVERWRITES a variable of the same name:
                # the file kept only the last step (2026-10-13).  Extend the
                # unlimited dimension with netCDF4 instead.
                _append_time_step_netcdf(path, chunk_ds, tdim)
        elif fmt == "zarr":
            if i == 0:
                if path.exists():
                    shutil.rmtree(path)
                chunk_ds = _sanitize_dataset_attrs(chunk_ds)
                zarr_enc = _zarr_encoding(chunk_ds, chunk_overrides)
                for name, e in _time_encoding(chunk_ds, tdim, src[tdim]).items():
                    zarr_enc.setdefault(name, {}).update(e)
                chunk_ds.to_zarr(path, mode="w", encoding=zarr_enc)
            else:
                chunk_ds.to_zarr(path, mode="a", append_dim=tdim)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        if progress:
            print(f"\r         timestep {i + 1}/{n_times}", end="", flush=True)

    if progress:
        print("\r" + " " * 40 + "\r", end="", flush=True)


def _mesh_output_name(fmt: str) -> str:
    return "fesom.mesh.diag.region.nc" if fmt == "netcdf" else "fesom.mesh.diag.region.zarr"


def _velocity_output_name(role: str, fmt: str) -> str:
    ext = "nc" if fmt == "netcdf" else "zarr"
    return f"{role}.region.{ext}"


def _combined_output_name(fmt: str) -> str:
    return "velocity.region.nc" if fmt == "netcdf" else "velocity.region.zarr"



def write_outputs(
    cfg: SubsetConfig,
    mesh_ds: xr.Dataset,
    velocity_subsets: Mapping[str, _LazySubset],
    out_dir: Path,
    manifest: dict[str, Any],
) -> SubsetArtifacts:
    """Write mesh and velocity subset outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    def _disk_size_mb(path: Path) -> float:
        if path.is_dir():
            return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)
        return path.stat().st_size / (1024 * 1024)

    # --- mesh (small, write in one shot) ---
    mesh_path = out_dir / _mesh_output_name(cfg.output.format)
    n_vel = len(velocity_subsets)
    n_total = 1 + n_vel
    print(f"       [1/{n_total}] writing mesh...", end="", flush=True)
    t0 = time.monotonic()
    _write_dataset(mesh_ds, mesh_path, cfg.output.format, cfg.output.zarr_chunks)
    print(f" {_disk_size_mb(mesh_path):.1f} MB on disk ({_fmt_elapsed(time.monotonic() - t0)})")

    # --- velocity (streamed timestep-by-timestep) ---
    velocity_paths: Dict[str, Path] = {}

    if cfg.output.layout == "split":
        vel_items: list[tuple[str, _LazySubset, Path]] = []
        for role, lazy in velocity_subsets.items():
            out_path = out_dir / _velocity_output_name(role, cfg.output.format)
            vel_items.append((role, lazy, out_path))
            velocity_paths[role] = out_path

        for idx, (role, lazy, out_path) in enumerate(vel_items):
            step_num = 2 + idx
            print(f"       [{step_num}/{n_total}] writing {role}...", flush=True)
            t0 = time.monotonic()
            _write_lazy_subset(lazy, out_path, cfg.output.format, lazy.global_attrs,
                               chunk_overrides=cfg.output.zarr_chunks)
            disk_mb = _disk_size_mb(out_path)
            print(f"       [{step_num}/{n_total}] {role}: {disk_mb:.1f} MB on disk ({_fmt_elapsed(time.monotonic() - t0)})")
    else:
        # Combined layout: write all variables together, one timestep at a time.
        out_path = out_dir / _combined_output_name(cfg.output.format)
        velocity_paths["combined"] = out_path
        # Use the first subset's global_attrs for the combined file.
        first_lazy = next(iter(velocity_subsets.values()))
        shared_attrs = first_lazy.global_attrs

        tdim = first_lazy.time_dim
        has_time = tdim is not None and first_lazy.source_da.sizes.get(tdim, 0) > 1
        n_times = first_lazy.source_da.sizes.get(tdim, 1) if has_time else 1

        print(f"       [2/{n_total}] writing combined ({n_times} timesteps)...", flush=True)
        t0 = time.monotonic()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        for i in range(n_times):
            chunk_vars = {}
            for role, lazy in velocity_subsets.items():
                if has_time:
                    full_slice = lazy.source_da.isel({tdim: [i]}).compute()
                else:
                    full_slice = lazy.source_da.compute()
                subset = full_slice.isel({lazy.spatial_dim: lazy.spatial_idx})
                subset = subset.rename(lazy.out_name)
                if i == 0:
                    subset.attrs = lazy.out_attrs
                chunk_vars[role] = subset
            chunk_ds = xr.Dataset(chunk_vars)
            if i == 0:
                chunk_ds.attrs = _netcdf_safe_attrs(shared_attrs)
                chunk_ds = _sanitize_dataset_attrs(chunk_ds)

            t_enc = (_time_encoding(chunk_ds, tdim,
                                    first_lazy.source_da[tdim] if tdim in first_lazy.source_da.coords else None)
                     if (i == 0 and has_time) else {})
            if cfg.output.format == "netcdf":
                if i == 0:
                    chunk_ds.to_netcdf(out_path, unlimited_dims=[tdim] if has_time else None,
                                       encoding=t_enc)
                else:
                    # mode="a" would OVERWRITE the variables (round 2 review):
                    # extend the unlimited time dimension instead
                    _append_time_step_netcdf(out_path, chunk_ds, tdim)
            elif cfg.output.format == "zarr":
                if i == 0:
                    if out_path.exists():
                        shutil.rmtree(out_path)
                    zenc = _zarr_encoding(chunk_ds, cfg.output.zarr_chunks)
                    for name, e in t_enc.items():
                        zenc.setdefault(name, {}).update(e)
                    chunk_ds.to_zarr(out_path, mode="w", encoding=zenc)
                else:
                    chunk_ds.to_zarr(out_path, mode="a", append_dim=tdim)

            if has_time:
                print(f"\r         timestep {i + 1}/{n_times}", end="", flush=True)
        if has_time:
            print("\r" + " " * 40 + "\r", end="", flush=True)

        disk_mb = _disk_size_mb(out_path)
        print(f"       [2/{n_total}] combined: {disk_mb:.1f} MB on disk ({_fmt_elapsed(time.monotonic() - t0)})")

    # Consolidate zarr metadata so xr.open_dataset() doesn't warn.
    if cfg.output.format == "zarr":
        import zarr
        zarr.consolidate_metadata(str(mesh_path))
        for vp in velocity_paths.values():
            zarr.consolidate_metadata(str(vp))

    artifacts = SubsetArtifacts(mesh_path=mesh_path, velocity_paths=velocity_paths)
    if cfg.output.write_manifest:
        manifest_path = out_dir / "subset_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        artifacts.manifest_path = manifest_path
    return artifacts


def build_manifest(
    cfg: SubsetConfig,
    config_path: Optional[Path],
    config_digest: str,
    mesh: MeshSource,
    selection: SubsetSelection,
    time_indices: Optional[np.ndarray],
    artifacts: Optional[SubsetArtifacts] = None,
) -> dict[str, Any]:
    """Build a JSON-serializable manifest."""
    manifest: dict[str, Any] = {
        "tool": "utils.extract_fesom_region",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path) if config_path else None,
        "config_sha256": config_digest,
        "source": {
            "mesh": str(mesh.path),
            "input": cfg.source.input,
            "u_path": cfg.source.u_path,
            "v_path": cfg.source.v_path,
            "w_path": cfg.source.w_path,
            "ssh_path": cfg.source.ssh_path,
        },
        "region": {
            "mode": cfg.region.mode,
            "bbox": list(cfg.region.bbox) if cfg.region.bbox else None,
            "center": list(cfg.region.center) if cfg.region.center else None,
            "radius_km": cfg.region.radius_km,
            "polygon_path": cfg.region.polygon_path,
            "polygon_format": cfg.region.polygon_format,
            "halo_layers": cfg.region.halo_layers,
        },
        "time": {
            "start": cfg.time.start,
            "stop": cfg.time.stop,
            "step": cfg.time.step,
            "selected_indices": time_indices.tolist() if time_indices is not None else None,
        },
        "output": {
            "dir": cfg.output.dir,
            "format": cfg.output.format,
            "layout": cfg.output.layout,
        },
        "source_counts": {
            "nodes": mesh.n_nodes,
            "elements": mesh.n_elements,
        },
        "subset_counts": {
            "nodes": int(selection.node_ids.size),
            "elements": int(selection.elem_ids.size),
        },
        "metadata": asdict(cfg.metadata),
    }
    if artifacts is not None:
        manifest["artifacts"] = {
            "mesh": str(artifacts.mesh_path),
            "velocity": {k: str(v) for k, v in artifacts.velocity_paths.items()},
            "manifest": str(artifacts.manifest_path) if artifacts.manifest_path else None,
            "config_copy": str(artifacts.config_copy_path) if artifacts.config_copy_path else None,
        }
    return manifest


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes}m {secs:.0f}s"


def run_subset(
    cfg: SubsetConfig,
    config_path: Optional[Path],
    config_text: str,
) -> SubsetArtifacts:
    """Run the full subset workflow."""
    t_total = time.monotonic()
    config_digest = hashlib.sha256(config_text.encode("utf-8")).hexdigest()

    print(f"[1/6] Loading mesh: {cfg.source.mesh}")
    t0 = time.monotonic()
    mesh = load_mesh_source(cfg.source.mesh)
    print(f"       {mesh.n_nodes:,} nodes, {mesh.n_elements:,} elements ({_fmt_elapsed(time.monotonic() - t0)})")

    has_extra = len(cfg.source.extra_vars) > 0
    n_steps = 7 if has_extra else 6

    print(f"[2/{n_steps}] Opening velocity sources...")
    t0 = time.monotonic()
    velocity_sources = load_velocity_sources(cfg.source)
    roles = ", ".join(sorted(velocity_sources.keys()))
    print(f"       variables: {roles} ({_fmt_elapsed(time.monotonic() - t0)})")

    extra_sources: Dict[str, VarSource] = {}
    if has_extra:
        print(f"       Opening extra variables...")
        t0 = time.monotonic()
        extra_sources = load_extra_var_sources(cfg.source.extra_vars)
        extra_roles = ", ".join(sorted(extra_sources.keys()))
        print(f"       extra: {extra_roles} ({_fmt_elapsed(time.monotonic() - t0)})")

    try:
        print(f"[3/{n_steps}] Selecting region (mode={cfg.region.mode})...")
        t0 = time.monotonic()
        selected_mask = select_region(mesh, cfg.region)
        if cfg.region.halo_layers > 0:
            print(f"       expanding halo: {cfg.region.halo_layers} layers...")
            selected_mask = expand_halo(mesh, selected_mask, cfg.region.halo_layers)
        selection = build_subset_selection(mesh, selected_mask)
        pct_nodes = 100.0 * selection.node_ids.size / mesh.n_nodes
        pct_elems = 100.0 * selection.elem_ids.size / mesh.n_elements
        print(
            f"       {selection.node_ids.size:,} nodes ({pct_nodes:.1f}%), "
            f"{selection.elem_ids.size:,} elements ({pct_elems:.1f}%) "
            f"({_fmt_elapsed(time.monotonic() - t0)})"
        )

        primary_time_var = velocity_sources["u"].data
        time_indices = resolve_time_indices(primary_time_var, cfg.time)
        if time_indices is not None:
            print(f"       time steps: {time_indices.size}")

        provenance = build_provenance_attrs(
            cfg=cfg,
            config_path=config_path,
            config_text=config_text,
            config_digest=config_digest,
            mesh=mesh,
            selection=selection,
            time_indices=time_indices,
        )

        print(f"[4/{n_steps}] Building subset mesh dataset...")
        t0 = time.monotonic()
        mesh_ds = build_subset_mesh_dataset(
            mesh,
            selection,
            attrs=provenance,
            copy_source_attrs=cfg.output.copy_source_attrs,
        )
        print(f"       done ({_fmt_elapsed(time.monotonic() - t0)})")

        print(f"[5/{n_steps}] Preparing velocity subsets...")
        t0 = time.monotonic()
        all_subsets: Dict[str, _LazySubset] = {}
        n_roles = len(velocity_sources)
        for i, (role, src) in enumerate(velocity_sources.items(), 1):
            t_var = time.monotonic()
            lazy = subset_var_array(src, mesh, selection, time_indices)
            global_attrs = dict(src.dataset.attrs) if cfg.output.copy_source_attrs else {}
            global_attrs.update(provenance)
            lazy.global_attrs = global_attrs
            all_subsets[role] = lazy
            shape_str = " x ".join(str(s) for s in lazy.shape)
            print(f"       [{i}/{n_roles}] {role}: {shape_str} ({_fmt_elapsed(time.monotonic() - t_var)})")
        print(f"       all variables done ({_fmt_elapsed(time.monotonic() - t0)})")

        if has_extra:
            print(f"[6/{n_steps}] Preparing extra variable subsets...")
            t0 = time.monotonic()
            n_extra = len(extra_sources)
            for i, (role, src) in enumerate(extra_sources.items(), 1):
                t_var = time.monotonic()
                lazy = subset_var_array(src, mesh, selection, time_indices)
                global_attrs = dict(src.dataset.attrs) if cfg.output.copy_source_attrs else {}
                global_attrs.update(provenance)
                lazy.global_attrs = global_attrs
                all_subsets[role] = lazy
                shape_str = " x ".join(str(s) for s in lazy.shape)
                print(f"       [{i}/{n_extra}] {role}: {shape_str} ({_fmt_elapsed(time.monotonic() - t_var)})")
            print(f"       all extra variables done ({_fmt_elapsed(time.monotonic() - t0)})")

        write_step = n_steps
        print(f"[{write_step}/{n_steps}] Writing output ({cfg.output.format}, {cfg.output.layout})...")
        t0 = time.monotonic()
        out_dir = Path(cfg.output.dir).expanduser().resolve()
        manifest = build_manifest(
            cfg=cfg,
            config_path=config_path,
            config_digest=config_digest,
            mesh=mesh,
            selection=selection,
            time_indices=time_indices,
        )
        artifacts = write_outputs(cfg, mesh_ds, all_subsets, out_dir, manifest)

        if cfg.output.write_config_copy:
            config_copy_path = out_dir / "subset_config.used.yaml"
            config_copy_path.write_text(config_text)
            artifacts.config_copy_path = config_copy_path

        if cfg.output.write_manifest:
            manifest = build_manifest(
                cfg=cfg,
                config_path=config_path,
                config_digest=config_digest,
                mesh=mesh,
                selection=selection,
                time_indices=time_indices,
                artifacts=artifacts,
            )
            assert artifacts.manifest_path is not None
            artifacts.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        print(f"       written to {cfg.output.dir} ({_fmt_elapsed(time.monotonic() - t0)})")

        total = _fmt_elapsed(time.monotonic() - t_total)
        print(
            f"\nDone: {selection.node_ids.size:,} nodes, "
            f"{selection.elem_ids.size:,} elements, "
            f"{len(all_subsets)} variables -> {cfg.output.dir} ({total})"
        )
        return artifacts
    finally:
        mesh.ds.close()
        close_velocity_sources(velocity_sources)
        close_velocity_sources(extra_sources)


def main() -> int:
    """CLI entry point."""
    args = build_parser().parse_args()
    cfg, _, config_path, config_text = load_subset_config(args)
    run_subset(cfg, config_path, config_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
