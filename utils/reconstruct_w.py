#!/usr/bin/env python3
"""Offline FESOM2 vertical velocity (W) reconstruction utility.

Reconstructs Cartesian vertical velocity from horizontal velocities
(u, v) and sea surface height (ssh).  Writes the result to a NetCDF
or Zarr file that can be directly read by kelp's FESOM2 reader
via ``open_dataset(w_path=...)``.

Usage::

    python utils/reconstruct_w.py config.yaml
    python utils/reconstruct_w.py config.yaml --method divergence

See ``utils/reconstruct_w.example.yaml`` for configuration format.
See ``docs/w_reconstruction.md`` for physics and derivation.
"""
from __future__ import annotations

import argparse
import sys
import time as _time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
import yaml

# Add parent directory to path so we can import kelp
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from kelp.readers import (get_reader, extract_time_info, _detect_engine,
                          _require_same_time_axis)
from kelp.w_reconstruct import (
    compute_ssh_tendency,
    compute_divergence_approx,
    compute_divergence_with_area,
    integrate_continuity,
    reconstruct_w_ale,
    compare_variants,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceConfig:
    meshpath: str = ""
    u_path: str = ""
    v_path: str = ""
    ssh_path: str = ""
    w_ale_path: Optional[str] = None


@dataclass(frozen=True)
class OutputConfig:
    dir: str = "."
    format: str = "netcdf"
    filename: Optional[str] = None


@dataclass(frozen=True)
class ReconstructConfig:
    source: SourceConfig = field(default_factory=SourceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    method: str = "auto"
    correct_bottom: bool = True
    use_nod_area: bool = True
    max_levels: Optional[int] = None
    time_range: Optional[list] = None


def _load_config(path: str) -> ReconstructConfig:
    """Load configuration from YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    src = SourceConfig(**raw.get("source", {}))
    out = OutputConfig(**raw.get("output", {}))

    return ReconstructConfig(
        source=src,
        output=out,
        method=raw.get("method", "auto"),
        correct_bottom=raw.get("correct_bottom", True),
        use_nod_area=raw.get("use_nod_area", True),
        max_levels=raw.get("max_levels"),
        time_range=raw.get("time_range"),
    )


# ---------------------------------------------------------------------------
# Mesh extras reader
# ---------------------------------------------------------------------------

def _read_mesh_extras(meshpath: str) -> dict:
    """Read additional fields from mesh diag for W reconstruction.

    Returns dict with available fields (missing keys omitted):
        nod_area:         (nz, nod2) depth-dependent control volume area
        nlevels_nod2D:    (nod2,)    active level count per node
        nlevels:          (elem,)    active level count per element
        ulevels:          (elem,)    upper active level per element
    """
    p = Path(meshpath).expanduser()
    engine = _detect_engine(p)
    ds = xr.open_dataset(str(p), engine=engine)

    extras = {}
    field_dtypes = {
        "nod_area": np.float64,
        "nlevels_nod2D": np.int32,
        "nlevels": np.int32,
        "ulevels": np.int32,
    }
    for key, dtype in field_dtypes.items():
        if key in ds:
            extras[key] = np.asarray(ds[key].values, dtype=dtype)
            print(f"    Found {key}: shape {extras[key].shape}")

    ds.close()

    # Cavity check
    if "ulevels" in extras and extras["ulevels"].max() > 1:
        warnings.warn(
            f"Mesh has cavities (max ulevels={extras['ulevels'].max()}). "
            "W reconstruction does not handle cavities — results near "
            "ice shelves will be inaccurate."
        )

    return extras


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

#: encoding keys of the source time axis that the output keeps, so any reader
#: decodes the W stamps to exactly the velocity file's instants
_TIME_ENCODING_KEYS = ("units", "calendar", "dtype")


def _write_output(w_all, time_coord, z_levels, output_cfg, meshpath):
    """Write reconstructed W to NetCDF or Zarr.

    ``time_coord`` is the SOURCE velocity file's time coordinate for the
    written records (an xarray DataArray: its values, attributes and its
    units/calendar encoding are kept) or a plain array of stamps.  The
    readers attach a W file onto the velocity dataset by these labels, so
    they must be the velocity file's own stamps — the numeric seconds written
    until 2026-10-04 matched nothing and the attached W came out all-NaN.
    """
    n_times, K, N = w_all.shape

    out_dir = Path(output_cfg.dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if output_cfg.filename:
        fname = output_cfg.filename
    else:
        ext = ".zarr" if output_cfg.format == "zarr" else ".nc"
        fname = f"w.reconstructed{ext}"

    out_path = out_dir / fname

    ds = xr.Dataset(
        {
            # Interfaces go on ``nz`` (K of them): ``nz1`` is the FESOM2 name for the
            # K-1 mid-layers, and the reader rejects a W that carries it.
            "w": (["time", "nz", "nod2"], w_all),
        },
        coords={
            "time": np.asarray(getattr(time_coord, "values", time_coord)),
            "nz": z_levels[:K],
        },
        attrs={
            "description": "Reconstructed Cartesian vertical velocity",
            "source_mesh": str(meshpath),
            "units": "m/s",
            "sign_convention": "positive-down (w > 0 = downwelling)",
            "nz_units": "metres (positive-down depth of z-level interfaces)",
        },
    )

    # Variable-level CF attribute: the KELP readers' ``mode.w_positive:
    # auto`` honours it, so this positive-down W is not flipped the way a
    # native (positive-up) FESOM2 ``w`` is.
    ds["w"].attrs.update({"positive": "down", "units": "m/s"})
    if hasattr(time_coord, "attrs"):
        ds["time"].attrs.update(time_coord.attrs)
        ds["time"].encoding.update(
            {k: v for k, v in getattr(time_coord, "encoding", {}).items()
             if k in _TIME_ENCODING_KEYS})

    if output_cfg.format == "zarr":
        encoding = {"w": {"chunks": (min(32, n_times), K, min(8192, N))}}
        ds.to_zarr(str(out_path), mode="w", encoding=encoding)
    else:
        encoding = {"w": {"zlib": True, "complevel": 4}}
        ds.to_netcdf(str(out_path), encoding=encoding)

    print(f"  Written: {out_path}  ({n_times} timesteps, {K} levels, {N} nodes)")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _w_ale_positive_down(ds_wale, key, ti, n_keep=None):
    """The model's ALE ``w`` at record ``ti`` in the positive-DOWN sign the
    ALE heuristic expects.  FESOM2 writes ``w`` positive UP (oce_ale.F90);
    the file convention is resolved like the tracker's readers (CF
    ``positive`` attribute, else FESOM2's ``up``).  Until 2026-09-23 the raw
    positive-up field went in, and the output was then stamped positive-down
    (review 2026-09-23)."""
    from kelp.readers import resolve_w_sign
    arr = np.asarray(ds_wale[key].isel(time=ti).values, dtype=np.float64)
    arr = resolve_w_sign(ds_wale[key], None, "up") * arr
    if n_keep is not None and arr.ndim == 2:
        # W has K interfaces for K-1 velocity layers
        arr = arr[:n_keep]
    return arr


#: where the u variable is looked for: the tool's own names and the name of
#: FESOM 1.4 level-wise output, refused by _refuse_level_inputs
_U_NAMES = ("u", "ua", "u_vel", "uo")

_LEVEL_FIX = (
    "This tool integrates layer u/v (K-1 layers between the K interfaces, "
    "FESOM2) only. FESOM 1.4 output carries the vertical velocity ('wo'): "
    "track with it (mesh.type: fesom14) instead of reconstructing it")


def _level_layout(ds, name: str) -> Optional[str]:
    """How ``ds[name]`` shows FESOM 1.4 data on the levels, or None."""
    from kelp.readers.fesom14._layout import DEPTH_DIMS, HORIZONTAL_DIMS
    dims = tuple(ds[name].dims)
    if len(dims) == 2 and dims[1] == "nodes_3d":
        return f"one vector over the 3-D nodes (dimensions {dims})"
    if len(dims) == 3 and dims[1] in DEPTH_DIMS and dims[2] in HORIZONTAL_DIMS:
        if dims[1] in ds.variables:
            z = np.asarray(ds[dims[1]].values, dtype=np.float64)
            if z.size and np.isfinite(z[0]) and abs(z[0]) < 1e-6:
                # layer centres lie below the surface; a level does not
                return (f"on the levels (dimensions {dims}, the first "
                        f"{dims[1]!r} at 0 m)")
    return None


def _refuse_level_inputs(source: SourceConfig) -> None:
    """Refuse a FESOM 1.4 mesh or u data on the levels, before any mesh is
    loaded or record read.

    Raises
    ------
    ValueError
        ``source.meshpath`` is a FESOM 1.4 ASCII mesh directory (it holds
        ``nod3d.out``), or the u variable of ``source.u_path`` is in a FESOM
        1.4 layout: one vector over the 3-D nodes, or level-wise on levels
        starting at the surface.
    """
    mp = Path(source.meshpath).expanduser() if source.meshpath else None
    if mp is not None and mp.is_dir() and (mp / "nod3d.out").is_file():
        raise ValueError(
            f"source.meshpath {mp} is a FESOM 1.4 ASCII mesh (it holds "
            f"nod3d.out), whose u/v lie on the levels. {_LEVEL_FIX}")
    up = Path(source.u_path).expanduser() if source.u_path else None
    if up is None or not up.exists():
        return
    ds = xr.open_dataset(up, engine=_detect_engine(up), decode_times=False)
    try:
        for name in _U_NAMES:
            if name in ds.variables:
                why = _level_layout(ds, name)
                if why is not None:
                    raise ValueError(
                        f"source.u_path {up}: {name!r} is FESOM 1.4 data, "
                        f"{why}. {_LEVEL_FIX}")
    finally:
        ds.close()


def run(cfg: ReconstructConfig):
    """Execute W reconstruction for z-level meshes only.

    FESOM 1.4 inputs (an ASCII mesh, u on the levels) are refused before
    anything is loaded: the reconstruction integrates layer u/v.
    """
    t0 = _time.perf_counter()
    print(f"W velocity reconstruction")
    print(f"  Method: {cfg.method}")
    print(f"  Bottom correction: {cfg.correct_bottom}")
    _refuse_level_inputs(cfg.source)

    # --- Load mesh ---
    from types import SimpleNamespace
    mesh_cfg = SimpleNamespace(meshpath=cfg.source.meshpath)
    reader = get_reader("fesom2")
    mesh = reader.load_mesh(mesh_cfg)

    z_levels = np.abs(mesh.sigma)
    K = len(z_levels)
    N = mesh.n_nodes
    print(f"  Mesh: {N:,} nodes, {mesh.n_elements:,} elements, {K} z-levels")

    # --- Mesh extras (Tier 2) ---
    mesh_extras = {}
    if cfg.use_nod_area:
        print(f"  Reading mesh extras from {cfg.source.meshpath} ...")
        mesh_extras = _read_mesh_extras(cfg.source.meshpath)

    # --- Open velocity datasets ---
    print(f"  Opening datasets ...")
    ds_u = xr.open_dataset(
        Path(cfg.source.u_path).expanduser(),
        engine=_detect_engine(cfg.source.u_path),
    )
    if cfg.source.v_path and cfg.source.v_path != cfg.source.u_path:
        ds_v = xr.open_dataset(
            Path(cfg.source.v_path).expanduser(),
            engine=_detect_engine(cfg.source.v_path),
        )
    else:
        ds_v = ds_u

    # SSH dataset
    if cfg.source.ssh_path and cfg.source.ssh_path != cfg.source.u_path:
        ds_ssh = xr.open_dataset(
            Path(cfg.source.ssh_path).expanduser(),
            engine=_detect_engine(cfg.source.ssh_path),
        )
    else:
        ds_ssh = ds_u

    # W ALE dataset (optional)
    ds_wale = None
    if cfg.source.w_ale_path:
        ds_wale = xr.open_dataset(
            Path(cfg.source.w_ale_path).expanduser(),
            engine=_detect_engine(cfg.source.w_ale_path),
        )

    # --- Extract time info ---
    times, file_dt = extract_time_info(ds_u)
    if times is None:
        raise ValueError("Cannot extract time coordinate from u dataset")

    n_times = len(times)
    print(f"  Time: {n_times} steps, dt={file_dt:.0f}s" if file_dt else
          f"  Time: {n_times} steps, irregular spacing")

    # --- Time range ---
    t_start = 0
    t_stop = n_times
    if cfg.time_range:
        t_start = cfg.time_range[0]
        t_stop = min(cfg.time_range[1], n_times)
    n_out = t_stop - t_start

    # --- Find variable names ---
    u_key = _find_var(ds_u, ("u", "ua", "u_vel"), "u-velocity")
    v_key = _find_var(ds_v, ("v", "va", "v_vel"), "v-velocity")
    ssh_key = _find_var(ds_ssh, ("ssh", "eta", "zos"), "SSH")

    # The inputs are paired by record POSITION below (isel(time=ti) on each)
    # and the output is stamped with u's time axis: a v, ssh or ALE-W file on
    # other records would give a wrong W that looks valid.
    u_da = ds_u[u_key]
    if ds_v is not ds_u:
        _require_same_time_axis(u_da, ds_v[v_key],
                                f"v file {cfg.source.v_path}",
                                f"the u file {cfg.source.u_path}")
    if ds_ssh is not ds_u:
        _require_same_time_axis(u_da, ds_ssh[ssh_key],
                                f"ssh file {cfg.source.ssh_path}",
                                f"the u file {cfg.source.u_path}")
    if ds_wale is not None:
        _require_same_time_axis(u_da, ds_wale,
                                f"ALE w file {cfg.source.w_ale_path}",
                                f"the u file {cfg.source.u_path}")

    # Check velocity units
    u_units = ds_u[u_key].attrs.get("units", "")
    if u_units and "m/s" not in u_units.lower() and u_units.lower() not in ("", "m s-1"):
        warnings.warn(
            f"u velocity units='{u_units}' — expected 'm/s'. "
            "If these are transport velocities (m^2/s), results will be wrong."
        )

    # --- Reconstruct ---
    print(f"  Processing {n_out} timesteps [{t_start}:{t_stop}] ...")
    w_all = np.zeros((n_out, K, N), dtype=np.float64)

    # Pre-fetch nod_area/nlevels for Tier 2
    nod_area = mesh_extras.get("nod_area")
    nlevels_elem = mesh_extras.get("nlevels")
    ulevels_elem = mesh_extras.get("ulevels")
    nlevels_nod = mesh_extras.get("nlevels_nod2D")

    residuals = []

    for out_i, ti in enumerate(range(t_start, t_stop)):
        step_t0 = _time.perf_counter()

        # Read u, v for this timestep
        u_var = ds_u[u_key].isel(time=ti)
        v_var = ds_v[v_key].isel(time=ti)

        # Strip depth to get (K-1, E) shape
        u_arr = np.asarray(u_var.values, dtype=np.float64)
        v_arr = np.asarray(v_var.values, dtype=np.float64)
        if cfg.max_levels and u_arr.ndim == 2:
            u_arr = u_arr[:cfg.max_levels]
            v_arr = v_arr[:cfg.max_levels]

        # SSH tendency: centered difference with real intervals
        if ti == t_start and ti == 0:
            # Forward difference at first step
            eta_prev = np.asarray(ds_ssh[ssh_key].isel(time=ti).values, dtype=np.float64)
            eta_next = np.asarray(ds_ssh[ssh_key].isel(time=min(ti + 1, n_times - 1)).values, dtype=np.float64)
            dt_real = times[min(ti + 1, n_times - 1)] - times[ti] if n_times > 1 else 1.0
        elif ti == t_stop - 1 and ti == n_times - 1:
            # Backward difference at last step
            eta_prev = np.asarray(ds_ssh[ssh_key].isel(time=ti - 1).values, dtype=np.float64)
            eta_next = np.asarray(ds_ssh[ssh_key].isel(time=ti).values, dtype=np.float64)
            dt_real = times[ti] - times[ti - 1]
        else:
            # Centered difference
            eta_prev = np.asarray(ds_ssh[ssh_key].isel(time=ti - 1).values, dtype=np.float64)
            eta_next = np.asarray(ds_ssh[ssh_key].isel(time=ti + 1).values, dtype=np.float64)
            dt_real = times[ti + 1] - times[ti - 1]

        if dt_real <= 0:
            raise ValueError(
                f"Non-positive time interval dt_real={dt_real:.1f}s at "
                f"timestep {ti}. Check for duplicate or out-of-order "
                f"timestamps in the input time axis."
            )

        ssh_t = compute_ssh_tendency(eta_prev, eta_next, dt_real)

        n_layers = u_arr.shape[0]
        z_lev = z_levels[:n_layers + 1]

        # --- Method dispatch ---
        if cfg.method == "ale":
            # Variant B only: ALE heuristic
            if ds_wale is None:
                raise ValueError("method='ale' requires w_ale_path in config")
            if out_i == 0:
                w_ale_key = _find_var(ds_wale, ("w", "wo", "wvel"), "w_ALE")
            w_ale_arr = _w_ale_positive_down(
                ds_wale, w_ale_key, ti,
                n_layers + 1 if cfg.max_levels else None)
            w_out = reconstruct_w_ale(
                w_ale_arr, z_lev[:w_ale_arr.shape[0]], ssh_t, mesh.depth,
            )
            bottom_res = 0.0  # no divergence residual for ALE method

        else:
            # Variant A: divergence (used by both "divergence" and "auto")
            if nod_area is not None:
                div_h = compute_divergence_with_area(
                    u_arr, v_arr, mesh, nod_area, nlevels_elem, ulevels_elem,
                )
            else:
                div_h = compute_divergence_approx(u_arr, v_arr, mesh)

            w_raw = integrate_continuity(
                div_h, z_lev, ssh_t, mesh.depth,
                nlevels_nod=nlevels_nod,
                correct_bottom=False,
            )
            bottom_res = np.max(np.abs(w_raw[-1, :]))

            if cfg.correct_bottom:
                w_out = integrate_continuity(
                    div_h, z_lev, ssh_t, mesh.depth,
                    nlevels_nod=nlevels_nod,
                    correct_bottom=True,
                )
            else:
                w_out = w_raw

            # Auto mode: cross-validate against ALE if available
            if cfg.method == "auto" and ds_wale is not None and out_i == 0:
                if out_i == 0:
                    w_ale_key = _find_var(ds_wale, ("w", "wo", "wvel"), "w_ALE")
                w_ale_arr = _w_ale_positive_down(
                    ds_wale, w_ale_key, ti,
                    n_layers + 1 if cfg.max_levels else None)
                w_ale_cart = reconstruct_w_ale(
                    w_ale_arr, z_lev[:w_ale_arr.shape[0]], ssh_t, mesh.depth,
                )
                diag = compare_variants(
                    w_out[:w_ale_arr.shape[0]], w_ale_cart,
                    z_lev[:w_ale_arr.shape[0]], ssh_t, w_ale_arr, mesh.depth,
                )
                print(f"  ALE diagnostics (timestep {ti}):")
                print(f"    |w_div - w_ale_cart| max={diag['err_abs_max']:.2e}"
                      f"  mean={diag['err_abs_mean']:.2e} m/s")
                print(f"    chi_ALE max={diag['chi_ALE_max']:.2f}"
                      f"  mean={diag['chi_ALE_mean']:.2f}")

        residuals.append(bottom_res)

        # Pad to full K levels if max_levels truncated
        if w_out.shape[0] < K:
            w_full = np.zeros((K, N), dtype=np.float64)
            w_full[:w_out.shape[0]] = w_out
            w_out = w_full

        w_all[out_i] = w_out

        elapsed = _time.perf_counter() - step_t0
        if (out_i + 1) % max(1, n_out // 10) == 0 or out_i == 0:
            print(f"    Step {out_i + 1}/{n_out}: {elapsed*1000:.0f}ms"
                  f"  |w_bottom|={bottom_res:.2e}")

    # --- Summary ---
    residuals = np.array(residuals)
    print(f"\n  Bottom residual (before correction):")
    print(f"    max={residuals.max():.2e}  mean={residuals.mean():.2e}"
          f"  median={np.median(residuals):.2e} m/s")
    print(f"  W statistics:")
    print(f"    max={np.max(w_all):.2e}  min={np.min(w_all):.2e}"
          f"  mean={np.mean(w_all):.2e} m/s")

    # --- Write output ---
    # The source's own time coordinate (values + units/calendar encoding),
    # not the relative seconds of ``times``: the readers attach the W file
    # onto the velocity file by these labels.
    time_out = ds_u["time"].isel(time=slice(t_start, t_stop))
    out_path = _write_output(w_all, time_out, z_levels,
                             cfg.output, cfg.source.meshpath)

    elapsed_total = _time.perf_counter() - t0
    print(f"\n  Done in {elapsed_total:.1f}s")
    return out_path


# Variable names that indicate transport (m²/s) rather than velocity (m/s)
_TRANSPORT_NAMES = {"uh", "vh", "ut", "vt", "u_transport", "v_transport"}


def _find_var(ds, aliases, description):
    """Find a variable in the dataset by alias list.

    Warns if the matched variable looks like transport (uh/vh) rather
    than velocity (u/v).
    """
    for alias in aliases:
        if alias in ds:
            return alias
    available = list(ds.data_vars)
    if len(available) == 1:
        name = available[0]
        if name.lower() in _TRANSPORT_NAMES:
            warnings.warn(
                f"Only variable in dataset is '{name}', which looks like "
                f"transport velocity (m²/s), not Cartesian velocity (m/s). "
                f"W reconstruction assumes u/v in m/s. If this is transport, "
                f"results will be wrong."
            )
        return name
    raise KeyError(
        f"Cannot find {description} variable. "
        f"Tried {aliases}, available: {available}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct Cartesian W from FESOM2 u, v, ssh",
    )
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--method", choices=["divergence", "ale", "auto"],
                        default=None, help="Override method from config")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    if args.method:
        cfg = ReconstructConfig(
            source=cfg.source, output=cfg.output,
            method=args.method,
            correct_bottom=cfg.correct_bottom,
            use_nod_area=cfg.use_nod_area,
            max_levels=cfg.max_levels,
            time_range=cfg.time_range,
        )

    run(cfg)


if __name__ == "__main__":
    main()
