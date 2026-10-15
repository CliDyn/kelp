#!/usr/bin/env python3
"""Create small real-mesh fixtures for regression tests.

Run once manually to generate fixture files in tests/fixtures/.
These files are committed to the repo (~50-100 KB each).

Usage::

    source .venv/bin/activate
    python tests/create_fixtures.py

Requires access to the real mesh data on the local machine.
"""
from __future__ import annotations

import sys
from pathlib import Path

import os

import numpy as np
import xarray as xr

# Add project root to path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from kelp.core.mesh import Mesh, build_node_to_elems, build_search

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURES_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Data paths: a FESOM-C output file and a FESOM2 mesh diagnostic, taken from
# the environment (KELP_FIXTURE_FESOMC, KELP_FIXTURE_FESOM2_MESH)
# ---------------------------------------------------------------------------
FESOMC_PATH = Path(os.environ.get("KELP_FIXTURE_FESOMC", "fesomc_input.nc")).expanduser()
FESOM2_MESH_PATH = Path(os.environ.get("KELP_FIXTURE_FESOM2_MESH", "fesom.mesh.diag.nc")).expanduser()


# ---------------------------------------------------------------------------
# FESOM-C mesh extraction
# ---------------------------------------------------------------------------

def _extract_fesomc(src_path: Path, lon_min: float, lon_max: float,
                    lat_min: float, lat_max: float, name: str) -> Path:
    """Extract a FESOM-C sub-mesh by bounding box.

    Preserves raw connectivity format (1-based, 4 columns with padding).
    """
    print(f"\n{'='*60}")
    print(f"FESOM-C: {name}")
    print(f"  bbox: lon [{lon_min}, {lon_max}], lat [{lat_min}, {lat_max}]")

    ds = xr.open_dataset(src_path)
    lon = ds["lon"].values
    lat = ds["lat"].values

    # Select nodes inside bbox
    node_mask = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
    node_ids = np.where(node_mask)[0]

    if len(node_ids) == 0:
        print(f"  WARNING: no nodes in bbox!")
        ds.close()
        return None

    # Build node ID mapping: old → new
    node_map = -np.ones(len(lon), dtype=np.int32)
    node_map[node_ids] = np.arange(len(node_ids), dtype=np.int32)

    # Select elements where ALL vertices are inside
    nv = ds["nv"].values  # (nele, 4), 1-based
    nv_0based = nv.copy() - 1
    nv_0based[nv_0based < 0] = -1

    elem_mask = np.ones(nv_0based.shape[0], dtype=bool)
    for col in range(nv_0based.shape[1]):
        col_ids = nv_0based[:, col]
        # Treat -1 padding as OK
        valid = (col_ids >= 0)
        inside = np.zeros(len(col_ids), dtype=bool)
        inside[valid] = node_mask[col_ids[valid]]
        inside[~valid] = True  # padding is always OK
        elem_mask &= inside

    elem_ids = np.where(elem_mask)[0]
    print(f"  nodes: {len(node_ids)}, elements: {len(elem_ids)}")

    # Remap connectivity (keep 1-based format with padding)
    nv_sub = nv[elem_ids].copy()
    for i in range(nv_sub.shape[0]):
        for j in range(nv_sub.shape[1]):
            old = nv_sub[i, j] - 1  # to 0-based
            if old < 0:
                nv_sub[i, j] = 0  # keep padding as 0 (will be -1 after sanitize)
            else:
                nv_sub[i, j] = node_map[old] + 1  # back to 1-based

    # Build output dataset
    out = xr.Dataset({
        "lon": (["node"], lon[node_ids]),
        "lat": (["node"], lat[node_ids]),
        "depth": (["node"], ds["depth"].values[node_ids] if "depth" in ds else
                  np.zeros(len(node_ids))),
        "nv": (["nele", "four"], nv_sub),
    })

    # Element centres
    if "lon_elem" in ds and "lat_elem" in ds:
        out["lon_elem"] = (["nele"], ds["lon_elem"].values[elem_ids])
        out["lat_elem"] = (["nele"], ds["lat_elem"].values[elem_ids])

    # Sigma levels (copy entirely)
    if "sigma_lev" in ds:
        out["sigma_lev"] = ds["sigma_lev"]

    # Metadata
    out.attrs["source"] = str(src_path.name)
    out.attrs["fixture_name"] = name
    out.attrs["bbox"] = f"lon[{lon_min},{lon_max}] lat[{lat_min},{lat_max}]"

    out_path = FIXTURES_DIR / f"fesomc_{name}.nc"
    out.to_netcdf(out_path)
    ds.close()

    size_kb = out_path.stat().st_size / 1024
    print(f"  written: {out_path.name} ({size_kb:.0f} KB)")
    return out_path


# ---------------------------------------------------------------------------
# FESOM2 mesh extraction
# ---------------------------------------------------------------------------

def _extract_fesom2(mesh_path: Path, lon_min: float, lon_max: float,
                    lat_min: float, lat_max: float, name: str,
                    max_levels: int = 10,
                    wrap_dateline: bool = False) -> Path:
    """Extract a FESOM2 sub-mesh by bounding box.

    Writes a mesh-only file in fesom.mesh.diag format (elements, nod_in_elem2D, etc.)
    """
    print(f"\n{'='*60}")
    print(f"FESOM2: {name}")
    print(f"  bbox: lon [{lon_min}, {lon_max}], lat [{lat_min}, {lat_max}]")
    if wrap_dateline:
        print(f"  (dateline-crossing bbox)")

    import netCDF4 as nc4
    ds = nc4.Dataset(str(mesh_path), "r")

    lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
    lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
    n_total = len(lon)

    # Node selection
    if wrap_dateline:
        # lon_min > lon_max means crossing dateline, e.g. [175, -175] = [175, 185]
        node_mask = ((lon >= lon_min) | (lon <= lon_max)) & \
                    (lat >= lat_min) & (lat <= lat_max)
    else:
        node_mask = (lon >= lon_min) & (lon <= lon_max) & \
                    (lat >= lat_min) & (lat <= lat_max)

    node_ids = np.where(node_mask)[0]

    if len(node_ids) == 0:
        print(f"  WARNING: no nodes in bbox!")
        ds.close()
        return None

    # Node mapping
    node_map = -np.ones(n_total, dtype=np.int32)
    node_map[node_ids] = np.arange(len(node_ids), dtype=np.int32)

    # Elements: (n3, elem) in file, 1-based
    elem_raw = np.asarray(ds.variables["elements"][:], dtype=np.int32)  # (3, E)
    elem_0 = elem_raw - 1  # 0-based, shape (3, E)

    # Select elements where ALL 3 vertices are inside
    elem_mask = node_mask[elem_0[0]] & node_mask[elem_0[1]] & node_mask[elem_0[2]]
    elem_ids = np.where(elem_mask)[0]

    print(f"  nodes: {len(node_ids)}, elements: {len(elem_ids)}")

    # Remap elements to new node IDs (keep 1-based for FESOM2 format)
    elem_sub = elem_raw[:, elem_ids].copy()  # (3, E_sub)
    for i in range(3):
        for j in range(elem_sub.shape[1]):
            old_0 = elem_sub[i, j] - 1
            elem_sub[i, j] = node_map[old_0] + 1  # 1-based new

    # Build node_to_elems for the subset
    N_sub = len(node_ids)
    elem_nodes_0based = (elem_sub - 1).T  # (E_sub, 3)
    n2e, n2e_count = build_node_to_elems(N_sub, elem_nodes_0based.astype(np.int32))
    # Convert back to 1-based for FESOM2 format (0 = padding)
    n2e_1based = n2e.copy()
    n2e_1based[n2e_1based >= 0] += 1
    n2e_1based[n2e_1based < 0] = 0

    # Depth
    if "zbar_n_bottom" in ds.variables:
        depth_sub = np.asarray(ds.variables["zbar_n_bottom"][:], dtype=np.float64)[node_ids]
    else:
        depth_sub = np.zeros(N_sub, dtype=np.float64)

    # Z-levels (copy entirely, optionally truncate)
    if "nz" in ds.variables:
        nz_full = np.asarray(ds.variables["nz"][:], dtype=np.float64)
        if max_levels and len(nz_full) > max_levels:
            nz_sub = nz_full[:max_levels]
        else:
            nz_sub = nz_full
    else:
        nz_sub = np.array([0.0], dtype=np.float64)

    ds.close()

    # Build output dataset
    out = xr.Dataset({
        "lon": (["nod2"], lon[node_ids]),
        "lat": (["nod2"], lat[node_ids]),
        "elements": (["n3", "elem"], elem_sub),
        "nod_in_elem2D": (["N", "nod2"], n2e_1based.T),  # (max_conn, N) → FESOM2 format
        "nod_in_elem2D_num": (["nod2"], n2e_count),
        "zbar_n_bottom": (["nod2"], depth_sub),
        "nz": (["nz"], nz_sub),
    })

    out.attrs["source"] = str(mesh_path.name)
    out.attrs["fixture_name"] = name
    out.attrs["bbox"] = f"lon[{lon_min},{lon_max}] lat[{lat_min},{lat_max}]"
    if max_levels:
        out.attrs["max_levels"] = max_levels

    out_path = FIXTURES_DIR / f"fesom2_{name}.nc"
    out.to_netcdf(out_path)

    size_kb = out_path.stat().st_size / 1024
    print(f"  written: {out_path.name} ({size_kb:.0f} KB)")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

#: a FESOM-C ASCII mesh directory (nod2d.out / elem2d.out / depth.out) packed
#: whole into one npz for the grid-hash cell tests (tests/test_grid_hash_cell_rule.py)
FESOMC_ASCII_DIR = Path(os.environ.get("KELP_FIXTURE_FESOMC_ASCII", "sns_multi_m8")).expanduser()


def _pack_fesomc_ascii(src_dir: Path, out: Path) -> None:
    """FESOM-C ASCII mesh (``nod2d.out``: count line, then ``id lon lat flag``;
    ``elem2d.out``: count line, then four 1-based node ids per row, a triangle
    repeating its first vertex; ``depth.out``: one value per node) -> one
    compressed npz: ``lon``, ``lat``, ``depth`` (positive down) float64 (N,),
    ``elem`` int32 (E, 4) 0-based in the FESOM-C repeated-first-vertex
    convention, ``source`` = the directory name."""
    nod = np.loadtxt(src_dir / "nod2d.out", skiprows=1)
    elem = np.loadtxt(src_dir / "elem2d.out", skiprows=1, dtype=np.int64) - 1
    depth = np.abs(np.loadtxt(src_dir / "depth.out"))
    assert nod.shape[0] == depth.shape[0] and elem.shape[1] == 4, (nod.shape, elem.shape, depth.shape)
    assert elem.min() >= 0 and elem.max() < nod.shape[0]
    np.savez_compressed(out, lon=nod[:, 1].astype(np.float64), lat=nod[:, 2].astype(np.float64),
                        depth=depth.astype(np.float64), elem=elem.astype(np.int32),
                        source=np.array(src_dir.name))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB): {nod.shape[0]} nodes, "
          f"{elem.shape[0]} elements")


def main():
    print("Creating real-mesh test fixtures")
    print(f"Output: {FIXTURES_DIR}/")

    # --- FESOM-C ---
    if FESOMC_PATH.exists():
        _extract_fesomc(FESOMC_PATH, 5.5, 7.5, 53.5, 55.0, "interior")
        _extract_fesomc(FESOMC_PATH, 3.0, 5.0, 52.0, 53.5, "coastal")
        _extract_fesomc(FESOMC_PATH, 7.5, 9.5, 55.0, 56.0, "boundary")
    else:
        print(f"\nWARNING: FESOM-C data not found at {FESOMC_PATH}"
              " — set KELP_FIXTURE_FESOMC")
    if FESOMC_ASCII_DIR.is_dir():
        _pack_fesomc_ascii(FESOMC_ASCII_DIR, FIXTURES_DIR / f"fesomc_{FESOMC_ASCII_DIR.name}.npz")
    else:
        print("\nWARNING: FESOM-C ASCII mesh directory not set — set KELP_FIXTURE_FESOMC_ASCII")

    # --- FESOM2 ---
    if FESOM2_MESH_PATH.exists():
        _extract_fesom2(FESOM2_MESH_PATH, 10, 25, -40, -30,
                        "midlat", max_levels=10)
        _extract_fesom2(FESOM2_MESH_PATH, 175, -175, -10, 0,
                        "dateline", max_levels=10, wrap_dateline=True)
        _extract_fesom2(FESOM2_MESH_PATH, 0, 60, 85, 89.5,
                        "pole", max_levels=10)
    else:
        print(f"\nWARNING: FESOM2 mesh not found at {FESOM2_MESH_PATH}"
              " — set KELP_FIXTURE_FESOM2_MESH")

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    total = 0
    for f in sorted(list(FIXTURES_DIR.glob("*.nc")) + list(FIXTURES_DIR.glob("*.npz"))):
        size = f.stat().st_size / 1024
        total += size
        print(f"  {f.name:30s} {size:6.0f} KB")
    print(f"  {'TOTAL':30s} {total:6.0f} KB")


if __name__ == "__main__":
    main()
