"""Synthetic FESOM2 fixtures for reader tests — no external data.

A 4-node square split into 2 triangles, with the mesh-diag variables our
loader needs, emitted in each of the three diag conventions; plus a
multi-year per-variable file set in the ``<var>.fesom.<year>.nc`` layout.
"""
from __future__ import annotations
from pathlib import Path
from typing import Callable

import numpy as np
import xarray as xr

# Geometry (degrees): unit square, 2 triangles.
LON_DEG = np.array([0.0, 1.0, 1.0, 0.0])
LAT_DEG = np.array([0.0, 0.0, 1.0, 1.0])
TRI_1BASED = np.array([[1, 2, 3], [1, 3, 4]], dtype=np.int32)  # (E=2, 3)
# node -> 1-based element ids (0-padded to max valence N=2)
NOD_IN_ELEM_1BASED = np.array([[1, 2], [1, 0], [1, 2], [2, 0]], dtype=np.int32)  # (nod=4, N=2)
NOD_IN_ELEM_NUM = np.array([2, 1, 2, 1], dtype=np.int32)
ZBAR_NEG = np.array([0.0, -10.0, -30.0])          # 3 interfaces, negative-down
ELEM_AREA = np.array([0.5, 0.5])
ZBAR_N_BOTTOM_NEG = np.array([-30.0, -30.0, -30.0, -30.0])
DEG2RAD = np.pi / 180.0


def _level_vars(elem_dim: str, nod_dim: str, *, nlevels=None, ulevels=None,
                zbar_e_bottom=None, nlevels_nod2D=None) -> dict:
    """The optional per-element / per-node level variables of a mesh diag,
    on the convention's element and node dimensions (``None`` = not
    written).  ``zbar_e_bottom`` is written negative, as FESOM2 writes it,
    whatever the sign given; ``nlevels``/``ulevels``/``nlevels_nod2D`` are
    written as int32, 1-based as FESOM2 writes them."""
    out = {}
    if nlevels is not None:
        out["nlevels"] = ((elem_dim,), np.asarray(nlevels, dtype=np.int32))
    if ulevels is not None:
        out["ulevels"] = ((elem_dim,), np.asarray(ulevels, dtype=np.int32))
    if zbar_e_bottom is not None:
        out["zbar_e_bottom"] = (
            (elem_dim,), -np.abs(np.asarray(zbar_e_bottom, dtype=np.float64)))
    if nlevels_nod2D is not None:
        out["nlevels_nod2D"] = (
            (nod_dim,), np.asarray(nlevels_nod2D, dtype=np.int32))
    return out


def write_meshdiag(path: Path, convention: str, *,
                   start_index: int | None = None,
                   nodes_units: str | None = None,
                   zero_nod_in_elem2D: bool = False,
                   corrupt_nodes: list[int] | None = None,
                   zbar=None, zbar_n_bottom=None,
                   nlevels=None, ulevels=None, zbar_e_bottom=None,
                   nlevels_nod2D=None) -> Path:
    """convention in {'core2', 'comp', 'ugrid'}.

    start_index : optional UGRID ``face_nodes`` start_index attribute.  When
        set, the connectivity is written with that base (0 -> 0-based ids,
        attr ``start_index=0``; 1 -> the default 1-based ids, attr
        ``start_index=1``).  ``None`` writes the historical 1-based ids with
        NO attribute (loader must default to 1).
    nodes_units : optional ``units`` attr on the packed ``nodes`` variable
        (core2 only); e.g. ``"degrees"`` to write the packed coords already
        in degrees and label them so the loader trusts the units.
    zero_nod_in_elem2D : write an all-zero ``nod_in_elem2D`` (keeping a valid
        ``nod_in_elem2D_num``) to mimic mesh diags that ship the table
        unpopulated — the loader must detect this and rebuild the node->element
        adjacency from the element connectivity.
    corrupt_nodes : optional list of node indices whose ``nod_in_elem2D`` rows
        are zeroed (their ``nod_in_elem2D_num`` stays > 0) to mimic *partial*
        corruption below the 50 % all-empty bar — the loader must still detect
        the count/data inconsistency and rebuild.
    zbar, zbar_n_bottom : optional level interfaces ``(K,)`` and node bottoms
        ``(4,)``, negative down as FESOM2 writes them (default ``ZBAR_NEG``
        and ``ZBAR_N_BOTTOM_NEG``; comp/ugrid store their absolute values,
        as before).
    nlevels, ulevels, zbar_e_bottom : optional per-element ``(2,)`` level
        variables of the FESOM2 mesh diagnostic (each element's number of
        levels, 1-based; its first level; its bed, written negative);
        ``nlevels_nod2D`` the per-node ``(4,)`` count.  ``None`` (default)
        writes none of them, as before.
    """
    zbar_neg = ZBAR_NEG if zbar is None else np.asarray(zbar, dtype=np.float64)
    bottom_neg = (ZBAR_N_BOTTOM_NEG if zbar_n_bottom is None
                  else np.asarray(zbar_n_bottom, dtype=np.float64))
    levels = dict(nlevels=nlevels, ulevels=ulevels,
                  zbar_e_bottom=zbar_e_bottom, nlevels_nod2D=nlevels_nod2D)
    if convention == "core2":
        if nodes_units is not None and "degree" in nodes_units.lower():
            packed = np.stack([LON_DEG, LAT_DEG])      # already degrees
        else:
            packed = np.stack([LON_DEG * DEG2RAD, LAT_DEG * DEG2RAD])
        nodes_var = (("n2", "nod_n"), packed)
        ds = xr.Dataset(
            {
                # packed coords, dims (n2, nod_n) — radians unless labelled
                "nodes": nodes_var,
                "elem": (("n3", "elem_n"), TRI_1BASED.T),          # (3, E)
                "zbar": (("nl",), zbar_neg),                        # negative
                "elem_area": (("elem_n",), ELEM_AREA),
                "zbar_n_bottom": (("nod_n",), bottom_neg),
                "nod_in_elem2D": (("N", "nod_n"), NOD_IN_ELEM_1BASED.T),
                "nod_in_elem2D_num": (("nod_n",), NOD_IN_ELEM_NUM),
                **_level_vars("elem_n", "nod_n", **levels),
            }
        )
        if nodes_units is not None:
            ds["nodes"].attrs["units"] = nodes_units
    elif convention == "comp":
        ds = xr.Dataset(
            {
                "lon": (("nod2",), LON_DEG),
                "lat": (("nod2",), LAT_DEG),
                "elements": (("n3", "elem"), TRI_1BASED.T),         # (3, E)
                "nz": (("nz",), np.abs(zbar_neg)),                  # positive
                "elem_area": (("elem",), ELEM_AREA),
                "zbar_n_bottom": (("nod2",), np.abs(bottom_neg)),
                "nod_in_elem2D": (("N", "nod2"), NOD_IN_ELEM_1BASED.T),
                "nod_in_elem2D_num": (("nod2",), NOD_IN_ELEM_NUM),
                **_level_vars("elem", "nod2", **levels),
            }
        )
    elif convention == "ugrid":
        if start_index is None:
            conn = TRI_1BASED                          # 1-based, no attr
        else:
            conn = (TRI_1BASED - 1 + int(start_index)).astype(np.int32)
        ds = xr.Dataset(
            {
                "lon": (("nod2",), LON_DEG),
                "lat": (("nod2",), LAT_DEG),
                "face_nodes": (("elem", "n3"), conn),               # (E, 3)
                "nz": (("nz",), np.abs(zbar_neg)),
                "elem_area": (("elem",), ELEM_AREA),
                "zbar_n_bottom": (("nod2",), np.abs(bottom_neg)),
                "nod_in_elem2D": (("nod2", "N"), NOD_IN_ELEM_1BASED),
                "nod_in_elem2D_num": (("nod2",), NOD_IN_ELEM_NUM),
                **_level_vars("elem", "nod2", **levels),
            }
        )
        if start_index is not None:
            ds["face_nodes"].attrs["start_index"] = int(start_index)
    else:
        raise ValueError(convention)
    if zero_nod_in_elem2D:
        # unpopulated table (all zeros) — nod_in_elem2D_num stays valid
        dims = ds["nod_in_elem2D"].dims
        ds["nod_in_elem2D"] = (dims, np.zeros_like(ds["nod_in_elem2D"].values))
    elif corrupt_nodes:
        # zero only the listed node rows; nod_in_elem2D_num stays > 0 there
        dims = ds["nod_in_elem2D"].dims
        node_dim = ds["nod_in_elem2D_num"].dims[0]      # convention-agnostic
        node_axis = dims.index(node_dim)
        arr = np.array(ds["nod_in_elem2D"].values)
        for j in corrupt_nodes:
            sl = [slice(None)] * arr.ndim
            sl[node_axis] = j
            arr[tuple(sl)] = 0
        ds["nod_in_elem2D"] = (dims, arr)
    ds.to_netcdf(path)
    return path


def write_meshdiag_zarr(path: Path, *, zero_nod_in_elem2D: bool = False,
                        zbar=None, zbar_n_bottom=None,
                        nlevels=None, ulevels=None, zbar_e_bottom=None,
                        nlevels_nod2D=None) -> Path:
    """Write a ``comp``-convention FESOM2 mesh diag as a Zarr store.

    This is the layout ``extract_fesom_region.py`` produces and
    ``_load_mesh_zarr`` reads.  ``zero_nod_in_elem2D`` writes the table all
    zeros (count stays valid) to exercise the degenerate-table rebuild on the
    Zarr path (the NetCDF path has its own coverage).  The level keywords
    are those of :func:`write_meshdiag`.
    """
    zbar_neg = ZBAR_NEG if zbar is None else np.asarray(zbar, dtype=np.float64)
    bottom_neg = (ZBAR_N_BOTTOM_NEG if zbar_n_bottom is None
                  else np.asarray(zbar_n_bottom, dtype=np.float64))
    ds = xr.Dataset(
        {
            "lon": (("nod2",), LON_DEG),
            "lat": (("nod2",), LAT_DEG),
            "elements": (("n3", "elem"), TRI_1BASED.T),         # (3, E)
            "nz": (("nz",), np.abs(zbar_neg)),                  # positive
            "elem_area": (("elem",), ELEM_AREA),
            "zbar_n_bottom": (("nod2",), np.abs(bottom_neg)),
            "nod_in_elem2D": (("N", "nod2"), NOD_IN_ELEM_1BASED.T),
            "nod_in_elem2D_num": (("nod2",), NOD_IN_ELEM_NUM),
            **_level_vars("elem", "nod2", nlevels=nlevels, ulevels=ulevels,
                          zbar_e_bottom=zbar_e_bottom,
                          nlevels_nod2D=nlevels_nod2D),
        }
    )
    if zero_nod_in_elem2D:
        dims = ds["nod_in_elem2D"].dims
        ds["nod_in_elem2D"] = (dims, np.zeros_like(ds["nod_in_elem2D"].values))
    ds.to_zarr(path, mode="w")
    return path


def write_multiyear(dirpath: Path, var: str, years_lengths: dict[int, int],
                    *, node_layout: bool = True, vertical: str | None = "nz1",
                    units_years: dict[int, int] | None = None,
                    values: Callable[[int, int], np.ndarray] | None = None,
                    continuous_time: bool = False):
    """Write <var>.fesom.<year>.nc files with per-file 'seconds since <year>'.

    years_lengths: {1958: 3, 1959: 3, 1960: 4} -> n time records per file.
    vertical: 'nz1' (mid-layer, 2 levels), 'nz' (interface, 3), or None (2D).
    units_years: optional {filename_year: units_year} override so a file
        named ``...<filename_year>.nc`` decodes its ``seconds since`` against
        a DIFFERENT year — used to fabricate overlapping/duplicate merged
        time axes for the time-axis guard test.
    values: optional ``values(year, k)`` giving level ``k`` of the file of
        ``year`` (anything that broadcasts to ``(ntime, nspace)``; ``k = 0``
        for a 2-D variable), stored as float32.  None writes 0.1 everywhere.
    continuous_time: when True the daily records continue across the files
        (a file's first record is one day after the previous file's last),
        all files counting from the first year's 1 January; default: each
        file counts its records from its own year's 1 January.
    """
    nod = 4
    space_dim = "nod2" if node_layout else "elem"
    nspace = 4 if node_layout else 2
    units_years = units_years or {}
    first_year = min(years_lengths) if years_lengths else None
    day0 = 0          # records before this file (continuous_time only)
    items = (sorted(years_lengths.items()) if continuous_time
             else years_lengths.items())
    for year, ntime in items:
        # one record per day, timestamp at 23:15 like the real files
        secs = ((np.arange(ntime) + day0) * 86400.0
                + 83700.0).astype("float64")
        if continuous_time:
            day0 += ntime
        coords = {"time": ("time", secs)}
        if vertical == "nz1":
            shape = (ntime, 2, nspace); dims = ("time", "nz1", space_dim)
        elif vertical == "nz":
            shape = (ntime, 3, nspace); dims = ("time", "nz", space_dim)
        else:
            shape = (ntime, nspace); dims = ("time", space_dim)
        data = np.full(shape, 0.1, dtype=np.float32)
        if values is not None:
            if len(shape) == 3:
                for k in range(shape[1]):
                    data[:, k, :] = np.broadcast_to(
                        np.asarray(values(year, k), dtype=np.float32),
                        (ntime, nspace))
            else:
                data[:] = np.broadcast_to(
                    np.asarray(values(year, 0), dtype=np.float32),
                    (ntime, nspace))
        ds = xr.Dataset({var: (dims, data)}, coords=coords)
        units_year = units_years.get(
            year, first_year if continuous_time else year)
        ds["time"].attrs["units"] = f"seconds since {units_year}-01-01 0:0:0"
        ds["time"].attrs["calendar"] = "standard"
        ds.to_netcdf(dirpath / f"{var}.fesom.{year}.nc")
