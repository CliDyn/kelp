"""Synthetic tripolar-like grids for the C-grid topology tests: all four NEMO
raster layouts ({T, F}-pivot north fold) x ({with, without} the halo).

An elliptic map (x = A cosh(mu) cos(nu), y = A sinh(mu) sin(nu), degrees at the
equator) is evaluated for every point kind on the doubled lattice with the SAME
fold centre ``cgrid_canonical`` uses; since pos(-mu, -nu) = pos(mu, nu) every
fold and seam identity holds for either pivot.  The raw raster of a layout is
the canonical halo form, or that minus the halo columns and the halo row.
"""
from types import SimpleNamespace

import numpy as np

from kelp.readers.structured import CgridLayout, _KIND_OFFSET, _centre, cgrid_to_canonical

NC, NY, A, DMU = 24, 10, 1.0, 0.15          # interior columns, canonical rows, focal half-width, d(mu)
KINDS = "TUVF"


def canon_layout(fold):
    return CgridLayout((NY, NC + 2), True, True, True, fold)


def raw_layout(fold, halo):
    return canon_layout(fold) if halo else CgridLayout((NY - 1, NC), False, False, True, fold)


def strip(a, halo):
    """Raw raster of a canonical-form array."""
    return a if halo else a[..., :NY - 1, 1:-1]


def orca_grid(fold, halo):
    """``xy[kind] = (lon, lat)`` on the raw raster, bit-exact copies."""
    lay = canon_layout(fold)
    jj, ii = np.mgrid[0:NY, 0:NC + 2]
    canon = {}
    for k in KINDS:
        oy, ox = _KIND_OFFSET[k]
        Y2, X2 = _centre(lay, k)
        mu = (Y2 - (2 * jj + oy)) * 0.5 * DMU
        nu = np.pi * ((2 * ii + ox) - X2) / NC
        x, y = A * np.cosh(mu) * np.cos(nu), A * np.sinh(mu) * np.sin(nu)
        canon[k] = (cgrid_to_canonical(x, lay, k), cgrid_to_canonical(y, lay, k))
    return SimpleNamespace(
        raw=raw_layout(fold, halo), canon=lay, halo=halo, fold=fold,
        xy_c=canon, xy={k: (strip(a, halo), strip(b, halo)) for k, (a, b) in canon.items()})
