"""Small, dataset-agnostic helpers for publication figures from KELP trajectory products.

Each helper is one recipe of the gallery catalog (`gallery/README.md`):

- `anomaly_vs_mean` + `symmetric_log_norm` — where similar periods (decades, seasons,
  runs) differ: each member minus the mean over the members, on one symmetric-log scale.
- `box_fraction` — the share of gridded particle counts inside a lon/lat box, for the CSV
  that a caption cites instead of a number read off the map.
- `lightness_spaced_colors` — N line colours at evenly spaced CIELAB lightness of a
  sequential colour map (distinct in greyscale and for colour-blind readers).
- `preproject_segments` — trajectory vertices projected ONCE into the axes projection
  as a LineCollection segment array (cartopy otherwise re-projects every segment on
  every draw: ~20x slower for static maps and videos alike).
- `set_view_from_points` — an inset's view from the projected bounding box of the
  points it must show (no lon/lat-rectangle bulge towards the pole).
- `contact_sheet` — 4-6 variants of one figure (palettes, layouts) on one labelled PNG,
  so the owner picks once instead of iterating one variant at a time.
- `palette_contrast` — WCAG contrast of every sampled palette bin against the map
  background, printed before saving (a bin below ~2 : 1 vanishes into the ocean).
- `old_vs_new` — the superseded figure and the new one side by side, for review only.
- `write_caption` — the `CAPTION_<name>.md` a render script writes next to its figure.

Inputs are plain NumPy arrays and matplotlib/cartopy objects; nothing here reads a
KELP store. Run `python gallery/figure_helpers.py` for the self-check (synthetic data).
"""
from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------------------
# anomalies of similar fields
# ---------------------------------------------------------------------------------------
def anomaly_vs_mean(fields: Sequence[np.ndarray]) -> Tuple[list, np.ndarray]:
    """Each field minus the mean over all fields.

    Parameters
    ----------
    fields : sequence of equally shaped arrays (e.g. one presence density per decade,
        all normalised the same way — per released particle, per area).

    Returns
    -------
    anomalies : list of arrays, ``fields[i] - mean``
    mean : the ensemble mean (NaN-aware: a cell NaN in every member stays NaN)
    """
    stack = np.stack([np.asarray(f, dtype=float) for f in fields])
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(stack, axis=0) if np.isnan(stack).any() else stack.mean(axis=0)
    return [s - mean for s in stack], mean


def symmetric_log_norm(anomalies: Iterable[np.ndarray], linthresh: float,
                       quantile: float = 99.5, linscale: float = 0.5):
    """One symmetric-log norm for all panels: +-L with L the `quantile` of |anomaly|.

    `linthresh` = the smallest density difference worth seeing (linear inside it); one
    norm for every panel of a montage — never a per-panel scale.
    """
    from matplotlib.colors import SymLogNorm
    vals = np.concatenate([np.abs(np.asarray(a, float)).ravel() for a in anomalies])
    vals = vals[np.isfinite(vals) & (vals > 0)]
    L = float(np.percentile(vals, quantile)) if vals.size else 1.0
    L = max(L, 10.0 * linthresh)
    return SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=-L, vmax=L, base=10)


def box_fraction(density: np.ndarray, lon_centres: np.ndarray, lat_centres: np.ndarray,
                 box: Tuple[float, float, float, float]) -> float:
    """Share of the particles of one period inside the box ``(W, E, S, N)``.

    `density` has shape ``(len(lat_centres), len(lon_centres))`` and holds particle COUNTS
    per cell of one period — never an anomaly, never counts ÷ cell area (a sum of per-area
    densities over unequal cells is not a share of particles); a box with W > E wraps the
    dateline. Write one row per panel and box to a CSV next to the figure; the caption
    formats its numbers from those rows.
    """
    d = np.nan_to_num(np.asarray(density, float))
    if (d < 0).any():
        raise ValueError("box_fraction takes particle counts, not an anomaly (it sums to about zero)")
    W, E, S, N = box
    lon = np.asarray(lon_centres, float)
    lat = np.asarray(lat_centres, float)
    in_lon = (lon >= W) & (lon <= E) if W <= E else (lon >= W) | (lon <= E)
    in_lat = (lat >= S) & (lat <= N)
    tot = d.sum()
    return float(d[np.ix_(in_lat, in_lon)].sum() / tot) if tot > 0 else float("nan")


# ---------------------------------------------------------------------------------------
# colours
# ---------------------------------------------------------------------------------------
def cie_lightness(rgb: np.ndarray) -> np.ndarray:
    """CIELAB L* (0-100) of sRGB colours in [0, 1], shape (..., 3) or (..., 4)."""
    c = np.asarray(rgb, float)[..., :3]
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    Y = lin @ np.array([0.2126, 0.7152, 0.0722])
    d = 6.0 / 29.0
    f = np.where(Y > d ** 3, np.cbrt(Y), Y / (3 * d * d) + 4.0 / 29.0)
    return 116.0 * f - 16.0


def relative_luminance(rgb: np.ndarray) -> np.ndarray:
    """WCAG relative luminance of sRGB colours in [0, 1], shape (..., 3) or (..., 4)."""
    c = np.asarray(rgb, float)[..., :3]
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return lin @ np.array([0.2126, 0.7152, 0.0722])


def palette_contrast(cmap, background, n: int = 13, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """WCAG contrast ratio (1-21) of `n` bins of `cmap` sampled on [lo, hi] against `background`.

    Print it before saving and name the bins below ≥ 2 : 1 against the ocean (≥ 2.5 : 1
    for thin lines); assert on it only for a palette the script chose itself —
    an owner-named palette is drawn as named and its weak bins reported. On a filled map the
    anomaly zero centre and the lowest density bins may fade into the ocean.
    """
    import matplotlib
    from matplotlib.colors import to_rgb
    cm = matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap
    Lc = relative_luminance(cm(np.linspace(lo, hi, n)))
    Lb = float(relative_luminance(np.array(to_rgb(background))))
    hi_l, lo_l = np.maximum(Lc, Lb), np.minimum(Lc, Lb)
    return (hi_l + 0.05) / (lo_l + 0.05)


def lightness_spaced_colors(cmap, n: int, l_from: float, l_to: float,
                            resolution: int = 1001) -> list:
    """`n` colours of a monotonic-lightness colour map at evenly spaced CIELAB L*.

    Use a perceptual sequential map (``cmocean.cm.deep``, ``cmcrameri.cm.batlow``); pass
    ``l_from > l_to`` for light-to-dark order. Lines drawn in these colours differ in
    lightness, not only in hue, so they stay distinguishable in greyscale.
    """
    import matplotlib
    cm = matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap
    xs = np.linspace(0.0, 1.0, resolution)
    L = cie_lightness(cm(xs))
    return [cm(xs[int(np.argmin(np.abs(L - t)))]) for t in np.linspace(l_from, l_to, n)]


# ---------------------------------------------------------------------------------------
# projection and views
# ---------------------------------------------------------------------------------------
def preproject_segments(projection, lon: np.ndarray, lat: np.ndarray,
                        values: Optional[np.ndarray] = None,
                        max_jump: Optional[float] = None):
    """Consecutive-vertex segments of tracks, projected ONCE into `projection`.

    Parameters
    ----------
    projection : the axes' cartopy CRS (``ax.projection``)
    lon, lat : arrays ``(n_tracks, n_records)``; NaN marks a missing vertex (not yet
        released, dead)
    values : optional ``(n_tracks, n_records)`` colour values; each segment takes the
        value of its first vertex
    max_jump : drop segments longer than this in projected units (a track crossing the
        map's cut, e.g. the dateline on a cylindrical projection); None keeps all

    Returns
    -------
    segs : ``(M, 2, 2)`` array for ``LineCollection(segs, ...)`` — draw it WITHOUT
        ``transform=``, the coordinates are already projected
    vals : ``(M,)`` segment values, or None
    """
    import cartopy.crs as ccrs
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    P, T = lon.shape
    xy = projection.transform_points(ccrs.PlateCarree(), lon.ravel(), lat.ravel())[:, :2]
    xy = xy.reshape(P, T, 2)
    a, b = xy[:, :-1], xy[:, 1:]
    ok = np.isfinite(a).all(axis=2) & np.isfinite(b).all(axis=2)
    if max_jump is not None:
        ok &= np.hypot(*(b - a).transpose(2, 0, 1)) <= max_jump
    segs = np.stack([a[ok], b[ok]], axis=1)
    vals = None if values is None else np.asarray(values, float)[:, :-1][ok]
    return segs, vals


def set_view_from_points(ax, points_lonlat: np.ndarray, pad: float = 0.0) -> None:
    """Set a cartopy axes' view to the projected bounding box of (lon, lat) points.

    A lon/lat rectangle drawn in a polar projection bulges towards the pole; the
    bounding box of the points that must be visible, in the axes' own projection, does
    not. `pad` widens the box by that fraction on each side.
    """
    import cartopy.crs as ccrs
    p = np.asarray(points_lonlat, float)
    xy = ax.projection.transform_points(ccrs.PlateCarree(), p[:, 0], p[:, 1])[:, :2]
    x0, x1 = xy[:, 0].min(), xy[:, 0].max()
    y0, y1 = xy[:, 1].min(), xy[:, 1].max()
    dx, dy = pad * (x1 - x0), pad * (y1 - y0)
    ax.set_xlim(x0 - dx, x1 + dx)
    ax.set_ylim(y0 - dy, y1 + dy)


# ---------------------------------------------------------------------------------------
# choosing between variants
# ---------------------------------------------------------------------------------------
def contact_sheet(paths: Sequence[str], out_path: str, labels: Optional[Sequence[str]] = None,
                  ncols: int = 3, tile_width: int = 900) -> str:
    """Tile PNG variants of one figure into one labelled sheet (PIL); returns `out_path`."""
    from PIL import Image, ImageDraw
    ims = [Image.open(p).convert("RGB") for p in paths]
    labels = list(labels) if labels is not None else [f"({i + 1})" for i in range(len(ims))]
    tiles = [im.resize((tile_width, int(im.height * tile_width / im.width))) for im in ims]
    th = max(t.height for t in tiles) + 40
    nrows = -(-len(tiles) // ncols)
    sheet = Image.new("RGB", (ncols * tile_width, nrows * th), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (t, lab) in enumerate(zip(tiles, labels)):
        x, y = (i % ncols) * tile_width, (i // ncols) * th
        sheet.paste(t, (x, y + 40))
        draw.text((x + 10, y + 10), str(lab), fill="black")
    sheet.save(out_path)
    return out_path


def old_vs_new(old_figure: str, new_figure: str, out_png: str, dpi: int = 100) -> str:
    """The superseded figure and the new one side by side, for REVIEW only (never the final figure).

    Either figure may be a PDF (rasterised with poppler's `pdftoppm`) or a PNG; both are
    already rendered files — this never re-renders anything.
    """
    import os
    import shutil
    import subprocess

    def _png(path: str, tag: str) -> str:
        if not path.lower().endswith(".pdf"):
            return path
        if shutil.which("pdftoppm") is None:
            raise RuntimeError("pdftoppm (poppler) not found: pass PNGs of the figures instead")
        stem = os.path.splitext(out_png)[0] + "_" + tag
        subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-singlefile", path, stem], check=True)
        return stem + ".png"

    return contact_sheet([_png(old_figure, "old"), _png(new_figure, "new")], out_png,
                         labels=["old: " + os.path.basename(old_figure), "new: " + os.path.basename(new_figure)], ncols=2)


def write_caption(out_md: str, *, replaces: str, command: str, data: str, commit: str, caption: str,
                  engine_commit: Optional[str] = None) -> str:
    """Write the `CAPTION_<name>.md` that goes next to a figure (called by the render script).

    Two revisions: `engine_commit` = the KELP engine that PRODUCED the data (the run store's
    `kelp_version` attribute or the ledger, never the clone that renders; omitted → "NOT GIVEN"),
    `commit` = the render script's own commit, "-dirty" appended when the script is modified or
    untracked (a dirty stamp cannot regenerate the figure).
    """
    text = (f"# {os.path.basename(out_md)}\n\n"
            f"Replaces: {replaces}\n\n"
            f"Regenerate:\n\n    {command}\n\n"
            f"Data: {data}; engine commit {engine_commit or 'NOT GIVEN'}; render-script commit {commit}\n\n"
            f"## Caption\n\n{caption}\n")
    with open(out_md, "w") as f:
        f.write(text)
    return out_md


# ---------------------------------------------------------------------------------------
# self-check on synthetic data
# ---------------------------------------------------------------------------------------
def _selfcheck(workdir: str = ".tmp/figure_helpers_check") -> None:
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    rng = np.random.default_rng(0)
    f = [rng.random((20, 30)) for _ in range(4)]
    an, mean = anomaly_vs_mean(f)
    assert np.allclose(sum(an), 0.0), "anomalies must sum to zero over the members"
    norm = symmetric_log_norm(an, linthresh=1e-3)
    assert norm.vmin == -norm.vmax and norm.vmax > 0

    lon_c, lat_c = np.linspace(-179.5, 179.5, 360), np.linspace(-89.5, 89.5, 180)
    dens = np.zeros((180, 360)); dens[160, 9] = 3.0; dens[160, 350] = 1.0   # lon -170.5 and 170.5, lat 70.5
    assert abs(box_fraction(dens, lon_c, lat_c, (170, -170, 60, 80)) - 1.0) < 1e-12   # dateline box
    assert abs(box_fraction(dens, lon_c, lat_c, (-175, -165, 60, 80)) - 0.75) < 1e-12
    try:
        box_fraction(an[0], np.arange(30.0), np.arange(20.0), (0, 10, 0, 10))
        raise AssertionError("box_fraction must refuse an anomaly")
    except ValueError:
        pass

    c = palette_contrast("Greys", "#000000", n=3)
    assert abs(c[0] - 21.0) < 0.1 and c.min() >= 1.0 - 1e-12   # white vs black = 21 : 1

    cols = lightness_spaced_colors("viridis", 5, 30, 85)
    L = cie_lightness(np.array(cols))
    assert np.all(np.diff(L) > 0) and np.ptp(np.diff(L)) < 1.5, L   # evenly spaced lightness

    proj = ccrs.NorthPolarStereo()
    lon = np.array([[170.0, 175.0, -178.0, np.nan], [0.0, 1.0, 2.0, 3.0]])
    lat = np.array([[80.0, 80.5, 81.0, 81.0], [75.0, 75.5, 76.0, 76.5]])
    segs, vals = preproject_segments(proj, lon, lat, values=lat)
    assert segs.shape == (5, 2, 2) and vals.shape == (5,)   # the NaN vertex drops one segment
    segs_pc, _ = preproject_segments(ccrs.PlateCarree(), lon, lat, max_jump=20.0)
    assert segs_pc.shape[0] == 4                             # the dateline jump is cut

    os.makedirs(workdir, exist_ok=True)
    paths = []
    for k, cm in enumerate(("viridis", "magma")):
        fig, ax = plt.subplots(subplot_kw=dict(projection=proj), figsize=(3, 3))
        set_view_from_points(ax, np.array([[-60, 60], [20, 60], [0, 85]]), pad=0.05)
        from matplotlib.collections import LineCollection
        lc = LineCollection(segs, cmap=cm); lc.set_array(vals); ax.add_collection(lc)
        p = os.path.join(workdir, f"variant_{k}.png"); fig.savefig(p, dpi=60); plt.close(fig)
        paths.append(p)
    out = contact_sheet(paths, os.path.join(workdir, "sheet.png"), labels=["viridis", "magma"], ncols=2, tile_width=200)
    assert os.path.getsize(out) > 0
    cmp_png = old_vs_new(paths[0], paths[1], os.path.join(workdir, "old_vs_new.png"))
    assert os.path.getsize(cmp_png) > 0
    import shutil
    if shutil.which("pdftoppm"):
        fig, ax = plt.subplots(figsize=(2, 2)); ax.plot([0, 1], [0, 1])
        pdf = os.path.join(workdir, "old.pdf"); fig.savefig(pdf); plt.close(fig)
        assert os.path.getsize(old_vs_new(pdf, paths[1], os.path.join(workdir, "old_vs_new_pdf.png"))) > 0
    cap = write_caption(os.path.join(workdir, "CAPTION_test.md"), replaces="old.pdf", command="python render.py --cache-dir X",
                        data="cache X", commit="abc1234", caption="Test caption.")
    assert "Regenerate" in open(cap).read() and "engine commit NOT GIVEN" in open(cap).read()
    cap = write_caption(os.path.join(workdir, "CAPTION_test.md"), replaces="old.pdf", command="python render.py --cache-dir X",
                        data="cache X", commit="abc1234-dirty", caption="Test caption.", engine_commit="0.9.0+4b8b354")
    assert "engine commit 0.9.0+4b8b354; render-script commit abc1234-dirty" in open(cap).read()
    print("figure_helpers self-check: OK")


if __name__ == "__main__":
    _selfcheck()
