# Spaghetti maps with a zoom inset

Read when a track map needs a zoom panel (a gateway, a shelf). Pre-projection (the skill's speed
bullet) applies to the main map and the inset alike.

- **Spaghetti with a zoom inset** (a gateway, a shelf): pre-project the tracks once
  (the skill's speed bullet), draw the inset from the same pre-projected data at a denser subsample, set
  the inset's view from the projected bounding box of the points it must show
  (`gallery/figure_helpers.py::set_view_from_points`), not from a lon/lat box:
  ```python
  segs_dense, vals_dense = preproject_segments(ax.projection, lon[::stride_inset], lat[::stride_inset], values=dz[::stride_inset])
  inset = ax.inset_axes([0.02, 0.05, 0.45, 0.45], projection=ax.projection)
  lc = LineCollection(segs_dense, cmap=cmap, norm=norm); lc.set_array(vals_dense); inset.add_collection(lc)
  g_lon, g_lat = np.meshgrid(np.linspace(W, E, 20), np.linspace(S, N, 20))   # the region the zoom must show, sampled
  pts_lonlat = np.c_[g_lon.ravel(), g_lat.ravel()]                           # never all track vertices
  set_view_from_points(inset, pts_lonlat, pad=0.05)   # never inset.set_extent([...lon/lat box...])
  ```
