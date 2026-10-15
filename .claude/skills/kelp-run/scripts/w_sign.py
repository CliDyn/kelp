# w_sign.py — python w_sign.py config.yaml : the FILE's mean w, 2°S–2°N, 150–100°W, level nearest 50 m, records 1–30
import dataclasses, inspect, sys
import numpy as np
from kelp.core.config import load_config
from kelp.core.rotation import euler_matrix, unrotate_lonlat
from kelp.readers import get_reader
cfg = load_config(sys.argv[1])
mode = dataclasses.replace(cfg.mode, w_positive="up")      # read AS IF positive up: file w = −(KELP's positive-down w)
rd = get_reader(cfg.mesh.type)
kw = {"mode_cfg": mode} if "mode_cfg" in inspect.signature(rd.load_mesh).parameters else {}
mesh = rd.load_mesh(cfg.mesh, **kw)
extra = {"eta_path": cfg.data.ncinput_eta} if cfg.data.ncinput_eta else {}
ds = rd.open_dataset(cfg.data.ncinput, v_path=cfg.data.ncinput_v or None, w_path=cfg.data.ncinput_w or None, mode_cfg=mode, **extra)
lon, lat = mesh.nodes_lon, mesh.nodes_lat
if mesh.rotation_angles is not None:                       # rotated mesh: back to geographic
    lon, lat = unrotate_lonlat(lon, lat, euler_matrix(*mesh.rotation_angles))
box = (np.abs(lat) <= 2) & (lon >= -150) & (lon <= -100)     # equatorial upwelling; a regional store needs its own region of known sign
assert box.sum() >= 10, "store does not cover 2°S–2°N, 150–100°W: run this on a config that points at the global SOURCE files (before any crop), or pick a region of known up- or downwelling inside the store"
k = int(np.argmin(np.abs(mesh.sigma - 50.0)))
n = ds.sizes["time"]; recs = range(min(1, n - 1), min(31, n))  # records 1–30 (record 0 may be a cold start), fewer on a short store
w = np.mean([-rd.read_velocities(ds, t, mesh, mode).w[k, box].mean() for t in recs])
print(f"mean w (file) = {w:+.2e} m/s at {mesh.sigma[k]:.0f} m over {int(box.sum())} nodes -> mode.w_positive: {'up' if w > 0 else 'down'}")
