# record0.py — python record0.py config.yaml : zero fraction of the surface elevation at records 0 / mid / end
import inspect, sys
import numpy as np
from kelp.core.config import load_config
from kelp.readers import get_reader
cfg = load_config(sys.argv[1]); rd = get_reader(cfg.mesh.type)
mesh = rd.load_mesh(cfg.mesh, **({"mode_cfg": cfg.mode} if "mode_cfg" in inspect.signature(rd.load_mesh).parameters else {}))
ds = rd.open_dataset(cfg.data.ncinput, v_path=cfg.data.ncinput_v or None, w_path=cfg.data.ncinput_w or None, mode_cfg=cfg.mode,
                     **({"eta_path": cfg.data.ncinput_eta} if cfg.data.ncinput_eta else {}))
n = ds.sizes["time"]
for t in (0, n // 2, n - 1):
    print(f"record {t}: ssh zero fraction {np.mean(rd.read_velocities(ds, t, mesh, cfg.mode).eta == 0.0):.3f}")   # ~1.0 at record 0 = cold start
