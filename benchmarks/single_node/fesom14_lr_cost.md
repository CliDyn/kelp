# fesom14_lr_cost — FESOM 1.4 output on the published LR mesh

What the FESOM 1.4 reader (`mesh.type: fesom14`, `docs/dev/readers.md`) costs
on real data: the published CMIP6 monthly means of AWI-CM-1-1-LR (FESOM 1.4,
126,859 nodes, 244,660 triangles recovered from the cell polygons, 46 levels;
licence CC BY-SA 4.0), level-wise `(time, depth, ncells)` files, one per
variable and decade. The I/O rows are measured; the step cost is pending.

## I/O (measured 2026-10-19)

Configuration: an HPC compute node (2× AMD EPYC 7763), 16 CPUs allocated,
`NUMBA_NUM_THREADS=8`; the files read where they lie, on the archive's shared
parallel file system; one decade (120 records) of `uo`, `vo`, `wo`, `zos` (no
sampled variable); float32 on disk, chunks `[1, 16, 42287]` (one record, 16
levels, a third of the nodes); the reader code of this release.

| operation | time | memory | note |
|---|---|---|---|
| `load_mesh` from the cell polygons | 7.1 s | — | triangle recovery, wet mask from record 0 of `uo`, `deptho` check, search structures |
| `open_dataset` | 0.95 s (0.39 s repeated) | — | headers, per-file time decoding and resolution from `time_bnds`, alignment |
| `prepare` | 0.41 s | ~11 bytes per slot of one 3-D variable at the peak | record 0 of each variable against the static mask; runs at every entry |
| a 12-record window, `ds.isel(time=slice(24, 36)).load()`, cold | 11.0 s (≈ 0.9 s per record) | 0.85 GB | what `numerics.prefetch_size: 12` reads at once |
| the same for records 0–11, partly in the file-system cache | 3.8 s | 0.85 GB | |
| `read_velocities` ×12 straight from the files | 11.3 s (≈ 0.94 s per record) | — | the file read dominates |
| `read_velocities` ×12 on a loaded window | 0.87 s (≈ 0.07 s per record) | — | the mask check on every level, zeros below the bed, the sign of w |

One record of u, v and w is 3 × 46 × 126,859 float32 values (70 MB). With
monthly records and `dt = 6 h` a record serves about 120 steps, so the cold read
of ~0.9 s per record is ≈ 8 ms per step when spread over them (arithmetic from
the rows above, not a separate measurement).

## Step cost (pending)

Not measured yet: the fused 3-D RK2 step (barycentric u/v and w, linear in time,
u/v linear in depth on the levels) with 1e5 particles on LR. The command, for an
otherwise idle compute node (a compute job: it needs the run owner's yes):

```bash
export NUMBA_NUM_THREADS=<physical cores> OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
LR=path/to/fesom14_lr      # uo, vo, wo, zos, deptho of the published LR output
python - "$LR" <<'EOF'
import sys, numpy as np, yaml
from kelp.core.config import MeshConfig
from kelp.readers.fesom14 import Fesom14Reader
m = Fesom14Reader.load_mesh(MeshConfig(type="fesom14", meshpath=sys.argv[1]))
rng = np.random.default_rng(0)
shallow = m.depth[m.elem_nodes].min(axis=1)
ok = np.flatnonzero((shallow >= 50.0) & (np.abs(m.elem_center_lat) <= 80.0))
e = rng.choice(ok, 100_000, replace=False)
z = rng.uniform(0.0, 0.9, e.size) * np.minimum(shallow[e], 3000.0)
np.savetxt("seeds_lr_1e5.dat", np.c_[m.elem_center_lon[e], m.elem_center_lat[e], z])
cfg = yaml.safe_load(open("examples/config_fesom14_levelwise.yaml"))
cfg["mode"]["sample_vars"] = []
cfg["numerics"]["output_freq"] = 40
cfg["mesh"]["meshpath"] = cfg["data"]["ncinput"] = sys.argv[1]
cfg["data"].update(inicloud="seeds_lr_1e5.dat", output_path="bench_lr.zarr")
cfg["time"]["duration_days"] = 10          # 40 steps of 6 h
yaml.safe_dump(cfg, open("bench_lr.yaml", "w"))
EOF
python -c "import numba, os; print(numba.get_num_threads(), len(os.sched_getaffinity(0)))"
/usr/bin/time -v kelp run bench_lr.yaml -v > bench_lr.log 2>&1
```

The row: the median of `step=` over steps 5–40 of the log (ms per step, the
tracking kernel; `io=` separately), thread-µs per particle = step × threads /
1e5, and the peak memory from `/usr/bin/time`.

| machine | CPU | threads | particles | step [ms] | io [ms, median] | thread-µs/particle | peak RSS | measured |
|---|---|---|---|---|---|---|---|---|
| — | — | — | 100,000 | — | — | — | — | pending |
