# Huge releases and showcases without output

Read before sizing a job of ≥ 10⁸ particles per process or a speed/memory showcase that writes no
trajectories (step 5 of the skill). The measured memory line itself is in `docs/performance.md` §3.

**Huge releases (≥ 10⁸ particles per process):** memory = a base + bytes per
particle — fit both from two one-node legs and check the fit on a third (the one
measured line, its configuration and its read floor are in `docs/performance.md` §3;
they hold for that configuration only). A small probe's peak is the READ, not the
particles — do not extrapolate from it. Each fit point must be PARTICLE-dominated:
its particle term (B_guess × n, B_guess = the measured line's bytes per particle) at
least the small probe's read peak — a
read-dominated point gives a slope ≈ 0 (the measured line used 10⁹ and 4 × 10⁹ per
process). Fill the table and both lines with YOUR numbers in the PLAN:
```
| fit leg | particles/process            | MaxRSS (sacct) |
| A       | ≥ read_peak / B_guess        | TBD |
| B       | ≥ 2 × A                      | TBD |
| check   | the target per-process count | TBD |
per_process = total / (nodes × processes_per_node)
peak_GiB    = max(read_peak_GiB, base_GiB + B_per_particle × per_process / 2**30)
```
B_guess only sizes the legs; never compute bytes per particle from a small probe (its
peak minus the base over n is the read, not the particles). When B_guess × the target
per-process count stays below the read peak, the target is read-dominated: the
one-node leg at the target count is the measurement, and the fit legs are skipped. When the
configuration differs from the measured line's (integrator, mesh), the one-node target leg runs
first and the fit follows only if its MaxRSS exceeds the read peak. B_guess = 181 B (the measured
line of docs/performance.md §3, zoom 11 RK2 — for sizing legs only). Without the shared cache every
process on a node pays the WHOLE read peak: processes per node × read peak must fit the node
before any particle is counted.
Even with a predicted fit, submit the smaller size first and read its MaxRSS before the
larger job; the larger job names the big-memory node class (`--constraint=512G`, …).
Tracking and memory WITHOUT writing anything — `writer=None` alone still keeps every
record in memory, so `output_freq` must exceed `n_steps`:
```python
num = dataclasses.replace(cfg.numerics, output_freq=n_steps + 1, checkpoint_freq=0)
lon, lat, z = (np.empty(n) for _ in range(3))           # n = particles in THIS process
for s in range(0, n, 20_000_000):                       # tile base seeds in chunks, never a 1e9-line file
    e = min(n, s + 20_000_000); idx = (np.arange(s, e) + k * n) % len(base)
    lon[s:e], lat[s:e], z[s:e] = base[idx, 0], base[idx, 1], base[idx, 2]   # geographic, as in the seed file
# mesh, ds: load and set up exactly as leg_driver.py in campaign-templates.md (load_mesh(..., mode_cfg=),
# set_exact_nearest, the cell rule) — a showcase driver skips none of them
from kelp.tracking import _maybe_rotate_seeds                  # geographic seeds -> the mesh frame
run_single_realization(mesh, ds, reader, _maybe_rotate_seeds(make_state(lon, lat, z), mesh), cfg.mode, num, n_steps=n_steps,
                       start_time_idx=cfg.time.start_time_idx, writer=None, verbose=True)   # + n_time_total, file_times, file_dt, shared_cache as in leg_driver.py
```
The driver computes the steady step from its own verbose lines and prints it — never
a literal example value (`io=`/`step=` are milliseconds in the log):
```python
import contextlib, io, re, sys
class Tee(io.StringIO):                                  # live log AND a copy to parse
    def write(self, x): sys.__stdout__.write(x); sys.__stdout__.flush(); return super().write(x)
log = Tee()
with contextlib.redirect_stdout(log):
    run_single_realization(...)                          # verbose=True, as above
pairs = [(float(a) / 1e3, float(b) / 1e3) for a, b in re.findall(r"\(io=([\d.]+) step=([\d.]+)", log.getvalue())]
s = sorted(st for i, (io_s, st) in enumerate(pairs) if i >= 1 and io_s < 0.5)
print(f"STEADY median={s[len(s) // 2]:.3f} s" if s else "STEADY median=NA (no step >= 2 with io < 0.5 s)")
```
The job script is the preamble + multi-shard `srun` block of
[campaign-templates.md](campaign-templates.md), copied (exported variables, the per-task
`NUMBA_CACHE_DIR` and rank inside the single quotes, `--wait=0`). It ends with the extractor over all `p*_$SLURM_JOB_ID.log`, `tail -1` =
the slowest process; never a mean. `jobs/`, in submission order, each script self-contained
(no input created by a later one):
```
01_fitA.sh, 02_fitB.sh               # skipped when the target is read-dominated
03_check_target_per_process.sh       # ONE node, the target per-process count
04_target.sh                         # all nodes; only after 03's sacct MaxRSS is read
```
`showcase.py` takes the per-process count and the rank as arguments, so every leg passes its own
count and tiles from the one base seed file. Each job ends with
`grep -h "STEADY median=" logs/<leg>_p*_$SLURM_JOB_ID.log | sort -t= -k2 -g | tail -1`
(the slowest process, seconds).
