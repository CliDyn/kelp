# Probes and timing legs on a running or finished campaign

Read before any benchmark or timing leg on a campaign that already has restarts.

The honest, cheap benchmark workload is a **resumed real mid-run state**, not
a re-run from the seeds: resume a numbered production restart for 30–1826
steps, one leg per question, and delete the
segments afterwards. Rules that each cost a leg once:

- The probe driver writes its end restart to `<segment_dir>/restart_out/`;
  the production `restarts/` dir stays read-only (a driver that wrote back
  made every later leg resume 30 steps further).
- Multi-process legs: `srun --wait=0`, staggered starts, per-process logs
  named with the job id, `python -u`.
- Compare the probe's end state **bitwise** against the stored production
  segment/restart at the same step: a layout or engine change is a free
  speed-up only after `0 statuses, 0 anchors, 0 positions differ`; when the
  change is meant to differ (`exact_nearest`, a smaller hash cell), report the
  difference as counts (statuses / anchors / positions, max displacement) and
  as ensemble statistics — individual paths diverge by chaos.
- Give every concurrently compiling process its own `NUMBA_CACHE_DIR`, or
  pre-warm the cache with one short run first (concurrent compiles into one
  cache crash with `NRT_adapt_ndarray_to_python: 'descr' is NULL`).
- Measure the phase split (`io= step= rec=` in `-v`) before optimising: one
  campaign's estimate "output-dominated" was wrong (measured tracking 65 %,
  output 33 %, input 2 %).
- The reported "minutes per model year" comes from such a leg
  (`steady step × steps per year`, worst shard, at a stated alive count and
  layout), not from a full re-run; quote allocated AND active
  particle-step rates for pulsed runs with outflow.
- An ensemble that differs only in RELEASE TIME is ONE run with `time.n_pulses` +
  `pulse_interval` (one pass over the forcing), never `time.num_runs` (one pass
  per realisation: 78 × 1.4 TB ≈ 59 TB of re-reads vs 0.9 TB — 65×). Analysis
  then masks `NOT_RELEASED` rows (NaN positions).
- A grid/method COMPARISON freezes one numerics protocol on both arms
  (same interpolator, `elem_search`, frame, diffusion and beaching OFF,
  same output cadence) — otherwise the separation measures the RNG.
- Instrument the campaign for the cost table from the first run; at
  campaign close write a reproducibility capsule (directory map, metric
  convention, contamination rule, job ledger) while the context is fresh.
  Every reported number traces to a dated campaign directory whose README
  names the question, setup, job id, log and the report section it feeds.
