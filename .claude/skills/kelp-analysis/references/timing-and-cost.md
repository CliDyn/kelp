# Timing numbers and cost comparisons from run logs

Read when quoting a range or a speed number, splitting a phase per event, or answering
"is re-running cheaper than storing?". The steady-step rule is in the skill itself.

- **Every range, minimum and "at every point" statement is computed from the table by a
  one-line script immediately before it is written** — a range summarised from memory
  ("1.6–2.1× at every point") was wrong (the table said 1.25–2.5×).
- **Per-event cost = the MEDIAN over the steps that carry the event** (snapshot-read
  steps, record steps) ÷ particles; never a per-step mean, which mixes one event step with
  the idle ones. The script prints a rates table — t_snap (s per snapshot), t_track (µs
  per particle and model day), t_write and t_reread (µs per particle-record); a derived
  rate more than 10× away from the measured example below is almost surely a unit or
  arithmetic error — find it before using it.
- **Break-even particle count** (writing the records costs as much as one full re-run),
  computed per mesh in the script and stated as an estimate, per process and model day:
  ```
  P* = N_snap·t_snap / (N_rec·t_write − t_track_day)     # N_snap, N_rec per model day; ≤ 0 → writing never outweighs a re-run
  P*(K) = (K−1)·N_snap·t_snap / (N_rec·t_write − (K−1)·(t_track_day − N_rec·t_reread))
        # K questions: the storing run answers the first in-run, the other K−1 re-read the archive; ≤ 0 → storing always wins
  ```
  Below P*(K) storing is cheaper, above it re-running each question is. P* is per
  process and per model day: it does not change with the run length. Re-reading is one
  to two orders of magnitude cheaper than writing; the script prints YOUR
  `t_write / t_reread`, and the answer quotes that line, never a ratio from this skill.
  Re-running saves the writing and the storage: it pays for large releases asked few
  questions; small releases, or several analyses of one archive, favour storing.
- **Compare costs in measured TIME per event, never in bytes.** The terms scale
  differently: reading the velocity fields costs a fixed time per snapshot (independent
  of the particles), tracking and writing cost per particle, re-reading a stored
  archive costs per particle-record. Measured once (zoom 9, one 128-thread process):
  11.7 s per daily snapshot read, 0.41 µs tracking per particle and model day, 1.1 µs
  per particle-record WRITTEN, ~0.015 µs per particle-record re-read (one node, 12
  workers). A byte comparison ("re-run once trajectories outweigh the fields") was off
  by two orders of magnitude.
- **A slower phase is not a regression until it is split per event** (per record, per
  read): median, p90, p99 and max for both runs, plus `git log <old>..<new> --` the
  code path. A mean mixes a shifted median with a file-system tail (see `kelp-diagnose`).
