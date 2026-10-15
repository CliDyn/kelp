# Reading a dt ladder (convergence of an event statistic)

- Transcribe the given rungs VERBATIM first, then derive; never re-type, round or
  shift a supplied value (a shifted L1 column once produced a fabricated
  "flat, converged" segment).
- Declare convergence per published quantity, against a tolerance stated BEFORE
  reading the table: the converged rung is the coarsest one inside it. Statistic,
  map and area converge at different rungs — in either order. Compute every
  rung's % deviation from the finest; never infer convergence from two rungs or
  assume monotonicity (one measured ladder was non-monotone: two intermediate
  rungs worse than the coarsest, convergence only ~16× below the cadence, and
  the area statistic converged rungs earlier than the event fraction — that
  is ONE ladder's shape, not the expected one). Run ≥ 4 rungs and bracket the answer.
- When the statistic is a discrete capture or exit event, the report contains
  exactly ONE line beginning `MECHANISM:`, written verbatim as
  `MECHANISM: UNKNOWN — not yet measured` until you have actually computed the
  median interpolated speed at the moment of capture, the census of
  zero-velocity/drying elements, and the control arm; only then replace
  UNKNOWN with those numbers. Never write a sample, expected or hypothetical
  mechanism line, and name no physical mechanism (any shedding, aliasing or resonance
  hypothesis) anywhere until that line reads "measured". A second `MECHANISM:`
  line is a defect.
- Fix the per-release analysis horizon FIRST and cut every realisation to its
  own window; the table repeats the published statistic at a SECOND horizon —
  if the two disagree, the horizon, not dt, is the leading uncertainty (a
  measured pair of horizons gave effect sizes differing by about a factor of
  two).
- Per-rung table to fill (every number not yet computed is `<TBD>`; never put an
  example's value in a result cell; both denominator columns are mandatory):
  `| dt | statistic (born denom.) | statistic (alive denom.) | % deviation from
  the finest rung | map L1 vs finest | median speed at capture | zero-velocity
  elements % | horizon |` plus the second-horizon row, then the line
  `CONVERGED AT: statistic <dt>, map <dt>, area <dt> (tolerance <X> %)` and the
  MECHANISM line.
