# Testing reversibility (forward → backward round trip)

Reversibility is measured on SHORT deterministic legs, never on a decade:
1. Same integrator and the same `uv_interp` on both legs; diffusion and beaching
   off; the backward leg starts exactly at the forward END record (a 5-day record
   off-by-one once manufactured a 1500 km "non-reversibility").
2. Seed in the deep interior, away from coasts and crop edges, where no depth
   or latitude clamp can move a particle; legs of days to weeks.
3. Report the DROPPED fraction as a number and compare on the common survivors
   (a 9 % silent drop was half of that 1500 km).
4. Closure distance (backward end vs the forward store's `release_lon/lat`; format 1: the
   seed file): median + p90 + p99 + max, normalized by the path length
   AND by the local mesh cell (median element edge); kilometres alone are not
   interpretable.
5. Calibrate before running: a clean same-scheme deep-interior round trip closes
   to ~0.16 km median in two 2-minute jobs. Fail criterion, relative to that
   floor: an order of magnitude above it, or ANY directional bias, is a bug;
   within a few × the floor is precision. Do not widen the band on your own
   ("a few km" is chaos after 10 YEARS, never after a week). Individual decadal
   paths are not reversible, only the ensemble/backward-provenance statistics
   are; a bug shows as a SYSTEMATIC bias in the short test.
6. Cost is sized by particles × steps × the measured seconds per step, never by
   mesh size: two short legs (a few thousand particles, days to weeks) are
   ~2 minutes each on one node. End the design with the filled line
   `Cost: <P> particles × <N> steps (leg = <days–weeks>) × <s/step> = <wall> per leg`,
   and the wall time you quote anywhere MUST equal that product (a 36× mismatch
   between the slot and the headline happened once). A decade-long leg is not
   typable into that slot — do not run one.
