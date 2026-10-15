# Events, section crossings and their time axes

Read when counting crossings of a section, arrivals or other events, or relating an event count to a
+τ map. The checks of the skill hold here too (status masks, release-relative time, denominators).

- **Events versus delayed snapshots — two time axes.** A count of events per year
  (crossings of a section, arrivals) is on the EVENT time axis; a map at +τ from
  release is on the RELEASE time axis. Every table column, legend and caption says which
  ("release decade" / "crossing decade"); a figure's axis label stays "year" or "decade",
  never "(event time)". A caption that relates the two quotes the event-age
  distribution (median age at the event, share of events within τ ± 1 yr of release).
  Two such products can peak a decade apart for the same releases and still agree,
  when the median event age exceeds the snapshot's τ. Age = t_event − t_release of THAT
  particle (its `release_time`; format 1: its first non-NaN record), never the decade start or
  the run start; the
  crossing loop below carries `t_rel` per particle, writes `age_yr` per event and
  prints the share with |age − τ| ≤ 1 yr (τ = the lead time of the map it is compared
  with, never the median age); the median age comes from the age block after the loop.
- **Age of an event in a pulsed run: one horizon H.** Releases near the end of the run cannot
  reach high ages before the last record, so a median age over all releases or all events is
  biased young; a subset of early releases without an age cap mixes different follow-up
  lengths. An age statistic (age at a crossing, transit time) counts only releases with
  release + H ≤ the last record and only events with age ≤ H — it describes the ages up to H.
  H and the counted releases stand next to every age number; the share of events older than H
  is its own number, from the releases followed H_TAIL > H. The owner may ask for the
  all-release median — give it, with one line on its bias: biased young, because late releases
  cannot reach high ages (a known sign, stated without waiting for numbers). H is the owner's. An owner who asks for
  all releases has chosen no horizon: give the owner's number with its bias line and list
  `AGE HORIZON — owner's choice: all releases` under Defaults not used (a ladder may go beside it).
  Only when the owner says nothing, run a stated ladder labelled "analyst's choice" and propose one H.
- **A calendar-year event series of a pulsed run.** With releases going on all run, old
  particles keep accumulating, so a raw count per year mixes circulation with population
  growth. With the owner's A the series counts events with age ≤ A, marks the years before first
  release + A as not comparable (shaded, line kept) and prints per year the share of events older
  than A — what the cut removed. Without an A never refuse: draw the series cut at a stated A
  (labelled "analyst's choice", earlier years shaded) beside the uncut one, and propose A in the
  handback. When the owner has fixed the series or its comparable years, draw the owner's series
  and years; the share older than A stays in the counts table the script writes — it enters a
  figure, a caption or a report table only when the owner asks about age.
- **The blocks below are written for a FORWARD run.** In a backward run the format-2 `time`
  counts DOWN (each label is the end of its step, earlier and earlier) and `release_time` is each
  particle's LATEST instant, so `t − t_rel`, `nanmin(t_rel)`, `t_last − t_rel` and
  `sort_values("t")` all turn round. Run the blocks unchanged on the negated axis — `-time` for
  `ds["time"]` and `-release_time` for `t_rel` (ages then come out positive, "first" is the first in
  the particle's own direction) — and convert instants back with `year(-t)` for the calendar.
- **Section crossings from stored records:** a crossing is a sign change of (position −
  section) between two consecutive records with the particle ALIVE in both; interpolate
  longitude and depth at the crossing; count direction separately and name the
  statistic: every crossing, the first per particle (the two tables the loop writes by
  default), or the owner's (e.g. net southward minus northward per particle, built from
  the same events table). An excursion shorter than the
  record interval is missed — state the record interval (median `diff(time)`). The whole
  loop (a meridional section swaps the roles of lon and lat; `TAU` in years comes from
  the command line):
  ```python
  import numpy as np, pandas as pd, cftime   # inputs: ds, block, LAT0, LON_A, LON_B, ALIVE = 1, TAU (yr); FORCING_T0 only for format 1
  # ds = xr.open_zarr(ARCHIVE, consolidated=False, decode_times=False): time in SECONDS (the loop interpolates it)
  REC = ds["lon"].dims[1]; n_rec, P = ds.sizes[REC], ds.sizes["trajectory"]   # REC = "time" (format 2) / "obs" (format 1)
  F2 = "release_time" in ds                                 # format 2: release time stored; format 1 (interpolated-mode archives only): first non-NaN record
  events, prev = [], None                                   # prev = the previous COLUMN, carried across block seams
  t_rel = ds["release_time"].values.astype(float) if F2 else np.full(P, np.nan)
  for b0 in range(0, n_rec, block):
      blk = {v: ds[v].isel({REC: slice(b0, b0 + block)}).values for v in ("lon", "lat", "z", "status")}
      t_blk = ds["time"].values[b0:b0 + block]
      for j in range(len(t_blk)):                           # per COLUMN, never per particle
          cur = {"lon": blk["lon"][:, j], "lat": blk["lat"][:, j], "z": blk["z"][:, j], "st": blk["status"][:, j], "t": t_blk[j]}
          if not F2: newly = np.isnan(t_rel) & np.isfinite(cur["lon"]); t_rel[newly] = cur["t"]
          if prev is not None:
              s0, s1 = prev["lat"] - LAT0, cur["lat"] - LAT0                    # zonal section LAT0, LON_A..LON_B
              x = (prev["st"] == ALIVE) & (cur["st"] == ALIVE) & (np.sign(s0) != np.sign(s1)) & (s0 != 0)
              f = s0[x] / (s0[x] - s1[x])                                       # fraction of the interval at the section
              dlon = (cur["lon"][x] - prev["lon"][x] + 180.0) % 360.0 - 180.0   # dateline-safe
              lon_c = (prev["lon"][x] + f * dlon + 180.0) % 360.0 - 180.0
              z_c = prev["z"][x] + f * (cur["z"][x] - prev["z"][x]); t_c = prev["t"] + f * (cur["t"] - prev["t"])
              keep = (LON_A <= lon_c) & (lon_c <= LON_B); pid = np.nonzero(x)[0][keep]
              events.append(pd.DataFrame({"pid": pid, "t": t_c[keep], "lon": lon_c[keep], "z": z_c[keep],
                                          "dir": np.where(s0[x][keep] > 0, "S", "N"),   # from the north = southward
                                          "age_yr": (t_c[keep] - t_rel[pid]) / (365.25 * 86400)}))
          prev = cur                                        # per COLUMN, also across the block seam
  cols = ["pid", "t", "lon", "z", "dir", "age_yr"]
  ev = pd.concat(events, ignore_index=True) if events else pd.DataFrame(columns=cols)   # typed empty table when nothing crossed
  ev.to_csv("crossings_every.csv", index=False)
  ev.sort_values("t").drop_duplicates(["pid", "dir"]).to_csv("crossings_first.csv", index=False)
  cadence = float(np.median(np.diff(ds["time"].values)))
  print(f"record interval {cadence / 86400:.1f} d: an excursion through the section shorter than this is missed")
  print(f"share of crossings with |age - TAU| <= 1 yr {np.mean(np.abs(ev.age_yr - TAU) <= 1.0):.1%} (the median age: the age block below, never ev.age_yr.median())")
  units, cal = ds["time"].attrs.get("units", ""), ds["time"].attrs.get("calendar", "standard")   # format 2: "seconds since <forcing's first raw record>"
  if " since " not in units: units = f"seconds since {FORCING_T0}"   # format 1 / a forcing without an epoch: its FIRST raw timestamp, typed from the forcing file
  year = lambda s: np.array([d.year for d in cftime.num2date(np.asarray(s, dtype=float), units, cal)], dtype=int)   # the store's calendar (noleap, 360_day)
  ev["crossing_year"] = year(ev.t)
  ev["release_decade"] = year(t_rel[ev.pid.to_numpy(dtype=int)]) // 10 * 10   # after the loop
  ev.groupby(["crossing_year", "dir"]).size().unstack("dir", fill_value=0).to_csv("counts_by_year_dir.csv")  # S and N never summed
  ```
  Keep the vectorised `[keep]` indexing (a per-event Python loop over `pid` easily pairs
  a crossing with another particle's depth and time); tables keep both time names,
  `crossing_year` and `release_decade`, never a bare `decade`.

  The AGE block (after the loop; `args.h_yr` / `args.h_tail_yr` = the owner's H and H_TAIL, default
  `None`; `band_of` = per-particle band from `release_z` (format 1: the seed metadata by `sid`)):
  ```python
  YR, t_last = 365.25 * 86400.0, float(ds["time"].values[-1])
  run_yr = (t_last - np.nanmin(t_rel)) / YR
  H_list = [args.h_yr] if args.h_yr is not None else [round(f * run_yr, 1) for f in (0.25, 0.5, 0.75)]   # no owner's H: a ladder, "analyst's choice"
  first = ev.sort_values("t").drop_duplicates(["pid", "dir"]).assign(band=lambda d: band_of[d.pid.to_numpy()])
  rows = []
  for H_YR in H_list:
      H_TAIL_YR = args.h_tail_yr if args.h_tail_yr is not None else min(1.5 * H_YR, 0.9 * run_yr)
      full, tail = t_rel + H_YR * YR <= t_last, t_rel + H_TAIL_YR * YR <= t_last   # releases with H (H_TAIL) years of data
      for (d, b), e in first.groupby(["dir", "band"]):                         # directions and bands apart
          sel = e[full[e.pid.to_numpy()] & (e.age_yr <= H_YR).to_numpy()]
          et = e[tail[e.pid.to_numpy()] & (e.age_yr <= H_TAIL_YR).to_numpy()]
          rows.append({"dir": d, "band": b, "H_yr": H_YR, "H_source": "owner" if args.h_yr is not None else "analyst's choice",
                       "n_particles_counted": int(full[band_of == b].sum()), "n_particles_band": int((band_of == b).sum()),
                       "n_events": len(sel), "median_age_yr": sel.age_yr.median(), "p90_age_yr": sel.age_yr.quantile(0.9),
                       "H_tail_yr": H_TAIL_YR, "share_older_than_H": (et.age_yr > H_YR).mean() if H_TAIL_YR > H_YR else np.nan})
  pd.DataFrame(rows).to_csv("age_at_crossing_H.csv", index=False)          # every age number of the text comes from this file
  ```
  The YEAR-SERIES block (`args.age_cut_yr` = the owner's A, default `None`):
  ```python
  run_yr = (float(ds["time"].values[-1]) - np.nanmin(t_rel)) / (365.25 * 86400.0)
  SERIES = "cut at A"   # cut at A (default; a stated A when the owner gave none) | owner's (the owner fixed the series or its comparable years)
  A_YR = args.age_cut_yr if args.age_cut_yr is not None else round(0.25 * run_yr, 1)
  A_SRC = "owner" if args.age_cut_yr is not None else "analyst's choice"
  first_year = int(year([np.nanmin(t_rel)])[0])
  g = ev.assign(le_A=ev.age_yr <= A_YR).groupby(["crossing_year", "dir"])
  yr = pd.DataFrame({"n_all": g.size(), "n_age_le_A": g["le_A"].sum()}).reset_index()
  yr["share_older_than_A"] = 1.0 - yr.n_age_le_A / yr.n_all                  # what a cut at A removes, per year
  yr["n_plotted"] = yr.n_age_le_A if SERIES == "cut at A" else yr.n_all       # n_all stays beside it
  yr["comparable"] = (yr.crossing_year >= first_year + A_YR) if SERIES == "cut at A" else True   # owner's: the owner's shading
  yr.to_csv("series_by_year_dir.csv", index=False); print(f"SERIES = {SERIES}; A = {A_YR:g} yr ({A_SRC})"); print(yr.to_string(index=False))
  ```
