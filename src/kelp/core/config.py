"""Configuration schema and loader for kelp.

Parameters mirror the legacy Fortran tracker's namelist.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import yaml


@dataclass
class ModeConfig:
    """Drift mode settings."""
    Fdrift2D: bool = True           # 2D tracking
    Fdrift3D: bool = False          # 3D tracking
    time_dir: float = 1.0               # +1.0 forward, -1.0 backward
    Vpuls: bool = False            # pulsation velocity (FESOM-C 2D baroclinic only)
    # Variable name overrides (None = reader auto-detects)
    var_u: Optional[str] = None
    var_v: Optional[str] = None
    var_w: Optional[str] = None
    var_eta: Optional[str] = None
    # 3D→2D reduction (when Fdrift2D=True and data has sigma layers)
    sigma_layer: Optional[int] = None   # extract this sigma layer (0=surface)
    sigma_mean: bool = False            # sigma-weighted vertical mean
    # 3D vertical interface limit (None = all interfaces)
    max_levels: Optional[int] = None    # keep only the first N interfaces from the surface
    # Scalar fields to sample along trajectories (TODO 2.4); dataset
    # variable names.  Empty list = sampling disabled (default path).
    sample_vars: List[str] = field(default_factory=list)
    # Opt into per-particle failure_reason tracking (int8 codes stamped by
    # every integrator path since 2026-08; flows into trajectory output +
    # checkpoints via the existing lazy writer plumbing).
    track_failure_reason: bool = False
    # Sign convention of the vertical velocity IN THE FILE (3-D only):
    # "up" = positive upward (FESOM2 ``w``, FESOM-C ``w`` = ``Wvel``, HEALPix
    # ``wo``/``avg_wo``), "down" = positive downward (KELP's internal
    # convention, e.g. ``utils/reconstruct_w.py`` output).  "auto" = the
    # variable's CF ``positive`` attribute, else the reader family default.
    w_positive: str = "auto"
    # Vertical rule for layer-centred horizontal velocity (3-D only):
    # "layer" = piecewise constant per layer (the default), "linear" =
    # linear in depth between the layer centres, constant above the first
    # centre and below a column's last wet centre.  On a z-level mesh:
    # node-layout u/v with numerics.uv_interp: barycentric (FESOM2
    # unod/vnod, HEALPix dual geometry), element-layout u/v with leastsq,
    # edge_leastsq or barycentric (FESOM2 u/v; the mesh diagnostic must
    # hold nlevels or zbar_e_bottom, no ice-shelf cavities); the output
    # attribute ``uv_vertical = "linear"`` records that it was on.
    uv_vertical: str = "layer"

    def __post_init__(self) -> None:
        if self.sample_vars is None:
            self.sample_vars = []
        if isinstance(self.sample_vars, str):
            self.sample_vars = [self.sample_vars]
        self.sample_vars = [str(v) for v in self.sample_vars]
        wp = str(self.w_positive).strip().lower()
        if wp not in _VALID_W_POSITIVE:
            raise ValueError(
                f"mode.w_positive must be one of {_VALID_W_POSITIVE}, "
                f"got {self.w_positive!r}")
        self.w_positive = wp
        # time_dir multiplies every interpolated velocity while the clock
        # advances by its sign only, so anything but +-1 silently rescales
        # (2.0) or freezes (0.0) the flow (audit 2026-10-04, F-040).
        td = self.time_dir
        if (isinstance(td, bool) or not isinstance(td, numbers.Real)
                or float(td) not in (1.0, -1.0)):
            raise ValueError(
                "mode.time_dir must be 1 (forward) or -1 (backward), got "
                f"{td!r}")
        self.time_dir = float(td)
        # The element u/v reducer and the scalar/node reducer of FESOM-C
        # resolved the pair in opposite orders (audit 2026-10-04, F-045).
        if self.sigma_layer is not None and self.sigma_mean:
            raise ValueError(
                "mode.sigma_layer and mode.sigma_mean are alternatives (one "
                "layer OR the thickness-weighted mean); set only one")
        uvv = str(self.uv_vertical).strip().lower()
        if uvv not in _VALID_UV_VERTICAL_OPTION:
            raise ValueError(
                f"mode.uv_vertical must be one of {_VALID_UV_VERTICAL_OPTION}, "
                f"got {self.uv_vertical!r}")
        if uvv == "linear" and not self.Fdrift3D:
            raise ValueError(
                "mode.uv_vertical: 'linear' interpolates the horizontal "
                "velocity in depth and needs 3-D tracking (mode.Fdrift3D: "
                "true); a 2-D run reads one level")
        self.uv_vertical = uvv


_VALID_W_POSITIVE = ("auto", "up", "down")
#: mode.uv_vertical: the vertical rule for layer-centred u/v
_VALID_UV_VERTICAL_OPTION = ("layer", "linear")
#: == numerics.integrators.INTEGRATORS (pinned by test)
_VALID_INTEGRATORS = ("euler", "rk2", "rk4", "rk45")
#: == numerics.vertical._W_METHOD_MAP (pinned by test)
_VALID_W_INTERP = ("nearest", "leastsq", "barycentric", "idw")
_VALID_ON_EXHAUSTED = ("stop", "cycle")
_VALID_TIME_AVERAGING = ("mean", "instant")
#: grid-hash cells must lie in (0, this) degrees (== core.mesh bound)
_GRID_HASH_CELL_MAX = 45.0


def _is_boolean(value) -> bool:
    """True for ``bool`` and numpy's ``bool_`` (neither is a cell size or a
    count, though both convert to 1.0)."""
    t = type(value)
    return isinstance(value, bool) or (t.__module__ == "numpy"
                                       and t.__name__ in ("bool", "bool_"))
_VALID_DIFFUSION_SCHEMES = ("uniform", "em", "m1")
_VALID_DIFFUSION_COMPOSITIONS = ("split", "combined", "split_rk")
_VALID_KH_EVAL_POSITIONS = ("start", "end")
_VALID_SAMPLE_INTERP = ("nearest", "leastsq", "barycentric", "idw")
_VALID_UV_INTERP = ("leastsq", "barycentric", "edge_leastsq")
_VALID_EDGE_LSQ_BC = ("freeslip", "noslip")
_VALID_ELEM_SEARCH = ("local", "walk")
_VALID_TEMPORAL_RECONSTRUCTION = ("linear", "pchip")
_VALID_BEACH_CRITERIA = ("depth", "coastal_element", "on_failure", "stochastic")
_VALID_RESUSPENSION = ("never", "auto", "stochastic", "auto_or_stochastic")


@dataclass
class DiffusionConfig:
    """Stochastic diffusion parameters.

    When ``enabled`` is False, kelp behaves exactly as before — no
    random kicks, no extra reads, no metadata.  When True the integrator
    composes a stochastic kick with the deterministic step according to
    ``scheme`` and ``composition`` (see ``docs/diffusion.md``).
    """
    enabled: bool = False
    scheme: str = "uniform"                 # "uniform" | "em" | "m1"
    kh_zonal: float = 0.0                   # m^2/s, used when scheme="uniform"
    kh_meridional: float = 0.0              # m^2/s, used when scheme="uniform"
    var_kh_zonal: Optional[str] = None      # dataset variable name
    var_kh_meridional: Optional[str] = None
    dres_m: Optional[float] = None          # central-diff offset in metres
    seed: int = 0                           # global RNG seed
    composition: str = "split"              # "split" | "combined" | "split_rk"
    # The split-RK stage: where to sample Kh and ∂K for split-RK kicks.
    #   "end"   — at the post-RK position (the default).
    #   "start" — at the pre-step position.
    # Ignored for split / combined modes.
    kh_eval_position: str = "end"
    # Vertical Az diffusion (TODO 2.2b) — split kick after the
    # deterministic step, 3-D runs only.  Default off = byte-identical.
    az_enabled: bool = False                # vertical Az kick on/off
    az_const: float = 0.0                   # m²/s, used when var_az unset
    var_az: Optional[str] = None            # dataset Az variable (FESOM2 z-level, v1)

    def __post_init__(self) -> None:
        if self.scheme not in _VALID_DIFFUSION_SCHEMES:
            raise ValueError(
                f"DiffusionConfig.scheme must be one of {_VALID_DIFFUSION_SCHEMES}, "
                f"got {self.scheme!r}"
            )
        if self.composition not in _VALID_DIFFUSION_COMPOSITIONS:
            raise ValueError(
                f"DiffusionConfig.composition must be one of "
                f"{_VALID_DIFFUSION_COMPOSITIONS}, got {self.composition!r}"
            )
        if self.kh_eval_position not in _VALID_KH_EVAL_POSITIONS:
            raise ValueError(
                f"DiffusionConfig.kh_eval_position must be one of "
                f"{_VALID_KH_EVAL_POSITIONS}, got {self.kh_eval_position!r}"
            )
        if self.kh_zonal < 0 or self.kh_meridional < 0:
            raise ValueError(
                f"DiffusionConfig.kh_zonal/kh_meridional must be >= 0, "
                f"got ({self.kh_zonal}, {self.kh_meridional})"
            )
        if self.az_const < 0:
            raise ValueError(
                f"DiffusionConfig.az_const must be >= 0, got {self.az_const}"
            )
        if self.az_enabled and not self.enabled:
            raise ValueError(
                "DiffusionConfig.az_enabled requires enabled: true (the "
                "whole diffusion block is dormant otherwise) — for a "
                "vertical-only run set enabled: true and leave "
                "kh_zonal/kh_meridional at 0"
            )
        if self.az_enabled and self.scheme in ("em", "m1"):
            raise ValueError(
                "DiffusionConfig.az_enabled is not supported with "
                f"scheme={self.scheme!r} (v1): the vertical Az kick "
                "composes with the uniform/split path only"
            )
        if self.dres_m is not None and self.dres_m <= 0:
            raise ValueError(
                f"DiffusionConfig.dres_m must be positive when set, got {self.dres_m}"
            )
        # ``seed`` feeds the counter-RNG via ``np.uint64(...)`` (see
        # ``diffusion.py``).  Negative values would wrap to huge
        # positive uints, and values ≥ 2**64 would overflow the cast.
        # Reject both at construction time to surface the misuse here.
        if not (0 <= self.seed < (1 << 64)):
            raise ValueError(
                f"DiffusionConfig.seed must be in [0, 2**64), got {self.seed}"
            )


@dataclass
class BeachingConfig:
    """Beaching model parameters (TODO 2.2).

    ``enabled=False`` (default) → byte-identical behaviour to no-beaching.
    """
    enabled: bool = False
    criteria: List[str] = field(default_factory=lambda: ["depth"])
    # depth criterion
    depth_min: float = 0.3        # m; beach when raw H < depth_min
    use_eta: bool = False         # H = depth (+ per-timestep eta when True)
    dry_mask_var: Optional[str] = None   # node var; >0.5 = dry (OR)
    zone_depth: Optional[float] = None   # stochastic zone; None → 2*depth_min
    # stochastic criterion
    t_beach: Optional[float] = 86400.0   # s; P = 1 - exp(-dt/t_beach)
    # resuspension
    resuspension: str = "never"   # never | auto | stochastic | auto_or_stochastic
    rewet_hysteresis: float = 0.1  # m above depth_min for auto release
    min_wet_steps: int = 1
    t_resuspend: Optional[float] = None
    seed: Optional[int] = None    # None → numerics.diffusion.seed

    def __post_init__(self) -> None:
        if isinstance(self.criteria, str):
            self.criteria = [self.criteria]
        if not self.enabled:
            return  # dormant: skip all validation
        bad = [c for c in self.criteria if c not in _VALID_BEACH_CRITERIA]
        if bad or not self.criteria:
            raise ValueError(
                f"beaching.criteria must be a non-empty subset of "
                f"{_VALID_BEACH_CRITERIA}, got {self.criteria!r}")
        if self.resuspension not in _VALID_RESUSPENSION:
            raise ValueError(
                f"beaching.resuspension must be one of "
                f"{_VALID_RESUSPENSION}, got {self.resuspension!r}")
        if "stochastic" in self.criteria and (
                self.t_beach is None or self.t_beach <= 0):
            raise ValueError(
                "beaching.t_beach (s) must be > 0 when the 'stochastic' "
                "criterion is enabled")
        if self.resuspension in ("stochastic", "auto_or_stochastic") and (
                self.t_resuspend is None or self.t_resuspend <= 0):
            raise ValueError(
                "beaching.t_resuspend (s) must be > 0 for stochastic "
                "resuspension")
        if self.min_wet_steps < 1:
            raise ValueError(
                f"beaching.min_wet_steps must be >= 1, got "
                f"{self.min_wet_steps}")
        if self.depth_min <= 0:
            raise ValueError(
                f"beaching.depth_min must be > 0, got {self.depth_min}")
        if self.seed is not None and not (0 <= int(self.seed) < 2**64):
            raise ValueError("beaching.seed must be in [0, 2**64)")


@dataclass
class NumericsConfig:
    """Numerical parameters."""
    dt: Optional[float] = None     # integration time step in seconds (None = auto-detect)
    file_dt: Optional[float] = None  # file time step in seconds (None = auto-detect from time coord)
    output_freq: int = 1                   # output interval (every output_freq steps)
    output_flush_every: int = 0    # NetCDF only: buffer N record points in RAM, then open(append)/write-batch/close so the file is unlocked & readable mid-run (0 = keep file open + sync each append, current/byte-identical behaviour; Zarr ignores this — it is already lock-free)
    Dmin: float = 0.5              # minimum water depth (m)
    z0b_min: float = 0.1           # bottom roughness length scale (m)
    Dch: float = 1.0               # search radius for nearest node (degrees)
    integrator: str = "euler"      # euler / rk2 / rk4 / rk45 (adaptive, 2D leastsq only)
    # rk45 only (TODO 2.3): per-substep absolute position tolerance (m)
    # and minimum substep (s); h_min None -> dt * 1e-4 resolved at run.
    rk45_tol_m: float = 1.0
    rk45_h_min: Optional[float] = None
    uv_interp: str = "leastsq"     # u/v interpolation: leastsq (Fortran-parity default) / barycentric / edge_leastsq
    edge_leastsq_bc: str = "freeslip"  # edge_leastsq boundary ghosts: freeslip (mirror image: tangential kept, normal reversed) / noslip (-u_own); inert for other uv_interp methods
    elem_search: str = "local"     # containing-element search: local (per-step nearest-node scan, default) / walk (search once at release, then cross edges incrementally; unfused paths only)
    w_interp: str = "leastsq"     # W interpolation: nearest / leastsq / barycentric / idw
    sample_interp: str = "leastsq"  # scalar sampling: nearest / leastsq / barycentric / idw
    temporal_interp: bool = False  # interpolate velocity between file time steps
    temporal_reconstruction: str = "linear"  # time blend between snapshots: linear (default, byte-identical) / pchip (4-snapshot monotone cubic)
    velocity_float32: bool = False  # keep u/v/w in float32 (native FESOM2 dtype) to halve snapshot memory/IO; default float64 (byte-identical). TODO 5.3
    prefetch_size: int = 1          # bulk-read this many velocity snapshots per disk read (1 = per-timestep, byte-identical). Cuts I/O on big meshes. TODO 5.2
    shared_fields_generation: Optional[str] = None  # namespace generation of the shared cache (default: $KELP_SHARED_FIELDS_GENERATION, else the SLURM array/job id): shards of one run share it, separate runs never reuse each other's snapshots
    shared_fields_dir: Optional[str] = None  # node-shared snapshot cache root: ONE co-resident process reads+publishes each processed snapshot, sibling shard processes attach the same bytes read-only (mmap). Point at node tmpfs or a run dir on the shared FS. None = off (byte-identical); incompatible knobs fail open with a warning. TODO 6.9
    checkpoint_freq: int = 0       # write restart checkpoint every N steps (0 = off)
    # Periodic boundary windows for idealized regional domains (channels,
    # ridges): after each step ALIVE particles outside [lo, hi) are folded
    # back in (post-step, before beaching/record).  Contract: the MESH must
    # carry a ghost band beyond the window at least one step's displacement
    # wide (the fold does not create seam connectivity).  None = off,
    # byte-identical.
    periodic_lon: Optional[Tuple[float, float]] = None
    periodic_lat: Optional[Tuple[float, float]] = None
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    beaching: BeachingConfig = field(default_factory=BeachingConfig)

    def __post_init__(self) -> None:
        # Allow ``diffusion`` to be passed as a plain dict (from YAML) and
        # convert it into a DiffusionConfig.  Unknown keys are dropped to
        # match the lenient behaviour of ``_build_section``.
        if self.diffusion is None:          # an empty YAML block is null
            self.diffusion = DiffusionConfig()
        if self.beaching is None:
            self.beaching = BeachingConfig()
        for _blk, _cls in (("diffusion", DiffusionConfig),
                           ("beaching", BeachingConfig)):
            if not isinstance(getattr(self, _blk), (dict, _cls)):
                raise ValueError(
                    f"numerics.{_blk} must be a mapping of keys (or empty), "
                    f"got {getattr(self, _blk)!r}")
        if isinstance(self.diffusion, dict):
            valid = {f.name for f in DiffusionConfig.__dataclass_fields__.values()}
            filtered = {k: v for k, v in self.diffusion.items() if k in valid}
            self.diffusion = DiffusionConfig(**filtered)
        if isinstance(self.beaching, dict):
            valid = {f.name for f in BeachingConfig.__dataclass_fields__.values()}
            filtered = {k: v for k, v in self.beaching.items() if k in valid}
            self.beaching = BeachingConfig(**filtered)
        # Integrator and W method were checked only when first used — the
        # integrator after the output file had been created (audit
        # 2026-10-04, F-063).
        if self.integrator not in _VALID_INTEGRATORS:
            raise ValueError(
                f"numerics.integrator must be one of {_VALID_INTEGRATORS}, "
                f"got {self.integrator!r}"
            )
        if self.w_interp not in _VALID_W_INTERP:
            raise ValueError(
                f"numerics.w_interp must be one of {_VALID_W_INTERP}, "
                f"got {self.w_interp!r}"
            )
        if self.uv_interp not in _VALID_UV_INTERP:
            raise ValueError(
                f"numerics.uv_interp must be one of {_VALID_UV_INTERP}, "
                f"got {self.uv_interp!r}"
            )
        if self.edge_leastsq_bc not in _VALID_EDGE_LSQ_BC:
            raise ValueError(
                f"numerics.edge_leastsq_bc must be one of "
                f"{_VALID_EDGE_LSQ_BC}, got {self.edge_leastsq_bc!r}"
            )
        if self.elem_search not in _VALID_ELEM_SEARCH:
            raise ValueError(
                f"numerics.elem_search must be one of {_VALID_ELEM_SEARCH}, "
                f"got {self.elem_search!r}"
            )
        # Periodic windows: 2 finite floats, hi > lo; the LON window must
        # lie inside [-180, 180] — the integrators normalize every
        # position to that branch after displacement (wrap_lon), so a
        # window like [300, 307] would misfold even stationary particles
        # (305 -> -55 -> wrong fold).  Express the channel in the
        # [-180, 180) branch instead.  Lat window inside the poles.
        for _pname in ("periodic_lon", "periodic_lat"):
            _win = getattr(self, _pname)
            if _win is None:
                continue
            if isinstance(_win, (str, bytes)):   # '12' iterated to (1, 2)
                raise ValueError(
                    f"numerics.{_pname} must be a [lo, hi] pair of "
                    f"numbers, got the string {_win!r}"
                )
            try:
                _vals = [float(v) for v in _win]
            except (TypeError, ValueError):
                raise ValueError(
                    f"numerics.{_pname} must be a [lo, hi] pair of "
                    f"numbers, got {_win!r}"
                ) from None
            if len(_vals) != 2 or not all(math.isfinite(v) for v in _vals):
                raise ValueError(
                    f"numerics.{_pname} must be two finite numbers "
                    f"[lo, hi], got {_win!r}"
                )
            lo, hi = _vals
            if hi <= lo:
                raise ValueError(
                    f"numerics.{_pname}: hi must exceed lo, got "
                    f"[{lo}, {hi}]"
                )
            if _pname == "periodic_lon" and (lo <= -180.0 or hi >= 180.0):
                raise ValueError(
                    f"numerics.periodic_lon window must lie STRICTLY "
                    f"inside (-180, 180): tracking normalizes longitudes "
                    f"to that branch after every displacement, and a "
                    f"window (or its ghost band) touching the ±180 cut "
                    f"lets the branch wrap relocate positions across the "
                    f"fold (silent misfold). Express the channel away "
                    f"from the dateline. Got [{lo}, {hi}]"
                )
            if _pname == "periodic_lat" and (lo <= -90.0 or hi >= 90.0):
                raise ValueError(
                    f"numerics.periodic_lat window must lie strictly "
                    f"inside (-90, 90) — clamp_lat would swallow the "
                    f"overshoot at a pole before the fold. Got "
                    f"[{lo}, {hi}]"
                )
            setattr(self, _pname, (lo, hi))
        if self.sample_interp not in _VALID_SAMPLE_INTERP:
            raise ValueError(
                f"numerics.sample_interp must be one of {_VALID_SAMPLE_INTERP}, "
                f"got {self.sample_interp!r}"
            )
        if self.temporal_reconstruction not in _VALID_TEMPORAL_RECONSTRUCTION:
            raise ValueError(
                "numerics.temporal_reconstruction must be one of "
                f"{_VALID_TEMPORAL_RECONSTRUCTION}, got "
                f"{self.temporal_reconstruction!r}"
            )
        if self.rk45_tol_m <= 0:
            raise ValueError(
                f"numerics.rk45_tol_m must be > 0 m, got {self.rk45_tol_m!r}"
            )
        if self.rk45_h_min is not None and self.rk45_h_min <= 0:
            raise ValueError(
                "numerics.rk45_h_min must be > 0 seconds (or null for the "
                f"dt-derived default), got {self.rk45_h_min!r}"
            )
        # Reject bool (isinstance(True, int) is True), non-int (quoted "4",
        # fractional 1.9), and < 1 — fail-fast rather than silently truncating
        # in _VelCache or crashing later on a string comparison.
        if self.shared_fields_dir is not None and not isinstance(
                self.shared_fields_dir, str):
            raise ValueError(
                "numerics.shared_fields_dir must be a directory path "
                f"string or null, got {self.shared_fields_dir!r}")
        if (isinstance(self.prefetch_size, bool)
                or not isinstance(self.prefetch_size, int)
                or self.prefetch_size < 1):
            raise ValueError(
                "numerics.prefetch_size must be an integer >= 1, got "
                f"{self.prefetch_size!r}"
            )
        # Reject bool / non-int / negative (0 = disabled = keep-file-open).
        if (isinstance(self.output_flush_every, bool)
                or not isinstance(self.output_flush_every, int)
                or self.output_flush_every < 0):
            raise ValueError(
                "numerics.output_flush_every must be an integer >= 0 "
                f"(0 = disabled), got {self.output_flush_every!r}"
            )


#: mesh.grid_hash_cell_rule values (== core.mesh.GRID_HASH_CELL_RULES, pinned by test)
_GRID_HASH_CELL_RULES = ("occupancy", "legacy")


@dataclass
class MeshConfig:
    """Mesh source settings."""
    type: str = "fesomc"           # fesomc / fesom2 / nemo / healpix / fesom14
    meshpath: str = ""             # path to mesh files (ASCII dir or NC file)
    # Earth radius (m) for mesh-derived areas/metrics.  None -> the mesh
    # family default chosen by the reader: FESOM2 6_367_500.0 (the FESOM
    # namelist value), FESOM-C 6_400_000.0 (legacy Fortran family value).
    # Set explicitly only to match a model build with a non-default radius.
    r_earth: Optional[float] = None
    # Nearest-node grid-hash cell size (degrees).  None -> chosen by
    # grid_hash_cell_rule below (by default measured per mesh, starting
    # from the reader's cell sqrt(domain_area / n_nodes) * 2, floored at
    # 0.05 deg).  An explicit value is pinned and always wins; with
    # exact_nearest on, the anchor of every point inside an element is exact
    # for any cell and only the speed changes (docs/configuration.md).
    grid_hash_cell_size: Optional[float] = None
    # Exact nearest-node anchor guard: every 3x3-window hit that is not
    # provably the true nearest is resolved by the in-kernel exact search.
    # Default ON since 2026-09-12; False = the pre-guard anchors bit for bit.
    exact_nearest: bool = True
    # Automatic grid-hash cell rule when grid_hash_cell_size is not set:
    # "occupancy" (default, 2026-09-22) measures window occupancy / scan
    # cost on element centres and takes the cheapest safe cell
    # (core.mesh.choose_grid_hash_cell; applied by run_tracking / kelp info
    # after the exact_nearest flag); "legacy" keeps the reader's mean-spacing
    # cell (pre-2026-09-22 behaviour, bit-identical).  An explicit
    # grid_hash_cell_size always wins.  The resolved cell is stamped in the
    # output metadata (grid_hash_cell, grid_hash_cell_rule).
    grid_hash_cell_rule: Optional[str] = "occupancy"
    # Coordinate-frame rotation for pole-safe tracking (FESOM2 and HEALPix;
    # the FESOM-C and NEMO readers refuse it).
    # None (default) -> no rotation, byte-identical behaviour.
    # "fesom2"       -> the standard FESOM Euler angles (50, 15, -90):
    #                   the computational pole sits on Greenland.
    # [a, b, g]      -> custom Euler angles in degrees.
    rotation: Optional[Union[str, List[float]]] = None
    # --- healpix-only keys (ignored by the other readers) ---------------
    # Zoom fallback when the store carries no healpix crs attrs:
    # nside = 2**healpix_zoom.
    healpix_zoom: Optional[int] = None
    # Regional crop at mesh build: pixel-centre test against [W, E, S, N]
    # (degrees; W > E = a window crossing the dateline).
    crop_bbox: Optional[List[float]] = None
    # Vertical interface depths for stores whose level axis is indices
    # (GRIB): a FESOM aux3d.out path OR an inline list of metres.  Also
    # feeds the deepest-finite-level bathymetry in 2-D.
    levels: Optional[Union[str, List[float]]] = None
    # Mesh mapping of the pixelation (docs/healpix.md §1 and §1b):
    # "dual"  -> nodes = pixel centres, corner dual cells as elements
    #            (node-layout u/v => uv_interp: barycentric);
    # "pixel" -> elements = the pixels themselves, nodes = pixel corners
    #            (element-layout u/v => the fused leastsq kernels;
    #            v1 is 2-D only, no sampling / spatial Kh).
    healpix_geometry: str = "dual"
    # --- fesom14-only key (refused for the other mesh types) ------------
    # Euler angles [alpha, beta, gamma] (degrees, as in the run's namelist)
    # of the rotated frame a FESOM 1.4 ASCII mesh is stored in; the reader
    # rotates the coordinates back to geographic on load.  None = the mesh
    # is geographic.  Never defaulted: the reader warns when a global mesh
    # without an Arctic node is loaded without it.  Unrelated to `rotation`
    # (the frame KELP tracks in).
    input_rotation: Optional[List[float]] = None

    def __post_init__(self) -> None:
        if self.r_earth is not None and self.r_earth <= 0:
            raise ValueError(
                f"mesh.r_earth must be > 0 when set, got {self.r_earth}"
            )
        self.exact_nearest = bool(self.exact_nearest)
        if self.grid_hash_cell_rule is None:        # YAML null = the default, like the siblings
            self.grid_hash_cell_rule = "occupancy"
        rule = str(self.grid_hash_cell_rule).lower()
        if rule not in _GRID_HASH_CELL_RULES:
            raise ValueError(
                "mesh.grid_hash_cell_rule must be 'occupancy' or 'legacy', "
                f"got {self.grid_hash_cell_rule!r}")
        self.grid_hash_cell_rule = rule
        if self.grid_hash_cell_size is not None:
            try:
                if _is_boolean(self.grid_hash_cell_size):
                    raise TypeError("a boolean")              # true -> 1.0
                cell = float(self.grid_hash_cell_size)
            except (TypeError, ValueError):
                cell = float("nan")
            # The mesh refuses a pinned cell outside (0, 45) on every rebuild;
            # accepting it here broke the first rebuild after the pin
            # (audit 2026-10-04, F-041).
            if not math.isfinite(cell) or not 0.0 < cell < _GRID_HASH_CELL_MAX:
                raise ValueError(
                    "mesh.grid_hash_cell_size must be a finite value in "
                    f"(0, {_GRID_HASH_CELL_MAX:g}) degrees when set, got "
                    f"{self.grid_hash_cell_size!r}"
                )
            self.grid_hash_cell_size = cell
        if self.healpix_geometry not in ("dual", "pixel"):
            raise ValueError(
                "mesh.healpix_geometry must be 'dual' or 'pixel', got "
                f"{self.healpix_geometry!r}"
            )
        if self.healpix_zoom is not None:
            if (isinstance(self.healpix_zoom, bool)
                    or not isinstance(self.healpix_zoom, int)
                    or self.healpix_zoom < 0):
                raise ValueError(
                    f"mesh.healpix_zoom must be an integer >= 0, got "
                    f"{self.healpix_zoom!r}")
        if self.crop_bbox is not None:
            try:
                if isinstance(self.crop_bbox, (str, bytes)):
                    raise TypeError("a string, not a list")   # '1234' -> 4 digits
                vals = [float(x) for x in self.crop_bbox]
            except (TypeError, ValueError):
                vals = None
            if (vals is None or len(vals) != 4
                    or not all(math.isfinite(v) for v in vals)):
                raise ValueError(
                    "mesh.crop_bbox must be [W, E, S, N] (4 finite degrees), "
                    f"got {self.crop_bbox!r}")
            if not (-90.0 <= vals[2] < vals[3] <= 90.0):
                raise ValueError(
                    "mesh.crop_bbox latitudes must satisfy "
                    f"-90 <= S < N <= 90, got S={vals[2]}, N={vals[3]}")
            self.crop_bbox = vals
        if self.levels is not None and not isinstance(self.levels, str):
            try:
                vals = [float(x) for x in self.levels]
            except (TypeError, ValueError):
                vals = None
            if (vals is None or len(vals) < 2
                    or not all(math.isfinite(v) for v in vals)):
                raise ValueError(
                    "mesh.levels must be an aux3d.out path or a list of "
                    f">= 2 finite interface depths (m), got {self.levels!r}")
            self.levels = vals
        if self.rotation is not None:
            if isinstance(self.rotation, str):
                if self.rotation != "fesom2":
                    raise ValueError(
                        f"mesh.rotation must be 'fesom2' or a 3-list of "
                        f"Euler angles, got {self.rotation!r}")
            else:
                try:
                    vals = [float(x) for x in self.rotation]
                except (TypeError, ValueError):
                    vals = None
                if (vals is None or len(vals) != 3
                        or not all(math.isfinite(v) for v in vals)):
                    raise ValueError(
                        f"mesh.rotation must be 'fesom2' or a 3-list of "
                        f"finite Euler angles (deg), got {self.rotation!r}")
                self.rotation = vals
        if self.input_rotation is not None and self.type != "fesom14":
            # the other readers would ignore it: a mesh stored in a rotated
            # frame would then be read as geographic without a word
            raise ValueError(
                "mesh.input_rotation is set, but mesh.type is "
                f"{self.type!r}: the angles of the frame a mesh is stored in "
                "apply to FESOM 1.4 ASCII meshes only (mesh.type: fesom14). "
                "Remove mesh.input_rotation (mesh.rotation sets the frame "
                "KELP tracks in)")
        if self.input_rotation is not None:
            vals = None
            if not isinstance(self.input_rotation, (str, bytes)):
                try:
                    items = list(self.input_rotation)
                    if not any(_is_boolean(x) for x in items):
                        vals = [float(x) for x in items]
                except (TypeError, ValueError):
                    vals = None
            if (vals is None or len(vals) != 3
                    or not all(math.isfinite(v) for v in vals)):
                raise ValueError(
                    "mesh.input_rotation must be the 3 finite Euler angles "
                    "[alpha, beta, gamma] (deg) of the frame a FESOM 1.4 "
                    "ASCII mesh is stored in, as in the run's namelist, got "
                    f"{self.input_rotation!r}")
            self.input_rotation = vals


@dataclass
class DataConfig:
    """Input/output paths and data properties.

    ``time_averaging`` describes whether the velocity fields in the input
    files are **time-averaged** (``"mean"``) or **instantaneous snapshots**
    (``"instant"``).  This is important because time-averaged output
    (e.g. FESOM2 monthly means) has timestamps at the *end* of the
    averaging window, so the effective time must be shifted to the
    *midpoint* of each window.  This field should always be set
    explicitly; if omitted a warning is printed with a suggestion based
    on the mesh type.  ``mesh.type: fesom14`` is the exception: its reader
    resolves what time each record represents, ``time_averaging`` is
    refused there, and ``time_stamp`` / ``time_window`` / ``time_offset``
    say what the stamps mean when the files cannot (refused with any other
    mesh type when the run starts).
    """
    ncinput: str = ""              # velocity NetCDF file or directory
    ncinput_v: str = ""            # separate v file (empty = u+v in same ncinput)
    ncinput_w: str = ""            # separate w file (empty = no w or w in ncinput)
    ncinput_eta: str = ""          # separate ssh store (healpix: the 2-D store; empty = ssh in ncinput or auto-derived sibling)
    inicloud: str = ""             # initial particle positions file
    output_path: str = ""          # trajectory NetCDF output (empty = in-memory only)
    time_averaging: Optional[str] = None  # "mean" or "instant" (None triggers warning)
    # Global index of inicloud's first particle in the full release (a shard
    # of a sharded run); None = the seed file's '# kelp particle_offset N'
    # header (utils/split_seeds.py) or 0.  Keys every random draw.
    particle_offset: Optional[int] = None
    # --- fesom14 only: what each record's time stamp means --------------
    # Passed to the reader's open_dataset only when set; any other mesh type
    # refuses them when the run starts.  time_stamp: start | centre | end |
    # instant (where the stamp sits in its averaging window; instant =
    # snapshots); time_window: calendar_month | calendar_year | calendar_day
    # | '<number> <s|min|h|d>' (the averaging window, or the snapshot
    # cadence); the two are given together or not at all, and then decide
    # over the files' time bounds and output schedule.  time_offset:
    # '<+-number> <s|min|h|d>' added to the raw stamps before anything else
    # (e.g. a raw job that started mid-day).
    time_stamp: Optional[str] = None
    time_window: Optional[str] = None
    time_offset: Optional[str] = None

    def __post_init__(self) -> None:
        if (self.time_stamp is not None or self.time_window is not None
                or self.time_offset is not None):
            # the reader's own check, imported only when a key is set
            from ..readers.fesom14._timeaxis import TimeOptions
            TimeOptions(stamp=self.time_stamp, window=self.time_window,
                        offset=self.time_offset).validate()
        if self.particle_offset is not None:
            from .release import check_particle_offset
            try:
                self.particle_offset = check_particle_offset(self.particle_offset)
            except ValueError:
                raise ValueError(
                    "data.particle_offset must be an integer >= 0 or null, got "
                    f"{self.particle_offset!r}") from None
        if (self.time_averaging is not None
                and self.time_averaging not in _VALID_TIME_AVERAGING):
            raise ValueError(
                "data.time_averaging must be 'mean', 'instant' or null, got "
                f"{self.time_averaging!r}")


@dataclass
class TimeConfig:
    """Time control.

    Two ways to specify the tracking window (``n_steps`` takes priority):

    * ``n_steps`` — number of tracking steps to perform.
    * ``end_time_idx`` — last file time index (legacy); converted to
      ``n_steps = end_time_idx - start_time_idx``.

    ``on_exhausted`` controls what happens when file time indices run out:

    * ``"stop"`` — halt the simulation (default).
    * ``"cycle"`` — wrap around to file index 0 (periodic forcing).
    """
    start_time_idx: int = 0                  # first file time index
    end_time_idx: Optional[int] = None       # legacy: last time index (use n_steps instead)
    n_steps: Optional[int] = None            # number of tracking steps (preferred)
    on_exhausted: str = "stop"               # "stop" or "cycle"
    num_runs: int = 1                        # number of realizations
    run_interval: int = 0                    # time layers between realizations
    # Delayed/staggered release (TODO 2.5): replicate the seed set into
    # ``n_pulses`` pulses at the same location(s), pulse p released
    # ``p * pulse_interval`` seconds after the run start.  n_pulses=1 +
    # pulse_interval=0 (defaults) => feature off, byte-identical.  Composes
    # additively with an optional per-particle 4th seed column (release_sec).
    n_pulses: int = 1                        # number of release pulses
    pulse_interval: float = 0.0              # seconds between pulses
    # Date-based regime (resolved by run_tracking._resolve_start_date):
    # start_date overrides start_time_idx; duration_days converts to
    # n_steps = duration_days * 86400 / dt.
    start_date: Optional[str] = None         # e.g. "2020-01-01T00:00"
    duration_days: Optional[float] = None    # e.g. 27.9

    def __post_init__(self) -> None:
        # resolve_time_params checked on_exhausted, but the start_date +
        # duration_days path never calls it: 'Cycle' ran as 'stop'
        # (audit 2026-10-04, F-064).
        if self.on_exhausted not in _VALID_ON_EXHAUSTED:
            raise ValueError(
                f"time.on_exhausted must be one of {_VALID_ON_EXHAUSTED}, "
                f"got {self.on_exhausted!r}")
        # duration_days is converted only together with start_date; alone it
        # was dropped silently.
        if self.duration_days is not None:
            dd = self.duration_days
            if (isinstance(dd, bool) or not isinstance(dd, numbers.Real)
                    or not math.isfinite(float(dd)) or float(dd) <= 0.0):
                raise ValueError(
                    "time.duration_days must be a finite number of days > 0, "
                    f"got {dd!r}")
            if self.start_date is None:
                raise ValueError(
                    "time.duration_days needs time.start_date (it is converted "
                    "to n_steps from the start date); with start_time_idx set "
                    "time.n_steps instead")


@dataclass
class AppConfig:
    """Top-level application configuration."""
    mode: ModeConfig = field(default_factory=ModeConfig)
    numerics: NumericsConfig = field(default_factory=NumericsConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    data: DataConfig = field(default_factory=DataConfig)
    time: TimeConfig = field(default_factory=TimeConfig)


def _load_yaml(path: Path) -> dict:
    """Read a YAML file, return dict."""
    try:
        return yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Config file not found: {path}") from exc


def _validate_config_dict(data: dict) -> dict:
    """Check that required sections are present."""
    required = {"mode", "numerics", "mesh", "data", "time"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"Config missing sections: {sorted(missing)}")
    return data


def _build_section(cls, data: dict, section: str):
    """Create a dataclass instance from a config section, ignoring unknown keys."""
    valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
    # An empty YAML block (`mode:` with nothing under it) is null: defaults.
    # Anything else that is not a mapping is a malformed section.
    raw = data.get(section)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"config section {section!r} must be a mapping of keys (or "
            f"empty), got {type(raw).__name__} {raw!r}")
    filtered = {k: v for k, v in raw.items() if k in valid_keys}
    return cls(**filtered)


def _resolve_paths(cfg: AppConfig, config_dir: Path) -> None:
    """Resolve relative paths in config relative to the config file directory.

    Expands ``~`` and makes relative paths absolute using *config_dir*
    as the base.  Modifies *cfg* in place.
    """
    def _resolve(p: str) -> str:
        if not p:
            return p
        expanded = Path(p).expanduser()
        if expanded.is_absolute():
            return str(expanded)
        return str((config_dir / expanded).resolve())

    cfg.mesh.meshpath = _resolve(cfg.mesh.meshpath)
    if isinstance(cfg.mesh.levels, str):
        cfg.mesh.levels = _resolve(cfg.mesh.levels)
    cfg.data.ncinput = _resolve(cfg.data.ncinput)
    cfg.data.ncinput_v = _resolve(cfg.data.ncinput_v)
    cfg.data.ncinput_w = _resolve(cfg.data.ncinput_w)
    cfg.data.ncinput_eta = _resolve(cfg.data.ncinput_eta)
    cfg.data.inicloud = _resolve(cfg.data.inicloud)
    cfg.data.output_path = _resolve(cfg.data.output_path)


def load_config(path: str) -> AppConfig:
    """Load an AppConfig from a YAML file.

    Relative paths in the config are resolved relative to the config
    file's directory (not the current working directory).  ``~`` is
    expanded.

    Parameters
    ----------
    path : str
        Path to the YAML configuration file.

    Returns
    -------
    AppConfig
        Fully populated configuration object with absolute paths.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If required sections are missing.
    """
    config_path = Path(path).resolve()
    data = _validate_config_dict(_load_yaml(config_path))
    cfg = AppConfig(
        mode=_build_section(ModeConfig, data, "mode"),
        numerics=_build_section(NumericsConfig, data, "numerics"),
        mesh=_build_section(MeshConfig, data, "mesh"),
        data=_build_section(DataConfig, data, "data"),
        time=_build_section(TimeConfig, data, "time"),
    )
    _resolve_paths(cfg, config_path.parent)
    return cfg


def resolve_time_params(time_cfg: TimeConfig) -> Tuple[int, int]:
    """Resolve (start_time_idx, n_steps) from a TimeConfig.

    Priority: ``n_steps`` > ``end_time_idx``.

    Parameters
    ----------
    time_cfg : TimeConfig

    Returns
    -------
    (start_time_idx, n_steps)

    Raises
    ------
    ValueError
        If neither ``n_steps`` nor ``end_time_idx`` is set, or if
        ``on_exhausted`` has an invalid value.
    """
    if time_cfg.on_exhausted not in ("stop", "cycle"):
        raise ValueError(
            f"on_exhausted must be 'stop' or 'cycle', got '{time_cfg.on_exhausted}'"
        )

    start = time_cfg.start_time_idx

    if time_cfg.n_steps is not None:
        return start, time_cfg.n_steps

    if time_cfg.end_time_idx is not None:
        return start, time_cfg.end_time_idx - start

    raise ValueError("TimeConfig must specify either 'n_steps' or 'end_time_idx'")


__all__ = [
    "ModeConfig",
    "NumericsConfig",
    "DiffusionConfig",
    "BeachingConfig",
    "MeshConfig",
    "DataConfig",
    "TimeConfig",
    "AppConfig",
    "load_config",
    "resolve_time_params",
]
