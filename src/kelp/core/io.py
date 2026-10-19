"""I/O utilities: VelocityField container and trajectory writer.

VelocityField is a simple container for one timestep of velocity data;
LazyLerpField is its lazily blended view between two snapshots.  The
trajectory writer lives in ``kelp.writers``.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

try:
    from numba import njit, prange
    HAS_NUMBA = True
# Broad except on purpose: a numpy/numba binary mismatch raises
# ValueError (not ImportError) at import time; core.io must stay
# importable in any environment and just lose the fast path.
except Exception:  # pragma: no cover - exercised only without numba
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

    prange = range


_VALID_KH_LAYOUTS = ("node", "element")
_VALID_UV_LAYOUTS = ("element", "node")
#: Placement of the 3-D horizontal velocity data points in the vertical:
#: ``"layer"`` (layer centres, piecewise constant in depth — the default),
#: ``"level"`` (on the levels, linear in depth) and ``"layer_linear"``
#: (layer-centred data evaluated linearly in depth).
_VALID_UV_VERTICAL = ("layer", "level", "layer_linear")

#: Arrays smaller than this take the plain numpy path in :func:`_blend` —
#: below ~64K elements the numba thread fork/join overhead exceeds the
#: blend itself (and the test-suite's tiny meshes stay off the JIT path).
_BLEND_MIN_SIZE = 1 << 16

#: One-time per-process markers for the "fast path missed only on
#: contiguity" warnings (shared idea with temporal_reconstruct).
_CONTIG_WARNED: set = set()


@njit(parallel=True, cache=True)
def _blend_kernel(a, b, alpha, out):  # pragma: no cover - numba-compiled
    """out[i] = (1-alpha)*a[i] + alpha*b[i] over 1-D float64 arrays.

    The per-element expression is EXACTLY the one numpy evaluates for
    ``(1-alpha)*a + alpha*b`` (two multiplies, one add, same order, no
    fastmath so no FMA contraction), so results are bit-identical to
    the numpy path for finite values — only the memory bandwidth
    changes (parallel streams instead of one).  NaN payload bits are
    not part of that promise.
    """
    beta = 1.0 - alpha
    for i in prange(a.shape[0]):
        out[i] = beta * a[i] + alpha * b[i]


def _blend(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """Allocate and return ``(1-alpha)*a + alpha*b``.

    Hot path of :meth:`VelocityField.lerp`: on temporal-interp runs this
    executes every tracking step over full-mesh arrays (23M elements on
    the global FESOM2 mesh ≈ 0.4 s/step with single-threaded numpy).
    Large C-contiguous float64 arrays go through a numba ``prange``
    kernel (memory-bandwidth bound → scales with threads); everything
    else falls back to the original numpy expression.  Both paths are
    bit-identical (same per-element arithmetic, ``fastmath`` off).
    """
    # Shape guard FIRST: the numba kernel has no bounds checking, so a
    # mismatched ``b`` would be read out of bounds where the old numpy
    # expression raised a broadcast ValueError (or legitimately
    # broadcast, e.g. (1, n) vs (n,)).  Falling back preserves that
    # numpy contract exactly.
    if (
        HAS_NUMBA
        and a.shape == b.shape
        and a.size >= _BLEND_MIN_SIZE
        and a.dtype == np.float64
        and b.dtype == np.float64
    ):
        if a.flags.c_contiguous and b.flags.c_contiguous:
            out = np.empty_like(a)
            _blend_kernel(a.reshape(-1), b.reshape(-1), float(alpha),
                          out.reshape(-1))
            return out
        # The field qualifies for the parallel kernel in every way EXCEPT
        # memory order — that is a reader handing back a Fortran-ordered /
        # strided array, and the silent serial fallback cost ~10x per step
        # on a real run before anyone noticed (2026-08).  Say so, once.
        if "blend" not in _CONTIG_WARNED:
            _CONTIG_WARNED.add("blend")
            warnings.warn(
                "whole-field lerp fell back to the serial numpy path: a "
                f"large float64 field (shape {a.shape}) is not "
                "C-contiguous.  A reader is producing Fortran-ordered or "
                "strided fields — wrap them in np.ascontiguousarray at "
                "the read site (see docs/performance.md).",
                RuntimeWarning, stacklevel=2,
            )
    # Compute in float64 even for float32 inputs (float32 velocity fields,
    # TODO 5.3): under NEP 50 a python-float * float32-array stays float32, so a
    # naive blend would round in float32 and diverge from the float64 path. The
    # blended field is a transient (the memory win is the cached float32
    # snapshots, not this result). asarray is a no-op for float64 inputs →
    # byte-identical default.
    return (1.0 - alpha) * np.asarray(a, dtype=np.float64) \
        + alpha * np.asarray(b, dtype=np.float64)


def _check_uv_vertical(u, w, uv_layout: str, uv_vertical: str) -> None:
    """The placement value and its shape contract (``VelocityField``).

    A 2-D field (``u`` 1-D) is always ``"layer"``.  ``"level"`` needs node
    layout, with or without ``w`` (level data sit on the nodes' levels);
    ``"layer_linear"`` takes node or element layout.  The rows are checked
    against ``w`` only for 3-D fields (``u`` 2-D and ``w`` present):
    ``"layer"`` and ``"layer_linear"`` need ``u.shape[0] == w.shape[0] - 1``
    (one row per layer), ``"level"`` needs ``u.shape[0] == w.shape[0]``
    (one row per level).
    """
    if uv_vertical not in _VALID_UV_VERTICAL:
        raise ValueError(
            f"VelocityField.uv_vertical must be one of {_VALID_UV_VERTICAL}, "
            f"got {uv_vertical!r}"
        )
    u_ndim = getattr(u, "ndim", None)
    if u_ndim == 1:
        if uv_vertical != "layer":
            raise ValueError(
                f"VelocityField: a 2-D field (u shape {u.shape}) must carry "
                f"uv_vertical='layer', got {uv_vertical!r}"
            )
        return
    if uv_vertical == "level" and uv_layout != "node":
        raise ValueError(
            f"VelocityField: uv_vertical={uv_vertical!r} needs "
            f"uv_layout='node', got {uv_layout!r}"
        )
    if u_ndim != 2 or w is None:
        return
    k_w = w.shape[0]
    want = k_w if uv_vertical == "level" else k_w - 1
    if u.shape[0] != want:
        rows = ("one per level, K" if uv_vertical == "level"
                else "one per layer, K-1")
        raise ValueError(
            f"VelocityField: uv_vertical={uv_vertical!r} needs u with "
            f"{want} rows ({rows}, K = {k_w} from w {w.shape}), got u "
            f"{u.shape}"
        )


def _require_same_uv_vertical(site: str, *placements: str) -> str:
    """Refuse mixed placements at a field-rebuild site; return the common one.

    ``site`` prefixes the message (the caller's name, e.g.
    ``"VelocityField.lerp"``); the message lists every placement in input
    order, so the odd one out is visible.
    """
    first = placements[0]
    if any(p != first for p in placements[1:]):
        raise ValueError(
            f"{site}: uv_vertical mismatch "
            + " vs ".join(repr(p) for p in placements)
        )
    return first


def _shared_token(*fields):
    """The parents' ``_dl_token`` when all hold the SAME object, else None."""
    tok = fields[0]._dl_token
    for f in fields[1:]:
        if f._dl_token is not tok:
            return None
    return tok


@dataclass
class VelocityField:
    """Velocity data for one timestep.

    Attributes
    ----------
    u : np.ndarray
        Eastward velocity on elements.
        Shape (E,) for 2D, (K-1, E) for 3D.
    v : np.ndarray
        Northward velocity on elements.
        Same shape as u.
    eta : np.ndarray
        Sea surface height at nodes, shape (N,).
    w : np.ndarray or None
        Vertical velocity at nodes, interface-based shape ``(K, N)``.
        None for 2D tracking.
    t : float or None
        Time value (seconds since reference, or model time index).
    kh_zonal, kh_meridional : np.ndarray or None
        Horizontal eddy diffusivity (m^2/s).  Both must be present
        together (or both ``None``).  Shape matches the chosen
        ``kh_layout``: ``(N,)`` for ``"node"`` or ``(E,)`` for
        ``"element"``.  No averaging is performed by
        readers — see ``docs/interpolation.md`` §5.
    kh_layout : str or None
        ``"node"`` or ``"element"``.  ``None`` only when both ``kh_*``
        fields are absent.
    uv_layout : str
        Mesh location of ``u`` / ``v``: ``"element"`` (the FESOM default,
        velocities on element centres) or ``"node"`` (velocities projected
        on nodes, e.g. FESOM2 ``unod`` / ``vnod``).  Readers auto-detect it
        from the variable dim names; defaults to ``"element"``.
    scalars : dict or None
        Optional dictionary of node-based scalar fields for trajectory
        sampling, keyed by variable name.  Each value is a NumPy array
        with shape ``(N,)`` for 2D/node scalars, or ``(K-1, N)`` / ``(K, N)``
        for 3D layer-based scalars. Lerped only when both sides of the
        interpolation carry the same keys and shapes.
    az : np.ndarray or None
        Optional vertical diffusivity Az, node-based at interfaces
        ``(K, N)`` (same layout as ``w``), read when the run configures
        ``diffusion.var_az`` (TODO 2.2b).  Lerped when both sides carry
        it; dropped to None otherwise.
    uv_vertical : str
        Placement of ``u`` / ``v`` in the vertical: ``"layer"`` (the
        default — layer centres, ``(K-1, ·)``, piecewise constant in
        depth), ``"level"`` (on the levels, ``(K, N)``, node layout,
        linear in depth) or ``"layer_linear"`` (layer centres
        ``(K-1, ·)``, node or element layout, evaluated linearly in
        depth).  The
        constructor checks a 3-D field's u rows against ``w`` (only when
        ``u`` is 2-D and ``w`` is present), so a dropped or wrong flag
        raises instead of mis-placing the data; a 2-D field (``u`` 1-D)
        is always ``"layer"``.  ``lerp``, ``LazyLerpField``, the PCHIP
        reconstruction and the shared cache carry it and refuse mixed
        placements.

    Notes
    -----
    ``_dl_token`` (not an ``__init__`` argument) is the record-check
    token of the depth-linear path; any rebuild through the constructor
    (``dataclasses.replace``, the shared cache) resets it to ``None``, and
    the blends keep it only when every parent holds the SAME object.
    """
    u: np.ndarray
    v: np.ndarray
    eta: np.ndarray
    w: Optional[np.ndarray] = None
    t: Optional[float] = None
    kh_zonal: Optional[np.ndarray] = None
    kh_meridional: Optional[np.ndarray] = None
    kh_layout: Optional[str] = None
    uv_layout: str = "element"  # "element" (centres) or "node" (FESOM2 unod/vnod)
    scalars: Optional[Dict[str, np.ndarray]] = None  # {varname: (N,) | (K-1, N) | (K, N)} node fields
    az: Optional[np.ndarray] = None  # vertical diffusivity (K, N) at interfaces
    uv_vertical: str = "layer"  # "layer" | "level" | "layer_linear"
    # record-check token of the depth-linear path; init=False, so every
    # rebuild through the constructor (dataclasses.replace) drops it
    _dl_token: object = field(default=None, init=False, repr=False,
                              compare=False)

    def __post_init__(self) -> None:
        # All-or-nothing: kh_zonal/kh_meridional/kh_layout must be
        # provided together.  This keeps every downstream check simple
        # ("either Kh is here or it isn't") and surfaces a partial
        # configuration as a config-time error instead of a NoneType
        # crash deep inside a Numba kernel.
        present = (
            self.kh_zonal is not None,
            self.kh_meridional is not None,
            self.kh_layout is not None,
        )
        if any(present) and not all(present):
            raise ValueError(
                "VelocityField: kh_zonal, kh_meridional, and kh_layout "
                "must all be set together (or all left as None); got "
                f"kh_zonal={'set' if present[0] else 'None'}, "
                f"kh_meridional={'set' if present[1] else 'None'}, "
                f"kh_layout={self.kh_layout!r}"
            )
        if self.kh_layout is not None:
            if self.kh_layout not in _VALID_KH_LAYOUTS:
                raise ValueError(
                    f"VelocityField.kh_layout must be one of "
                    f"{_VALID_KH_LAYOUTS}, got {self.kh_layout!r}"
                )
            if self.kh_zonal.shape != self.kh_meridional.shape:
                raise ValueError(
                    "VelocityField: kh_zonal and kh_meridional must have "
                    f"the same shape, got {self.kh_zonal.shape} vs "
                    f"{self.kh_meridional.shape}"
                )
        if self.uv_layout not in _VALID_UV_LAYOUTS:
            raise ValueError(
                f"uv_layout must be 'element' or 'node', got "
                f"{self.uv_layout!r}"
            )
        _check_uv_vertical(self.u, self.w, self.uv_layout, self.uv_vertical)
        if self.scalars is not None and len(self.scalars) == 0:
            self.scalars = None

    def lerp(self, other: VelocityField, alpha: float) -> VelocityField:
        """Linear interpolation between two velocity fields.

        Parameters
        ----------
        other : VelocityField
            The "next" velocity field (at alpha=1).
        alpha : float
            Interpolation weight in [0, 1].
            0 → self, 1 → other.

        Returns
        -------
        VelocityField
            Interpolated field.
        """
        # Every field carries u/v, so this check is unconditional — unlike
        # the kh_layout check below, which only fires when both sides have
        # Kh.  A layout mismatch means the two snapshots were read from
        # incompatible sources; lerping element- and node-based velocities
        # would silently mix mesh locations.
        if self.uv_layout != other.uv_layout:
            raise ValueError(
                "VelocityField.lerp: uv_layout mismatch "
                f"{self.uv_layout!r} vs {other.uv_layout!r}"
            )
        uv_vertical = _require_same_uv_vertical(
            "VelocityField.lerp", self.uv_vertical, other.uv_vertical)
        beta = 1.0 - alpha
        w_interp = None
        if self.w is not None and other.w is not None:
            w_interp = _blend(self.w, other.w, alpha)
        t_interp = None
        if self.t is not None and other.t is not None:
            t_interp = beta * self.t + alpha * other.t
        kh_zonal_interp = None
        kh_meridional_interp = None
        kh_layout_interp = None
        if self.kh_zonal is not None and other.kh_zonal is not None:
            # Both sides present: interpolate.  Shape mismatches are an
            # upstream bug (different layout or differently-loaded
            # fields), so raise instead of silently picking one side.
            if self.kh_zonal.shape != other.kh_zonal.shape:
                raise ValueError(
                    "VelocityField.lerp: kh_zonal shape mismatch "
                    f"{self.kh_zonal.shape} vs {other.kh_zonal.shape}"
                )
            if self.kh_layout != other.kh_layout:
                raise ValueError(
                    "VelocityField.lerp: kh_layout mismatch "
                    f"{self.kh_layout!r} vs {other.kh_layout!r}"
                )
            kh_zonal_interp = _blend(self.kh_zonal, other.kh_zonal, alpha)
            kh_meridional_interp = _blend(
                self.kh_meridional, other.kh_meridional, alpha
            )
            kh_layout_interp = self.kh_layout
        scalars_interp = None
        if self.scalars is not None and other.scalars is not None:
            if set(self.scalars) != set(other.scalars):
                raise ValueError(
                    "VelocityField.lerp: scalars key mismatch "
                    f"{sorted(self.scalars)} vs {sorted(other.scalars)}"
                )
            scalars_interp = {}
            for k in self.scalars:
                if self.scalars[k].shape != other.scalars[k].shape:
                    raise ValueError(
                        f"VelocityField.lerp: scalars[{k!r}] shape mismatch "
                        f"{self.scalars[k].shape} vs {other.scalars[k].shape}"
                    )
                scalars_interp[k] = _blend(
                    self.scalars[k], other.scalars[k], alpha)
        az_interp = None
        if self.az is not None and other.az is not None:
            if self.az.shape != other.az.shape:
                raise ValueError(
                    "VelocityField.lerp: az shape mismatch "
                    f"{self.az.shape} vs {other.az.shape}"
                )
            az_interp = _blend(self.az, other.az, alpha)
        out = VelocityField(
            u=_blend(self.u, other.u, alpha),
            v=_blend(self.v, other.v, alpha),
            eta=_blend(self.eta, other.eta, alpha),
            w=w_interp,
            t=t_interp,
            kh_zonal=kh_zonal_interp,
            kh_meridional=kh_meridional_interp,
            kh_layout=kh_layout_interp,
            uv_layout=self.uv_layout,
            scalars=scalars_interp,
            az=az_interp,
            uv_vertical=uv_vertical,
        )
        out._dl_token = _shared_token(self, other)
        return out


class LazyLerpField:
    """A ``VelocityField`` view over ``(vel_a, vel_b, alpha)`` that blends
    each field on FIRST ACCESS and caches the result.

    Temporal-interp tracking builds a blended field every step; most
    steps only ever touch a subset of the fields (the fused 2D tinterp
    kernels bypass even ``u``/``v`` by blending the raw snapshots inline
    — see ``_run_fused_2d_tinterp``).  Materializing lazily makes each
    consumer pay only for what it reads: beaching reads ``.eta`` (one
    node-array blend), record sampling reads ``.scalars``/``.eta``, the
    unfused/3D paths read everything and behave exactly as before.

    Every blended array comes from the same :func:`_blend` as
    ``VelocityField.lerp``, so values are bit-identical to the eager
    path.  ``.w`` answers ``None`` without any blend when either parent
    lacks it — the ``_is_3d()`` probe stays free in 2D runs.
    """

    #: array fields blended on demand (mirrors VelocityField.lerp)
    _LAZY_FIELDS = ("u", "v", "eta", "w", "kh_zonal", "kh_meridional", "az")

    def __init__(self, vel_a: VelocityField, vel_b: VelocityField,
                 alpha: float) -> None:
        if vel_a.uv_layout != vel_b.uv_layout:
            raise ValueError(
                "LazyLerpField: uv_layout mismatch "
                f"{vel_a.uv_layout!r} vs {vel_b.uv_layout!r}")
        uv_vertical = _require_same_uv_vertical(
            "LazyLerpField", vel_a.uv_vertical, vel_b.uv_vertical)
        # Fail-fast metadata validation mirroring VelocityField.lerp —
        # all shape/layout/key checks are metadata-only (no blends), so
        # incompatible snapshots surface at payload-build time with the
        # same errors the eager path raised, not as a later cryptic
        # broadcast failure — or, worse, as an unchecked out-of-bounds
        # read inside a fused numba kernel that indexes BOTH raw parents
        # (the kernels have no bounds checking, so the core u/v/eta/w
        # shape guard below is a hard safety requirement of exposing
        # .vel_a/.vel_b to them).
        for fname in ("u", "v", "eta"):
            fa, fb = getattr(vel_a, fname), getattr(vel_b, fname)
            if fa.shape != fb.shape:
                raise ValueError(
                    f"LazyLerpField: {fname} shape mismatch "
                    f"{fa.shape} vs {fb.shape}")
        if (vel_a.w is not None and vel_b.w is not None
                and vel_a.w.shape != vel_b.w.shape):
            raise ValueError(
                "LazyLerpField: w shape mismatch "
                f"{vel_a.w.shape} vs {vel_b.w.shape}")
        if (vel_a.az is not None and vel_b.az is not None
                and vel_a.az.shape != vel_b.az.shape):
            raise ValueError(
                "LazyLerpField: az shape mismatch "
                f"{vel_a.az.shape} vs {vel_b.az.shape}")
        both_kh = vel_a.kh_zonal is not None and vel_b.kh_zonal is not None
        if both_kh:
            if vel_a.kh_zonal.shape != vel_b.kh_zonal.shape:
                raise ValueError(
                    "LazyLerpField: kh_zonal shape mismatch "
                    f"{vel_a.kh_zonal.shape} vs {vel_b.kh_zonal.shape}")
            if vel_a.kh_layout != vel_b.kh_layout:
                raise ValueError(
                    "LazyLerpField: kh_layout mismatch "
                    f"{vel_a.kh_layout!r} vs {vel_b.kh_layout!r}")
        sa, sb = vel_a.scalars, vel_b.scalars
        if sa is not None and sb is not None:
            if set(sa) != set(sb):
                raise ValueError(
                    "LazyLerpField: scalars key mismatch "
                    f"{sorted(sa)} vs {sorted(sb)}")
            for k in sa:
                if sa[k].shape != sb[k].shape:
                    raise ValueError(
                        f"LazyLerpField: scalars[{k!r}] shape mismatch "
                        f"{sa[k].shape} vs {sb[k].shape}")
        self.vel_a = vel_a
        self.vel_b = vel_b
        self.alpha = float(alpha)
        # Plain (non-lazy) metadata — identical rules to VelocityField.lerp:
        # kh_layout survives only when BOTH sides carry Kh.
        self.uv_layout = vel_a.uv_layout
        # Plain attributes, never lazy (__getattr__ raises for unknown
        # names): the placement, and the record-check token — kept only
        # when both parents hold the SAME token object.
        self.uv_vertical = uv_vertical
        self._dl_token = _shared_token(vel_a, vel_b)
        self.kh_layout = vel_a.kh_layout if both_kh else None
        self.t = (None if vel_a.t is None or vel_b.t is None
                  else (1.0 - self.alpha) * vel_a.t + self.alpha * vel_b.t)
        self._cache: Dict[str, object] = {}

    def __getattr__(self, name: str):
        # Only called when normal attribute lookup fails -> lazy fields.
        # _cache is read via __dict__ to avoid recursing here if an
        # attribute is probed before __init__ finishes (e.g. by
        # copy/unpickle protocols) — missing cache means "not initialized",
        # which must surface as AttributeError, not KeyError.
        if "_cache" not in self.__dict__:
            raise AttributeError(name)
        if name in LazyLerpField._LAZY_FIELDS:
            cache = self.__dict__["_cache"]
            if name not in cache:
                a = getattr(self.vel_a, name)
                b = getattr(self.vel_b, name)
                if a is None or b is None:
                    cache[name] = None          # e.g. w in 2D: stays free
                elif name.startswith("kh_") and self.kh_layout is None:
                    cache[name] = None          # one-sided Kh -> dropped
                else:
                    cache[name] = _blend(a, b, self.alpha)
            # Promote to a real instance attribute: subsequent reads skip
            # __getattr__ entirely (it sits in per-particle Python loops
            # on the unfused stage paths).
            self.__dict__[name] = cache[name]
            return cache[name]
        if name == "scalars":
            cache = self.__dict__["_cache"]
            if "scalars" not in cache:
                sa, sb = self.vel_a.scalars, self.vel_b.scalars
                if sa is None or sb is None:
                    cache["scalars"] = None
                else:
                    if set(sa) != set(sb):
                        raise ValueError(
                            "LazyLerpField: scalars key mismatch "
                            f"{sorted(sa)} vs {sorted(sb)}")
                    cache["scalars"] = {
                        k: _blend(sa[k], sb[k], self.alpha) for k in sa}
            self.__dict__["scalars"] = cache["scalars"]
            return cache["scalars"]
        raise AttributeError(name)

    def lerp(self, other: VelocityField, alpha: float) -> VelocityField:
        """Materialize, then delegate.

        Keeps ``_pick_kh_eval_vel`` / ``_make_stage_vels`` (unfused RK,
        sigma) working unchanged when the step field is lazy.
        """
        return self.materialize().lerp(other, alpha)

    def materialize(self) -> VelocityField:
        """Return the full eager ``VelocityField`` for this blend."""
        out = VelocityField(
            u=self.u, v=self.v, eta=self.eta, w=self.w, t=self.t,
            kh_zonal=self.kh_zonal, kh_meridional=self.kh_meridional,
            kh_layout=self.kh_layout, uv_layout=self.uv_layout,
            scalars=self.scalars, az=self.az, uv_vertical=self.uv_vertical,
        )
        out._dl_token = self._dl_token
        return out


__all__ = ["VelocityField", "LazyLerpField"]
