"""Inline tinterp fast path: fused 2D kernels consume the raw snapshot
pair from a LazyLerpField and must (a) never materialize u/v and
(b) stay bit-identical to the materialized-field path.
"""
from __future__ import annotations

import numpy as np
import pytest

from kelp.core.io import LazyLerpField, VelocityField
from kelp.core.mesh import Mesh, build_node_to_elems, build_search
from kelp.core.particles import make_state
from kelp.numerics.integrators import euler_step, rk2_step, rk4_step


def _make_mesh():
    """Small flat patch (same construction as test_integrators_time_aware)."""
    span = 0.15
    nx = 5
    xs = np.linspace(-span, span, nx)
    ys = np.linspace(-span, span, nx)
    gx, gy = np.meshgrid(xs, ys)
    lon = gx.ravel().astype(np.float64)
    lat = gy.ravel().astype(np.float64)
    n_nodes = lon.shape[0]
    depth = np.full(n_nodes, 200.0, dtype=np.float64)

    elems = []
    for j in range(nx - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            elems.append([n0, n0 + 1, n0 + nx])
            elems.append([n0 + 1, n0 + nx + 1, n0 + nx])
    elem_nodes = np.array(elems, dtype=np.int32)
    lon_c = np.array([lon[e].mean() for e in elem_nodes], dtype=np.float64)
    lat_c = np.array([lat[e].mean() for e in elem_nodes], dtype=np.float64)
    n2e, n2e_count = build_node_to_elems(n_nodes, elem_nodes)

    mesh = Mesh(name="t", nodes_lon=lon, nodes_lat=lat, depth=depth,
                elem_nodes=elem_nodes,
                elem_center_lon=lon_c, elem_center_lat=lat_c,
                node_to_elems=n2e, node_to_elems_count=n2e_count,
                sigma=np.array([]), vertical_coord="zlevel")
    build_search(mesh)
    return mesh


def _fields(mesh, seed):
    rng = np.random.default_rng(seed)
    E, N = mesh.n_elements, mesh.n_nodes
    return VelocityField(u=rng.uniform(0.05, 0.4, E),
                         v=rng.uniform(-0.2, 0.2, E),
                         eta=np.zeros(N))


def _state():
    return make_state(np.array([-0.05, 0.04, 0.0]),
                      np.array([0.03, -0.06, 0.0]))


@pytest.mark.parametrize("step_fn", [rk2_step, rk4_step])
@pytest.mark.parametrize("alpha0", [0.0, 0.31, 0.97])
def test_lazy_pair_bitwise_equals_materialized(step_fn, alpha0):
    """The raw-pair fast path must reproduce the materialized path bitwise."""
    mesh = _make_mesh()
    va, vb = _fields(mesh, 1), _fields(mesh, 2)
    state = _state()
    dt, dt_to_next = 300.0, 1000.0

    eager = va.lerp(vb, alpha0)                      # old behavior
    s_ref = step_fn(state, eager, mesh, dt, 1.0,
                    vel_next=vb, dt_to_next=dt_to_next)

    lazy = LazyLerpField(va, vb, alpha0)             # new fast path
    s_new = step_fn(state, lazy, mesh, dt, 1.0,
                    vel_next=vb, dt_to_next=dt_to_next)

    assert np.array_equal(s_ref.lon, s_new.lon)
    assert np.array_equal(s_ref.lat, s_new.lat)
    assert np.array_equal(s_ref.status, s_new.status)
    assert np.array_equal(s_ref.u, s_new.u)
    assert np.array_equal(s_ref.v, s_new.v)


@pytest.mark.parametrize("step_fn", [rk2_step, rk4_step])
def test_fused_pair_path_does_not_materialize_uv(step_fn, monkeypatch):
    """The fused tinterp dispatch must read the RAW pair — zero blends."""
    import kelp.core.io as lio
    mesh = _make_mesh()
    va, vb = _fields(mesh, 3), _fields(mesh, 4)
    state = _state()

    blended = []
    real = lio._blend
    monkeypatch.setattr(lio, "_blend",
                        lambda a, b, al: blended.append(a.size) or real(a, b, al))
    lazy = LazyLerpField(va, vb, 0.4)
    step_fn(state, lazy, mesh, 300.0, 1.0, vel_next=vb, dt_to_next=1000.0)
    assert blended == []        # raw pair went straight into the kernel


@pytest.mark.parametrize("step_fn", [rk2_step, rk4_step])
def test_time_aware_validation_still_fires_on_lazy(step_fn):
    """Shape validation must work (without materializing) on lazy fields."""
    mesh = _make_mesh()
    va, vb = _fields(mesh, 5), _fields(mesh, 6)
    bad_next = VelocityField(u=np.zeros(3), v=np.zeros(3), eta=np.zeros(2))
    lazy = LazyLerpField(va, vb, 0.5)
    with pytest.raises(ValueError):
        step_fn(_state(), lazy, mesh, 300.0, 1.0,
                vel_next=bad_next, dt_to_next=1000.0)


@pytest.mark.parametrize("step_fn", [rk2_step, rk4_step])
@pytest.mark.parametrize("alpha0", [0.2, 0.85])
def test_lazy_pair_bitwise_backward(step_fn, alpha0):
    """BACKWARD: vel_next is the bracket's LOWER snapshot (A), so the
    bracket pair and the stage target differ — the regression that the
    forward-only tests missed."""
    mesh = _make_mesh()
    va, vb = _fields(mesh, 7), _fields(mesh, 8)
    state = _state()
    dt, dt_to_next = 300.0, 800.0     # dt_to_next = sim_time - ta

    eager = va.lerp(vb, alpha0)
    s_ref = step_fn(state, eager, mesh, dt, -1.0,
                    vel_next=va, dt_to_next=dt_to_next)

    lazy = LazyLerpField(va, vb, alpha0)
    s_new = step_fn(state, lazy, mesh, dt, -1.0,
                    vel_next=va, dt_to_next=dt_to_next)

    assert np.array_equal(s_ref.lon, s_new.lon)
    assert np.array_equal(s_ref.lat, s_new.lat)
    assert np.array_equal(s_ref.u, s_new.u)
    assert np.array_equal(s_ref.v, s_new.v)


def test_euler_lazy_bitwise_equals_materialized():
    """Euler ignores vel_next; its velocity IS the start-of-step field —
    the lazy fast path must match the materialized field bitwise."""
    mesh = _make_mesh()
    va, vb = _fields(mesh, 9), _fields(mesh, 10)
    state = _state()
    s_ref = euler_step(state, va.lerp(vb, 0.31), mesh, 300.0, 1.0)
    s_new = euler_step(state, LazyLerpField(va, vb, 0.31), mesh, 300.0, 1.0)
    assert np.array_equal(s_ref.lon, s_new.lon)
    assert np.array_equal(s_ref.lat, s_new.lat)
    assert np.array_equal(s_ref.status, s_new.status)
    assert np.array_equal(s_ref.u, s_new.u)
    assert np.array_equal(s_ref.v, s_new.v)


def test_euler_lazy_does_not_materialize_uv(monkeypatch):
    import kelp.core.io as lio
    mesh = _make_mesh()
    va, vb = _fields(mesh, 11), _fields(mesh, 12)
    state = _state()
    blended = []
    real = lio._blend
    monkeypatch.setattr(lio, "_blend",
                        lambda a, b, al: blended.append(a.size) or real(a, b, al))
    euler_step(state, LazyLerpField(va, vb, 0.6), mesh, 300.0, 1.0)
    assert blended == []


# ---------------------------------------------------------------------------
# End-to-end: lazy payload vs eagerly materialized payload, bit-identical
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("time_dir,start_idx", [(1.0, 0), (-1.0, 4)])
def test_end_to_end_trajectories_bit_identical(tmp_path, monkeypatch,
                                               time_dir, start_idx):
    """run_single_realization with the LazyLerpField payload must produce
    trajectories bit-identical to a payload that eagerly materializes."""
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    import kelp.tracking as tr
    from kelp.core.config import ModeConfig, NumericsConfig
    from kelp.readers import get_reader, extract_time_info
    from test_temporal_interp import _make_fesomc_timed

    nc_path = _make_fesomc_timed(tmp_path, n_time=5, dt_seconds=3600.0,
                                 u_val=0.4, v_val=0.1)
    reader = get_reader("fesomc")
    mesh = reader.load_mesh(type("", (), {"meshpath": str(nc_path)})())
    ds = reader.open_dataset(str(nc_path))
    file_times, file_dt = extract_time_info(ds)

    mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=time_dir)
    numerics = NumericsConfig(dt=900.0, integrator="rk2", output_freq=1)
    def init():
        return make_state(np.array([4.8, 5.2]), np.array([5.1, 4.9]))

    orig_payload = tr._build_vel_payload

    def eager_payload(*a, **kw):
        vel, vn, dtn, fi = orig_payload(*a, **kw)
        if isinstance(vel, LazyLerpField):
            vel = vel.materialize()
        return vel, vn, dtn, fi

    def run(eager):
        if eager:
            monkeypatch.setattr(tr, "_build_vel_payload", eager_payload)
        else:
            monkeypatch.setattr(tr, "_build_vel_payload", orig_payload)
        return tr.run_single_realization(
            mesh, ds, reader, init(), mode, numerics,
            start_time_idx=start_idx, n_steps=12,
            temporal_interp=True, file_times=file_times, file_dt=file_dt,
        )

    rec_e = run(eager=True)
    rec_l = run(eager=False)
    arr_e, arr_l = rec_e.to_arrays(), rec_l.to_arrays()
    assert np.array_equal(arr_e["lon"], arr_l["lon"])
    assert np.array_equal(arr_e["lat"], arr_l["lat"])
    assert np.array_equal(arr_e["status"], arr_l["status"])


@pytest.mark.parametrize("step_fn,wrapper", [
    ("euler_step", "_run_fused_euler_2d_pair"),
    ("rk2_step", "_run_fused_2d_tinterp"),
    ("rk4_step", "_run_fused_2d_tinterp"),
])
def test_pair_fast_path_is_actually_taken(step_fn, wrapper, monkeypatch):
    """Positive dispatch assertion: a refactor that silently reroutes the
    lazy field to a slower (but bit-identical) path must fail a test."""
    import kelp.numerics.integrators as integ
    mesh = _make_mesh()
    va, vb = _fields(mesh, 13), _fields(mesh, 14)
    state = _state()

    calls = []
    real = getattr(integ, wrapper)

    def spy(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(integ, wrapper, spy)
    fn = getattr(integ, step_fn)
    kwargs = {} if step_fn == "euler_step" else {
        "vel_next": vb, "dt_to_next": 1000.0}
    fn(state, LazyLerpField(va, vb, 0.4), mesh, 300.0, 1.0, **kwargs)
    assert calls, f"{wrapper} was not taken for the lazy field"


def test_end_to_end_barycentric_tinterp_bit_identical(tmp_path, monkeypatch):
    """Unfused tinterp path (barycentric u/v): stage 0 is the lazy field,
    later stages go through LazyLerpField.lerp -> materialize."""
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    import kelp.tracking as tr
    from kelp.core.config import ModeConfig, NumericsConfig
    from kelp.readers import get_reader, extract_time_info
    from test_temporal_interp import _make_fesomc_timed

    nc_path = _make_fesomc_timed(tmp_path, n_time=5, dt_seconds=3600.0,
                                 u_val=0.4, v_val=0.1)
    reader = get_reader("fesomc")
    mesh = reader.load_mesh(type("", (), {"meshpath": str(nc_path)})())
    ds = reader.open_dataset(str(nc_path))
    file_times, file_dt = extract_time_info(ds)

    mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0)
    numerics = NumericsConfig(dt=900.0, integrator="rk2", output_freq=1,
                              uv_interp="barycentric")

    orig_payload = tr._build_vel_payload

    def eager_payload(*a, **kw):
        vel, vn, dtn, fi = orig_payload(*a, **kw)
        if isinstance(vel, LazyLerpField):
            vel = vel.materialize()
        return vel, vn, dtn, fi

    def run(eager):
        monkeypatch.setattr(tr, "_build_vel_payload",
                            eager_payload if eager else orig_payload)
        init = make_state(np.array([4.8, 5.2]), np.array([5.1, 4.9]))
        return tr.run_single_realization(
            mesh, ds, reader, init, mode, numerics,
            start_time_idx=0, n_steps=12,
            temporal_interp=True, file_times=file_times, file_dt=file_dt,
        )

    arr_e = run(eager=True).to_arrays()
    arr_l = run(eager=False).to_arrays()
    assert np.array_equal(arr_e["lon"], arr_l["lon"])
    assert np.array_equal(arr_e["lat"], arr_l["lat"])


def test_alpha0_zero_branch_expression_semantics():
    """Documents WHY the tinterp core branches on alpha0 == 0.0: the
    nested blend computes 0.0*B, which differs from the old single
    blend for non-finite B (0*Inf = NaN where Inf used to propagate).
    The kernel's alpha0==0 branch keeps the exact old expression; this
    pins the expression-level divergence the branch protects against."""
    A, B, T, alpha = 0.3, np.inf, np.inf, 0.5
    old_single = (1 - alpha) * A + alpha * T                  # Inf
    nested = (1 - alpha) * (1.0 * A + 0.0 * B) + alpha * T    # NaN
    assert np.isinf(old_single)
    assert np.isnan(nested)


def test_alpha0_zero_branch_in_real_kernel_ignores_bracket_b():
    """The kernel-level pin for the alpha0==0 branch: with alpha0=0 the
    unused bracket-B slot must be completely ignored — Inf there must
    not leak into the result (a deleted branch would NaN it via 0*Inf)."""
    from kelp.numerics.interpolate import _interp_leastsq_core_tinterp
    mesh = _make_mesh()
    rng = np.random.default_rng(60)
    E = mesh.n_elements
    u_a = rng.uniform(0.1, 0.4, E)
    v_a = rng.uniform(-0.2, 0.2, E)
    u_t = rng.uniform(0.1, 0.4, E)
    v_t = rng.uniform(-0.2, 0.2, E)
    u_inf = np.full(E, np.inf)

    node = int(mesh.n_nodes // 2)        # interior node, rich stencil
    args = (mesh.node_to_elems_count, mesh.node_to_elems,
            mesh.elem_center_lon, mesh.elem_center_lat)

    U_inf_b, V_inf_b, ok1 = _interp_leastsq_core_tinterp(
        0.0, 0.0, node, 1.0, 0.0, 0.5, *args,
        u_a, v_a, u_inf, u_inf, u_t, v_t)
    U_ref, V_ref, ok2 = _interp_leastsq_core_tinterp(
        0.0, 0.0, node, 1.0, 0.0, 0.5, *args,
        u_a, v_a, u_a, v_a, u_t, v_t)

    assert ok1 and ok2
    assert np.isfinite(U_inf_b) and np.isfinite(V_inf_b)
    assert U_inf_b == U_ref and V_inf_b == V_ref
