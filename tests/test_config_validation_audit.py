"""Configuration values that used to pass the load-time checks silently.

Audit 2026-10-04 (F-040, F-041, F-044, F-045, F-063, F-064): each case below
was accepted at load and then either changed the physics without a word
(``time_dir: 2.0`` doubled every displacement), was ignored (``Cycle`` ran as
``stop``), failed only after the output file had been created (a misspelled
integrator), or was mangled (a quoted window split into characters).
"""
import textwrap

import pytest

from kelp.core.config import (
    DataConfig, MeshConfig, ModeConfig, NumericsConfig, TimeConfig, load_config,
)


# --- mode.time_dir -----------------------------------------------------------

@pytest.mark.parametrize("bad", [2.0, 0.0, -0.5, 1.5, True, "forward"])
def test_time_dir_other_than_plus_minus_one_is_refused(bad):
    """The value multiplies every interpolated velocity while the clock
    advances by its sign only: 2.0 doubled the displacement per step."""
    with pytest.raises(ValueError, match="time_dir"):
        ModeConfig(time_dir=bad)


@pytest.mark.parametrize("ok, expected", [(1, 1.0), (-1, -1.0),
                                          (1.0, 1.0), (-1.0, -1.0)])
def test_time_dir_accepts_unit_values(ok, expected):
    td = ModeConfig(time_dir=ok).time_dir
    assert td == expected and isinstance(td, float)


# --- numerics.integrator / w_interp ------------------------------------------

def test_unknown_integrator_is_refused_at_load():
    """Checked only inside the run before: after the output file had been
    created (and an existing file at that path overwritten)."""
    with pytest.raises(ValueError, match="integrator"):
        NumericsConfig(integrator="rk3")


def test_integrator_names_match_the_registry():
    from kelp.core.config import _VALID_INTEGRATORS
    from kelp.numerics.integrators import INTEGRATORS
    assert set(_VALID_INTEGRATORS) == set(INTEGRATORS)


def test_unknown_w_interp_is_refused_at_load():
    with pytest.raises(ValueError, match="w_interp"):
        NumericsConfig(w_interp="bilinear")


def test_w_interp_names_match_the_vertical_methods():
    from kelp.core.config import _VALID_W_INTERP
    from kelp.numerics.vertical import _W_METHOD_MAP
    assert set(_VALID_W_INTERP) == set(_W_METHOD_MAP)


# --- time.on_exhausted / data.time_averaging ---------------------------------

@pytest.mark.parametrize("bad", ["Cycle", "cycles", "wrap", ""])
def test_on_exhausted_is_validated_at_load(bad):
    """Validated only by resolve_time_params, which the start_date +
    duration_days path skips: 'Cycle' then ran as 'stop'."""
    with pytest.raises(ValueError, match="on_exhausted"):
        TimeConfig(on_exhausted=bad)


def test_time_averaging_is_validated_at_load():
    with pytest.raises(ValueError, match="time_averaging"):
        DataConfig(time_averaging="means")
    for ok in (None, "mean", "instant"):
        DataConfig(time_averaging=ok)


def test_duration_days_without_start_date_is_refused():
    """duration_days is converted only together with start_date; alone it
    was dropped without a word (and n_steps silently took over)."""
    with pytest.raises(ValueError, match="duration_days"):
        TimeConfig(duration_days=30.0, n_steps=10)
    TimeConfig(duration_days=30.0, start_date="2020-01-01")


def test_start_date_with_end_time_idx_alone_is_refused(tmp_path):
    """With start_date set and neither n_steps nor duration_days, the run
    took n_steps = end_time_idx - start_time_idx, ignoring the record that
    start_date resolved to (bug audit 2026-11-04).  Refused at load, so
    ``kelp info`` and ``kelp run`` both refuse it."""
    with pytest.raises(ValueError) as exc:
        TimeConfig(start_date="2020-01-03", end_time_idx=10)
    msg = str(exc.value)
    for part in ("time.start_date", "time.end_time_idx",
                 "time.duration_days", "time.n_steps"):
        assert part in msg, (part, msg)
    p = tmp_path / "c.yaml"
    p.write_text(textwrap.dedent("""\
        mode:
        numerics:
          dt: 3600
        mesh:
          type: fesomc
          meshpath: m.nc
        data:
          ncinput: d.nc
        time:
          start_date: "2020-01-03"
          end_time_idx: 10
    """))
    with pytest.raises(ValueError, match="time.end_time_idx"):
        load_config(str(p))
    # n_steps or duration_days set: they decide, as before
    assert TimeConfig(start_date="2020-01-03", n_steps=5).n_steps == 5
    TimeConfig(start_date="2020-01-03", end_time_idx=10, n_steps=5)
    TimeConfig(start_date="2020-01-03", end_time_idx=10, duration_days=2.0)
    # end_time_idx without start_date: the legacy window, unchanged
    TimeConfig(start_time_idx=2, end_time_idx=10)


# --- strings, booleans and bounds --------------------------------------------

def test_quoted_periodic_window_is_refused():
    """A quoted '12' was iterated character by character -> (1.0, 2.0)."""
    with pytest.raises(ValueError, match="periodic_lon"):
        NumericsConfig(periodic_lon="12")


def test_quoted_crop_bbox_is_refused():
    with pytest.raises(ValueError, match="crop_bbox"):
        MeshConfig(crop_bbox="1234")


def test_boolean_grid_hash_cell_is_refused():
    with pytest.raises(ValueError, match="grid_hash_cell_size"):
        MeshConfig(grid_hash_cell_size=True)


def test_grid_hash_cell_at_or_above_45_degrees_is_refused():
    """The pin accepted 60 deg, built the hash, and the next rebuild
    (e.g. seed filtering) raised on its own (0, 45) bound."""
    with pytest.raises(ValueError, match="grid_hash_cell_size"):
        MeshConfig(grid_hash_cell_size=60.0)
    assert MeshConfig(grid_hash_cell_size=44.9).grid_hash_cell_size == 44.9


def test_set_grid_hash_cell_applies_the_same_bound():
    import numpy as np
    from kelp.core.mesh import Mesh, set_grid_hash_cell, build_node_to_elems
    lon = np.array([0.0, 1.0, 0.0]); lat = np.array([0.0, 0.0, 1.0])
    en = np.array([[0, 1, 2]], dtype=np.int32)
    n2e, cnt = build_node_to_elems(3, en)
    mesh = Mesh(name="tri", nodes_lon=lon, nodes_lat=lat, depth=np.ones(3), elem_nodes=en,
                elem_center_lon=np.array([1 / 3]), elem_center_lat=np.array([1 / 3]),
                node_to_elems=n2e, node_to_elems_count=cnt)
    with pytest.raises(ValueError, match="45"):
        set_grid_hash_cell(mesh, 60.0)


# --- FESOM-C 2-D reduction ---------------------------------------------------

def test_sigma_layer_and_sigma_mean_together_are_refused():
    """The element u/v reducer took the mean, the scalar/node reducer the
    layer: the run tracked the mean and sampled T/S at the layer."""
    with pytest.raises(ValueError, match="sigma_layer"):
        ModeConfig(sigma_layer=1, sigma_mean=True)


# --- load_config -------------------------------------------------------------

def test_empty_section_loads_with_its_defaults(tmp_path):
    """`mode:` with nothing under it is YAML null: load_config raised
    AttributeError ('NoneType' object has no attribute 'items')."""
    p = tmp_path / "c.yaml"
    p.write_text(textwrap.dedent("""\
        mode:
        numerics:
          dt: 3600
        mesh:
          type: fesom2
          meshpath: m.nc
        data:
          ncinput: d.nc
        time:
          n_steps: 2
    """))
    cfg = load_config(str(p))
    assert cfg.mode.time_dir == 1.0 and cfg.mode.Fdrift3D is False


@pytest.mark.parametrize("bad", [-1.0, 0.0, float("inf"), float("nan"), True])
def test_duration_days_must_be_a_positive_finite_number(bad):
    with pytest.raises(ValueError, match="duration_days"):
        TimeConfig(duration_days=bad, start_date="2020-01-01")


def test_numpy_bool_grid_hash_cell_is_refused():
    import numpy as np
    with pytest.raises(ValueError, match="grid_hash_cell_size"):
        MeshConfig(grid_hash_cell_size=np.bool_(True))


def test_set_grid_hash_cell_refuses_booleans():
    import numpy as np
    from kelp.core.mesh import Mesh, set_grid_hash_cell, build_node_to_elems
    lon = np.array([0.0, 1.0, 0.0]); lat = np.array([0.0, 0.0, 1.0])
    en = np.array([[0, 1, 2]], dtype=np.int32)
    n2e, cnt = build_node_to_elems(3, en)
    mesh = Mesh(name="tri", nodes_lon=lon, nodes_lat=lat, depth=np.ones(3),
                elem_nodes=en, elem_center_lon=np.array([1 / 3]),
                elem_center_lat=np.array([1 / 3]), node_to_elems=n2e,
                node_to_elems_count=cnt)
    for flag in (True, np.bool_(True)):
        with pytest.raises(ValueError, match="cell"):
            set_grid_hash_cell(mesh, flag)


@pytest.mark.parametrize("body", ["mode: false\n", "mode: []\n", "mode: 3\n"])
def test_a_section_that_is_not_a_mapping_is_refused(tmp_path, body):
    """Only an EMPTY section (YAML null) means 'defaults'."""
    p = tmp_path / "c.yaml"
    p.write_text(body + "numerics: {dt: 3600}\nmesh: {type: fesom2}\n"
                 "data: {ncinput: d.nc}\ntime: {n_steps: 2}\n")
    with pytest.raises(ValueError, match="mode"):
        load_config(str(p))


@pytest.mark.parametrize("bad", [False, [], 3])
def test_nested_diffusion_and_beaching_blocks_must_be_mappings(bad):
    with pytest.raises(ValueError, match="diffusion"):
        NumericsConfig(diffusion=bad)
    with pytest.raises(ValueError, match="beaching"):
        NumericsConfig(beaching=bad)
