"""Tests for kelp.core.config."""
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from kelp.core.config import (
    AppConfig, ModeConfig, NumericsConfig, DiffusionConfig,
    MeshConfig, DataConfig, TimeConfig, load_config,
    resolve_time_params,
)


def _write_yaml(content: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".yml")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


FULL_YAML = """\
mode:
  Fdrift2D: true
  Fdrift3D: false
  time_dir: 1.0
  Vpuls: false
numerics:
  dt: 600.0
  output_freq: 2
  Dmin: 1.0
  z0b_min: 0.05
  Dch: 0.5
  integrator: rk2
  chunk: 10
mesh:
  type: fesomc
  meshpath: some/path
data:
  ncinput: data/simulation.nc
  inicloud: data/particles.csv
  outstem: output/drift
time:
  start_time_idx: 10
  end_time_idx: 200
  num_runs: 3
  run_interval: 50
"""


class TestLoadConfig:
    def test_full_config(self):
        path = _write_yaml(FULL_YAML)
        try:
            cfg = load_config(path)
            assert isinstance(cfg, AppConfig)
            assert cfg.mode.Fdrift2D is True
            assert cfg.mode.time_dir == 1.0
            assert cfg.numerics.dt == 600.0
            assert cfg.numerics.integrator == "rk2"
            assert cfg.mesh.type == "fesomc"
            assert cfg.data.ncinput == str(
                Path(path).resolve().parent / "data" / "simulation.nc")
            assert cfg.time.start_time_idx == 10
            assert cfg.time.end_time_idx == 200
            assert cfg.time.num_runs == 3
        finally:
            os.unlink(path)

    def test_missing_section_raises(self):
        yaml = "mode:\n  Fdrift2D: true\nnumerics:\n  dt: 100\n"
        path = _write_yaml(yaml)
        try:
            with pytest.raises(ValueError, match="Config missing sections"):
                load_config(path)
        finally:
            os.unlink(path)

    def test_absolute_path_kept(self, tmp_path):
        """An absolute path is taken as written; a relative one is joined
        to the config file's directory."""
        data = str(tmp_path.resolve() / "elsewhere" / "sim.nc")
        cfg_path = tmp_path / "config.yml"
        cfg_path.write_text(FULL_YAML.replace("data/simulation.nc", data))
        cfg = load_config(str(cfg_path))
        assert cfg.data.ncinput == data
        assert cfg.mesh.meshpath == str(tmp_path.resolve() / "some" / "path")

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent/config.yml")

    def test_unknown_keys_ignored(self):
        yaml = FULL_YAML + "  extra_unknown_field: 42\n"
        path = _write_yaml(yaml)
        try:
            cfg = load_config(path)
            assert cfg.time.end_time_idx == 200
        finally:
            os.unlink(path)


class TestDefaults:
    def test_mode_defaults(self):
        m = ModeConfig()
        assert m.Fdrift2D is True
        assert m.Fdrift3D is False
        assert m.time_dir == 1.0

    def test_numerics_defaults(self):
        n = NumericsConfig()
        assert n.dt is None  # auto-detect from file
        assert n.integrator == "euler"

    def test_time_defaults(self):
        t = TimeConfig()
        assert t.num_runs == 1
        assert t.run_interval == 0
        assert t.n_steps is None
        assert t.end_time_idx is None
        assert t.on_exhausted == "stop"
        assert t.start_date is None
        assert t.duration_days is None


class TestResolveTimeParams:
    def test_n_steps(self):
        cfg = TimeConfig(start_time_idx=5, n_steps=20)
        start, n = resolve_time_params(cfg)
        assert start == 5
        assert n == 20

    def test_end_time_idx(self):
        cfg = TimeConfig(start_time_idx=5, end_time_idx=15)
        start, n = resolve_time_params(cfg)
        assert start == 5
        assert n == 10

    def test_n_steps_priority(self):
        """n_steps takes priority over end_time_idx."""
        cfg = TimeConfig(start_time_idx=0, n_steps=20, end_time_idx=100)
        start, n = resolve_time_params(cfg)
        assert n == 20

    def test_neither_raises(self):
        cfg = TimeConfig(start_time_idx=0)
        with pytest.raises(ValueError, match="n_steps.*end_time_idx"):
            resolve_time_params(cfg)

    def test_on_exhausted_validation(self):
        # refused when the section is built (audit 2026-10-04): the
        # start_date + duration_days path never calls resolve_time_params
        with pytest.raises(ValueError, match="on_exhausted"):
            TimeConfig(n_steps=10, on_exhausted="invalid")

    def test_on_exhausted_cycle(self):
        cfg = TimeConfig(n_steps=10, on_exhausted="cycle")
        start, n = resolve_time_params(cfg)
        assert n == 10

    def test_yaml_with_n_steps(self):
        yaml_content = """\
mode:
  Fdrift2D: true
numerics:
  dt: 600.0
mesh:
  type: fesomc
  meshpath: some/path
data:
  ncinput: data/sim.nc
  inicloud: data/p.csv
time:
  start_time_idx: 5
  n_steps: 50
  on_exhausted: cycle
  num_runs: 2
"""
        path = _write_yaml(yaml_content)
        try:
            cfg = load_config(path)
            assert cfg.time.n_steps == 50
            assert cfg.time.on_exhausted == "cycle"
            assert cfg.time.end_time_idx is None
            start, n = resolve_time_params(cfg.time)
            assert start == 5
            assert n == 50
        finally:
            os.unlink(path)


class TestDiffusionConfig:
    def test_defaults_disabled(self):
        d = DiffusionConfig()
        assert d.enabled is False
        assert d.scheme == "uniform"
        assert d.kh_zonal == 0.0
        assert d.kh_meridional == 0.0
        assert d.var_kh_zonal is None
        assert d.var_kh_meridional is None
        assert d.dres_m is None
        assert d.seed == 0
        assert d.composition == "split"
        assert d.kh_eval_position == "end"

    def test_kh_eval_position_valid_values(self):
        DiffusionConfig(kh_eval_position="start")
        DiffusionConfig(kh_eval_position="end")

    def test_kh_eval_position_invalid_raises(self):
        with pytest.raises(ValueError, match="kh_eval_position"):
            DiffusionConfig(kh_eval_position="middle")

    def test_numerics_default_has_disabled_diffusion(self):
        n = NumericsConfig()
        assert isinstance(n.diffusion, DiffusionConfig)
        assert n.diffusion.enabled is False

    def test_numerics_accepts_dict_diffusion(self):
        n = NumericsConfig(diffusion={"enabled": True, "scheme": "uniform", "kh_zonal": 10.0})
        assert isinstance(n.diffusion, DiffusionConfig)
        assert n.diffusion.enabled is True
        assert n.diffusion.kh_zonal == 10.0

    def test_numerics_dict_drops_unknown_keys(self):
        n = NumericsConfig(diffusion={"enabled": True, "bogus_field": 99})
        assert n.diffusion.enabled is True
        assert not hasattr(n.diffusion, "bogus_field")

    def test_invalid_scheme_raises(self):
        with pytest.raises(ValueError, match="scheme"):
            DiffusionConfig(scheme="bogus")

    def test_invalid_composition_raises(self):
        with pytest.raises(ValueError, match="composition"):
            DiffusionConfig(composition="weird")

    def test_negative_kh_zonal_raises(self):
        with pytest.raises(ValueError, match="kh_zonal"):
            DiffusionConfig(kh_zonal=-1.0)

    def test_negative_kh_meridional_raises(self):
        with pytest.raises(ValueError, match="kh_zonal|kh_meridional"):
            DiffusionConfig(kh_meridional=-2.0)

    def test_dres_m_zero_or_negative_raises(self):
        with pytest.raises(ValueError, match="dres_m"):
            DiffusionConfig(dres_m=0.0)
        with pytest.raises(ValueError, match="dres_m"):
            DiffusionConfig(dres_m=-5.0)

    def test_dres_m_none_allowed(self):
        d = DiffusionConfig(dres_m=None)
        assert d.dres_m is None

    def test_negative_seed_raises(self):
        with pytest.raises(ValueError, match="seed"):
            DiffusionConfig(seed=-1)

    def test_seed_overflow_raises(self):
        with pytest.raises(ValueError, match="seed"):
            DiffusionConfig(seed=1 << 64)

    def test_seed_zero_and_max_allowed(self):
        DiffusionConfig(seed=0)
        DiffusionConfig(seed=(1 << 64) - 1)

    def test_yaml_with_diffusion_section(self):
        yaml_content = """\
mode:
  Fdrift2D: true
numerics:
  dt: 600.0
  diffusion:
    enabled: true
    scheme: uniform
    kh_zonal: 10.0
    kh_meridional: 5.0
    seed: 42
mesh:
  type: fesomc
  meshpath: some/path
data:
  ncinput: data/sim.nc
  inicloud: data/p.csv
time:
  start_time_idx: 0
  n_steps: 5
"""
        path = _write_yaml(yaml_content)
        try:
            cfg = load_config(path)
            assert isinstance(cfg.numerics.diffusion, DiffusionConfig)
            assert cfg.numerics.diffusion.enabled is True
            assert cfg.numerics.diffusion.scheme == "uniform"
            assert cfg.numerics.diffusion.kh_zonal == 10.0
            assert cfg.numerics.diffusion.kh_meridional == 5.0
            assert cfg.numerics.diffusion.seed == 42
        finally:
            os.unlink(path)

    def test_yaml_without_diffusion_section_is_disabled(self):
        """A config without ``numerics.diffusion`` defaults to disabled."""
        path = _write_yaml(FULL_YAML)
        try:
            cfg = load_config(path)
            assert isinstance(cfg.numerics.diffusion, DiffusionConfig)
            assert cfg.numerics.diffusion.enabled is False
            assert cfg.numerics.diffusion.kh_zonal == 0.0
        finally:
            os.unlink(path)

    def test_yaml_invalid_scheme_raises(self):
        yaml_content = """\
mode:
  Fdrift2D: true
numerics:
  dt: 600.0
  diffusion:
    enabled: true
    scheme: bogus
mesh:
  type: fesomc
  meshpath: some/path
data:
  ncinput: data/sim.nc
  inicloud: data/p.csv
time:
  start_time_idx: 0
  n_steps: 5
"""
        path = _write_yaml(yaml_content)
        try:
            with pytest.raises(ValueError, match="scheme"):
                load_config(path)
        finally:
            os.unlink(path)


class TestSamplingConfig:
    def test_defaults(self):
        """Config built without sampling keys has sensible defaults."""
        yaml_content = """\
mode:
  Fdrift2D: true
numerics:
  dt: 600.0
mesh:
  type: fesomc
  meshpath: some/path
data:
  ncinput: data/sim.nc
  inicloud: data/p.csv
time:
  start_time_idx: 0
  n_steps: 5
"""
        path = _write_yaml(yaml_content)
        try:
            cfg = load_config(path)
            assert cfg.mode.sample_vars == []
            assert cfg.numerics.sample_interp == "leastsq"
        finally:
            os.unlink(path)

    def test_sample_vars_parsed(self):
        """sample_vars from list in YAML are stored as a list."""
        yaml_content = """\
mode:
  Fdrift2D: true
  sample_vars:
    - temp
    - salt
numerics:
  dt: 600.0
mesh:
  type: fesomc
  meshpath: some/path
data:
  ncinput: data/sim.nc
  inicloud: data/p.csv
time:
  start_time_idx: 0
  n_steps: 5
"""
        path = _write_yaml(yaml_content)
        try:
            cfg = load_config(path)
            assert cfg.mode.sample_vars == ["temp", "salt"]
        finally:
            os.unlink(path)

    def test_sample_vars_single_string_coerced(self):
        """Single string sample_vars is coerced to a list."""
        yaml_content = """\
mode:
  Fdrift2D: true
  sample_vars: temp
numerics:
  dt: 600.0
mesh:
  type: fesomc
  meshpath: some/path
data:
  ncinput: data/sim.nc
  inicloud: data/p.csv
time:
  start_time_idx: 0
  n_steps: 5
"""
        path = _write_yaml(yaml_content)
        try:
            cfg = load_config(path)
            assert cfg.mode.sample_vars == ["temp"]
        finally:
            os.unlink(path)

    def test_sample_interp_validated(self):
        """Invalid sample_interp value raises ValueError."""
        yaml_content = """\
mode:
  Fdrift2D: true
numerics:
  dt: 600.0
  sample_interp: cubic
mesh:
  type: fesomc
  meshpath: some/path
data:
  ncinput: data/sim.nc
  inicloud: data/p.csv
time:
  start_time_idx: 0
  n_steps: 5
"""
        path = _write_yaml(yaml_content)
        try:
            with pytest.raises(ValueError, match="sample_interp"):
                load_config(path)
        finally:
            os.unlink(path)

    def test_sample_vars_null_handled(self):
        """sample_vars: null in YAML is converted to empty list."""
        yaml_content = """\
mode:
  Fdrift2D: true
  sample_vars: null
numerics:
  dt: 600.0
mesh:
  type: fesomc
  meshpath: some/path
data:
  ncinput: data/sim.nc
  inicloud: data/p.csv
time:
  start_time_idx: 0
  n_steps: 5
"""
        path = _write_yaml(yaml_content)
        try:
            cfg = load_config(path)
            assert cfg.mode.sample_vars == []
        finally:
            os.unlink(path)


def _deep_merge(base: dict, extra: dict) -> dict:
    """Recursively merge *extra* into *base* (in place) and return it.

    Nested dicts are merged key-by-key so that e.g.
    ``{"numerics": {"beaching": {...}}}`` does not clobber the existing
    ``numerics`` section (a shallow ``dict.update`` would be a bug).
    """
    for key, value in extra.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _load_minimal(tmp_path, extra: dict) -> AppConfig:
    """Build a minimal config, recursively merge *extra*, dump + load.

    The base mirrors the minimal YAML used throughout this file (all five
    required sections).  *extra* is deep-merged so nested overrides such as
    ``numerics.beaching`` augment rather than replace the base section.
    """
    base = {
        "mode": {"Fdrift2D": True},
        "numerics": {"dt": 600.0},
        "mesh": {"type": "fesomc", "meshpath": "some/path"},
        "data": {"ncinput": "data/sim.nc", "inicloud": "data/p.csv"},
        "time": {"start_time_idx": 0, "n_steps": 5},
    }
    merged = _deep_merge(base, extra)
    path = tmp_path / "config.yml"
    path.write_text(yaml.dump(merged))
    return load_config(str(path))


class TestBeachingConfig:
    def test_defaults_disabled(self, tmp_path):
        cfg = _load_minimal(tmp_path, {})
        b = cfg.numerics.beaching
        assert b.enabled is False
        assert b.criteria == ["depth"]
        assert b.depth_min == 0.3 and b.use_eta is False
        assert b.resuspension == "never"
        assert b.dry_mask_var is None and b.zone_depth is None

    def test_dict_to_dataclass(self, tmp_path):
        cfg = _load_minimal(tmp_path, {"numerics": {"beaching": {
            "enabled": True, "criteria": ["depth", "stochastic"],
            "t_beach": 3600.0}}})
        assert cfg.numerics.beaching.enabled is True
        assert cfg.numerics.beaching.criteria == ["depth", "stochastic"]

    def test_unknown_criterion_raises(self, tmp_path):
        with pytest.raises(ValueError, match="criteria"):
            _load_minimal(tmp_path, {"numerics": {"beaching": {
                "enabled": True, "criteria": ["depth", "tsunami"]}}})

    def test_stochastic_requires_t_beach(self, tmp_path):
        with pytest.raises(ValueError, match="t_beach"):
            _load_minimal(tmp_path, {"numerics": {"beaching": {
                "enabled": True, "criteria": ["stochastic"],
                "t_beach": None}}})

    def test_stochastic_resuspension_requires_t_resuspend(self, tmp_path):
        with pytest.raises(ValueError, match="t_resuspend"):
            _load_minimal(tmp_path, {"numerics": {"beaching": {
                "enabled": True, "resuspension": "stochastic"}}})

    def test_bad_resuspension_mode_raises(self, tmp_path):
        with pytest.raises(ValueError, match="resuspension"):
            _load_minimal(tmp_path, {"numerics": {"beaching": {
                "enabled": True, "resuspension": "sometimes"}}})

    def test_depth_min_positive(self, tmp_path):
        with pytest.raises(ValueError, match="depth_min"):
            _load_minimal(tmp_path, {"numerics": {"beaching": {
                "enabled": True, "depth_min": -1.0}}})

    def test_seed_falls_back_to_diffusion(self, tmp_path):
        cfg = _load_minimal(tmp_path, {"numerics": {
            "diffusion": {"seed": 777},
            "beaching": {"enabled": True}}})
        assert cfg.numerics.beaching.seed is None  # raw value stays None
        # resolution happens in tracking (Task 4); config only stores.

    def test_disabled_skips_validation(self, tmp_path):
        # enabled=False short-circuits: nonsense values load fine
        cfg = _load_minimal(tmp_path, {"numerics": {"beaching": {
            "enabled": False, "depth_min": -5.0,
            "criteria": ["tsunami"]}}})
        assert cfg.numerics.beaching.enabled is False

    def test_negative_timescales_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="t_beach"):
            _load_minimal(tmp_path, {"numerics": {"beaching": {
                "enabled": True, "criteria": ["stochastic"],
                "t_beach": -100.0}}})
        with pytest.raises(ValueError, match="t_resuspend"):
            _load_minimal(tmp_path, {"numerics": {"beaching": {
                "enabled": True, "resuspension": "auto_or_stochastic",
                "t_resuspend": 0.0}}})

    def test_min_wet_steps_validated(self, tmp_path):
        with pytest.raises(ValueError, match="min_wet_steps"):
            _load_minimal(tmp_path, {"numerics": {"beaching": {
                "enabled": True, "min_wet_steps": 0}}})


def test_temporal_reconstruction_default_and_validation():
    assert NumericsConfig().temporal_reconstruction == "linear"
    assert NumericsConfig(
        temporal_reconstruction="pchip").temporal_reconstruction == "pchip"
    with pytest.raises(ValueError, match="temporal_reconstruction"):
        NumericsConfig(temporal_reconstruction="cubic")
