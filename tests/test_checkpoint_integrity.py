"""A half-written checkpoint must never load as valid (review 2026-09-22, F02).

``write_checkpoint`` overwrote the restart arrays one by one while
``restart_valid`` stayed 1 from the previous checkpoint (and was even set
before the optional fields), so an exception or a killed process between two
arrays left a file that ``load_restart`` accepted: lon from the new step, lat
from the old one.  The embedded checkpoint now clears ``restart_valid`` before
touching the payload and sets it only after every field is written; the
standalone restart is written to a temporary file and renamed into place.
"""
import os

import numpy as np
import pytest

from kelp.core.particles import make_state
from kelp.writers import TrajectoryWriter, load_restart, write_restart_file


class _DiesOnLat:
    """A ParticleState stand-in that raises when the writer reaches ``lat``
    — i.e. after ``restart_lon`` has been overwritten.  ``reads_before``
    accesses of ``lat`` are allowed first: the embedded checkpoint reads
    every array once to check its length BEFORE it clears the validity flag
    (output format 2), so the kill must come on the payload read."""

    def __init__(self, good, reads_before=0):
        self._good = good
        self._left = reads_before

    def __getattr__(self, name):
        if name == "lat":
            if self._left <= 0:
                raise RuntimeError("process killed mid-checkpoint")
            self._left -= 1
        return getattr(self._good, name)


def _states():
    a = make_state(np.array([5.0, 5.5]), np.array([5.0, 5.5]))
    b = make_state(np.array([6.0, 6.5]), np.array([6.0, 6.5]))
    return a, b


@pytest.mark.parametrize("ext,flush_every", [(".zarr", 0), (".nc", 0),
                                             (".nc", 2)])
def test_interrupted_checkpoint_is_rejected(tmp_path, ext, flush_every):
    a, b = _states()
    path = str(tmp_path / f"traj{ext}")
    kw = {"flush_every": flush_every} if ext == ".nc" else {}
    w = TrajectoryWriter(path, 2, **kw)
    w.open()
    w.write_checkpoint(a, 0, 0)
    with pytest.raises(RuntimeError, match="killed"):
        w.write_checkpoint(_DiesOnLat(b, reads_before=1), 1, 1)
    w.close()
    with pytest.raises(ValueError, match="No valid restart"):
        load_restart(path)


@pytest.mark.parametrize("ext", [".zarr", ".nc"])
def test_complete_checkpoint_still_loads(tmp_path, ext):
    a, b = _states()
    path = str(tmp_path / f"ok{ext}")
    w = TrajectoryWriter(path, 2)
    w.open()
    w.write_checkpoint(a, 0, 0)
    w.write_checkpoint(b, 1, 1)
    w.close()
    r = load_restart(path)
    assert r["step_i"] == 1
    np.testing.assert_array_equal(r["state"].lon, b.lon)
    np.testing.assert_array_equal(r["state"].lat, b.lat)


def test_standalone_restart_keeps_the_previous_file(tmp_path):
    a, b = _states()
    path = str(tmp_path / "restart_step001.nc")
    write_restart_file(path, a, 0, 0)
    with pytest.raises(RuntimeError, match="killed"):
        write_restart_file(path, _DiesOnLat(b), 1, 1)
    r = load_restart(path)
    assert r["step_i"] == 0
    np.testing.assert_array_equal(r["state"].lon, a.lon)
    leftovers = [f for f in os.listdir(tmp_path) if f != "restart_step001.nc"]
    assert leftovers == []
