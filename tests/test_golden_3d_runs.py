"""Golden 3-D trajectories: every case of ``tests/_gen_golden_3d.py`` re-run
and compared with the files generated on ``main`` before the depth-linear
u/v work.

Two modes, chosen once per golden directory by its ``ENV.json`` fingerprint
(``tests/_bitwise.golden_env_matches``):

* ``BITWISE`` (JIT set) / ``BITWISE_NOJIT`` (numba-off set) True — this
  environment generated the goldens (same Python, numpy, numba, llvmlite,
  machine, libc, numpy AVX512F dispatch, NUMBA_DISABLE_JIT): every array is
  compared bit for bit (dtype, shape, raw bytes).  This is the proof that a
  change leaves existing runs unchanged.
* False — another environment (e.g. CI with pip wheels), where the last bit
  of a transcendental function may differ: integer, bool and string data
  stay exact, float arrays must match in dtype and shape and to
  ``rtol = atol = 1e-9`` (NaN where the golden has NaN).

A failure means an existing 3-D run changed.  If that change is intended,
regenerate both sets (the generator's docstring has the commands) and say
why in the commit.  Attributes compare as strings; ``kelp_version`` names
the commit, so it is checked against ``tracking._kelp_version()`` instead.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("healpy")                 # the node and HEALPix cases

from kelp import tracking  # noqa: E402
from tests._bitwise import assert_bitwise, golden_env_matches  # noqa: E402
from tests._gen_golden_3d import CASES, run_case  # noqa: E402

_HERE = Path(__file__).resolve().parent
_GEN = _HERE / "_gen_golden_3d.py"
_VERSION = "attr__kelp_version"
GOLDEN_DIR = _HERE / "data" / "golden_3d"
GOLDEN_NOJIT_DIR = _HERE / "data" / "golden_3d_nojit"
#: True: compare the JIT set bit for bit; False: the tolerance mode
BITWISE = golden_env_matches(GOLDEN_DIR)
#: the same for the numba-off set (its subprocess runs NUMBA_DISABLE_JIT=1)
BITWISE_NOJIT = golden_env_matches(GOLDEN_NOJIT_DIR, disable_jit=True)
RTOL = ATOL = 1e-9


def _load(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def _compare(got, golden, bitwise):
    exp = _load(golden)
    assert sorted(got) == sorted(exp), (
        f"keys differ: missing {sorted(set(exp) - set(got))}, "
        f"extra {sorted(set(got) - set(exp))}")
    for k in sorted(exp):
        a, b = got[k], exp[k]
        if k == _VERSION:
            assert str(a) == tracking._kelp_version()
        elif k.startswith(("attr__", "dims__")):
            assert str(a) == str(b), f"{k}: {str(a)!r} != {str(b)!r}"
        elif bitwise or b.dtype.kind not in "fc":
            assert_bitwise(a, b, k)
        else:
            assert a.dtype == b.dtype, f"{k}: dtype {a.dtype} != {b.dtype}"
            assert a.shape == b.shape, f"{k}: shape {a.shape} != {b.shape}"
            np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL,
                                       equal_nan=True, err_msg=k)


@pytest.mark.parametrize("name", sorted(CASES))
def test_golden_bitwise(name):
    _compare(run_case(name), GOLDEN_DIR / f"{name}.npz", BITWISE)


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(CASES))
def test_golden_bitwise_numba_off(name, tmp_path):
    """The pure-Python kernels (``NUMBA_DISABLE_JIT=1``, a fresh
    interpreter) against their own golden set."""
    env = dict(os.environ, NUMBA_DISABLE_JIT="1", TMPDIR=str(tmp_path))
    proc = subprocess.run(
        [sys.executable, str(_GEN), "--write", str(tmp_path), "--case", name],
        env=env, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    _compare(_load(tmp_path / f"{name}.npz"),
             GOLDEN_NOJIT_DIR / f"{name}.npz", BITWISE_NOJIT)


def test_env_gate(tmp_path):
    """The fingerprint gate: a matching ENV.json opens the bitwise mode, an
    altered or missing one closes it."""
    import json
    from tests._bitwise import ENV_FILE, env_fingerprint
    fp = env_fingerprint()
    (tmp_path / ENV_FILE).write_text(json.dumps(fp))
    assert golden_env_matches(tmp_path)
    assert not golden_env_matches(tmp_path, disable_jit=not
                                  fp["numba_disable_jit"])
    (tmp_path / ENV_FILE).write_text(json.dumps(dict(fp, numpy="0.0")))
    assert not golden_env_matches(tmp_path)
    (tmp_path / ENV_FILE).unlink()
    assert not golden_env_matches(tmp_path)


def test_compare_modes():
    """One float moved by 1 ULP fails the bitwise mode and passes the
    tolerance mode; moved by 1e-6 relative it fails both; an integer
    change fails both."""
    golden = GOLDEN_DIR / "healpix_3d_rk2.npz"
    exp = _load(golden)
    ulp = dict(exp, lon=exp["lon"].copy())
    ulp["lon"][0, 0] = np.nextafter(ulp["lon"][0, 0], np.inf)
    with pytest.raises(AssertionError):
        _compare(ulp, golden, bitwise=True)
    _compare(ulp, golden, bitwise=False)
    far = dict(exp, lon=exp["lon"] * (1.0 + 1e-6))
    for bitwise in (True, False):
        with pytest.raises(AssertionError):
            _compare(far, golden, bitwise=bitwise)
    st = dict(exp, status=exp["status"].copy())
    st["status"][0, 0] += 1
    with pytest.raises(AssertionError):
        _compare(st, golden, bitwise=False)
