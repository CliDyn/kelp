"""Bit-for-bit array comparison for golden tests, and the environment gate.

Two arrays are bit-identical when dtype, shape and raw bytes agree
(``a.tobytes() == b.tobytes()``): signed zeros and NaN payloads count, so
``np.array_equal`` (NaN-blind, ``-0.0 == 0.0``) is not enough.  On a
mismatch ``np.testing.assert_array_equal`` supplies the readable
diagnostic; a difference it cannot see (signed zero, NaN payload) is
reported by the first differing element's bit pattern.

Bits are reproducible only in the environment that wrote them: another
numpy (its SIMD dispatch, e.g. SVML on AVX-512 hosts), numba/llvmlite or
libc can move the last bit of a transcendental function.  A golden
directory therefore carries ``ENV.json`` (:func:`env_fingerprint` at
generation time, no host names or paths); :func:`golden_env_matches` tells
a test whether it runs in that environment.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

ENV_FILE = "ENV.json"


def assert_bitwise(a, b, name: str = "array") -> None:
    """Assert that ``a`` and ``b`` are bit-identical (dtype, shape, bytes).

    Parameters
    ----------
    a, b : array_like
        Actual and expected values.
    name : str
        Label used in the failure message.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.dtype != b.dtype:
        raise AssertionError(f"{name}: dtype {a.dtype} != {b.dtype}")
    if a.shape != b.shape:
        raise AssertionError(f"{name}: shape {a.shape} != {b.shape}")
    if a.tobytes() == b.tobytes():
        return
    np.testing.assert_array_equal(
        a, b, err_msg=f"{name}: not bit-identical", strict=True)
    # equal as values, different as bits: signed zero or NaN payload
    fa = np.ascontiguousarray(a).reshape(-1)
    fb = np.ascontiguousarray(b).reshape(-1)
    ua = fa.view(np.uint8).reshape(a.size, -1)
    ub = fb.view(np.uint8).reshape(b.size, -1)
    bad = np.flatnonzero((ua != ub).any(axis=1))
    i = int(bad[0])
    idx = tuple(int(j) for j in np.unravel_index(i, a.shape))
    # the bytes reversed (little-endian host) = the bit pattern
    raise AssertionError(
        f"{name}: {bad.size} element(s) equal in value but not in bits "
        f"(signed zero or NaN payload); first at {idx}: "
        f"{fa[i].item()!r} (0x{ua[i][::-1].tobytes().hex()}) vs "
        f"{fb[i].item()!r} (0x{ub[i][::-1].tobytes().hex()})")


def _version(module: str) -> Optional[str]:
    try:
        return str(__import__(module).__version__)
    except Exception:
        return None


def _numpy_avx512f() -> Optional[bool]:
    """Whether numpy dispatches AVX512F on this CPU (None: unknown)."""
    for mod in ("numpy._core._multiarray_umath",
                "numpy.core._multiarray_umath"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                feats = __import__(mod, fromlist=["__cpu_features__"]
                                   ).__cpu_features__
            return bool(feats.get("AVX512F", False))
        except Exception:
            continue
    return None


def _numba_disable_jit() -> bool:
    try:
        from numba.core import config
        return bool(config.DISABLE_JIT)
    except Exception:
        return os.environ.get("NUMBA_DISABLE_JIT", "0").strip() not in (
            "", "0")


def env_fingerprint(disable_jit: Optional[bool] = None) -> dict:
    """The environment that decides the bits of a golden run.

    ``disable_jit`` overrides the running process's ``NUMBA_DISABLE_JIT``
    (a test comparing the output of a numba-off subprocess)."""
    return {
        "python": "%d.%d" % sys.version_info[:2],
        "numpy": _version("numpy"),
        "numba": _version("numba"),
        "llvmlite": _version("llvmlite"),
        "machine": platform.machine(),
        "libc": list(platform.libc_ver()),
        "numpy_avx512f": _numpy_avx512f(),
        "numba_disable_jit": (_numba_disable_jit() if disable_jit is None
                              else bool(disable_jit)),
    }


def write_env_fingerprint(dirname) -> None:
    """Write :func:`env_fingerprint` of this process to ``dirname/ENV.json``."""
    Path(dirname, ENV_FILE).write_text(
        json.dumps(env_fingerprint(), indent=2, sort_keys=True) + "\n")


def golden_env_matches(dirname, *,
                       disable_jit: Optional[bool] = None) -> bool:
    """True when ``dirname/ENV.json`` equals this environment's fingerprint
    (bitwise comparison is meaningful); False otherwise, also when the file
    is missing or unreadable.  Never raises."""
    try:
        stored = json.loads(Path(dirname, ENV_FILE).read_text())
        running = json.loads(json.dumps(env_fingerprint(disable_jit)))
        return stored == running
    except Exception:
        return False
