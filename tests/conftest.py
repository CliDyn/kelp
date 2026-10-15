"""Ensure the kelp package is importable from tests.

Normally ``pip install -e .`` (editable install) handles this,
but the fallback below also lets the tests run on a clean checkout
without installation.
"""
import os
import sys
from pathlib import Path

# Cap numba's thread pool for the test suite. Must happen BEFORE numba is
# first imported (conftest runs before any test module). Numba's default is
# all logical CPUs — 256 on an HPC login node — and the per-call OpenMP
# fork/barrier+spin-wait on the tiny test arrays then dominates the runtime
# by an order of magnitude. setdefault → an explicit NUMBA_NUM_THREADS in
# the environment still wins.
os.environ.setdefault("NUMBA_NUM_THREADS", str(min(8, os.cpu_count() or 8)))

# tests/conftest.py is at repo/tests/, package source is at repo/src/kelp/.
_src = str(Path(__file__).resolve().parents[1] / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
