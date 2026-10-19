"""kelp CLI entrypoint.

Usage::

    kelp run config.yaml          # run tracking from a YAML config
    kelp info config.yaml         # show mesh stats and time range
    kelp --version
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .. import __version__


def _cmd_run(args: argparse.Namespace) -> int:
    """Execute ``kelp run config.yaml``."""
    from ..core.config import load_config
    from ..tracking import run_tracking

    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        print(f"error: config file not found: {config_path}", file=sys.stderr)
        return 2

    config = load_config(str(config_path))
    output = args.output if args.output else None
    if args.seed is not None:
        # CLI override of the diffusion RNG seed.  Use
        # ``dataclasses.replace`` so the new ``DiffusionConfig`` runs
        # ``__post_init__`` validation (e.g. negative seeds raise).
        from dataclasses import replace
        config.numerics.diffusion = replace(
            config.numerics.diffusion, seed=args.seed,
        )
    if args.flush_every is not None:
        # CLI override of numerics.output_flush_every.  ``replace`` re-runs
        # NumericsConfig validation (rejects negative / non-int).
        from dataclasses import replace
        config.numerics = replace(
            config.numerics, output_flush_every=args.flush_every,
        )
    run_tracking(
        config,
        output_path=output,
        verbose=args.verbose,
        checkpoint_freq=args.checkpoint_freq,
        temporal_interp=args.temporal_interp,
    )
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    """Execute ``kelp info config.yaml``."""
    from ..core.config import load_config
    # the helpers a run uses too, from the registry: info never imports the
    # tracking loop and its integrators
    from ..readers import (get_reader, _describe_record_times,
                           _prepare_dataset, _time_option_kwargs)

    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        print(f"error: config file not found: {config_path}", file=sys.stderr)
        return 2

    config = load_config(str(config_path))
    try:
        reader = get_reader(config.mesh.type)
    except KeyError as exc:
        print(f"error: unknown mesh type {config.mesh.type!r}: {exc}",
              file=sys.stderr)
        return 1
    try:
        # the stamp options reach open_dataset as in a run (fesom14 only)
        time_kw = _time_option_kwargs(config)
    except (TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Config: {config_path}")
    print(f"  Mesh type: {config.mesh.type}")
    print(f"  Mesh path: {config.mesh.meshpath}")
    print(f"  Data path: {config.data.ncinput}")

    # review finding #2 (round 2): the mesh-load path used to bubble
    # tracebacks while the dataset path silently returned 0.  Now both
    # surface to stderr + return 1, so CI / scripts get a clean failure
    # signal regardless of which half of the config is broken.
    try:
        import inspect as _inspect
        if "mode_cfg" in _inspect.signature(reader.load_mesh).parameters:
            mesh = reader.load_mesh(config.mesh, mode_cfg=config.mode)
        else:
            mesh = reader.load_mesh(config.mesh)
    except Exception as exc:
        print(f"error: could not load mesh {config.mesh.meshpath!r}: {exc}",
              file=sys.stderr)
        return 1
    print(f"  Mesh stats:")
    print(f"    nodes:    {mesh.n_nodes}")
    print(f"    elements: {mesh.n_elements}")
    print(f"    sigma:    {mesh.sigma.shape[0]} interfaces"
          f" ({mesh.vertical_coord})")
    # the grid-hash cell the run would use (explicit pin, or the rule under
    # the exact-nearest flag), with its window statistics
    try:
        from ..core.mesh import (set_exact_nearest, set_grid_hash_cell,
                                 apply_grid_hash_cell_rule, describe_grid_hash)
        set_exact_nearest(mesh, bool(getattr(config.mesh, "exact_nearest", False)))
        if getattr(config.mesh, "grid_hash_cell_size", None) is not None:
            set_grid_hash_cell(mesh, config.mesh.grid_hash_cell_size)
            apply_grid_hash_cell_rule(mesh)              # pinned: labels it explicit
        else:
            apply_grid_hash_cell_rule(
                mesh, getattr(config.mesh, "grid_hash_cell_rule", "occupancy"))
        print(f"    {describe_grid_hash(mesh)}")
    except Exception as exc:                       # stats only — never fail `info` on them
        print(f"    grid hash: (not evaluated: {exc})")

    try:
        # the separate v/w paths, as a run passes them
        ds = reader.open_dataset(config.data.ncinput,
                                 v_path=config.data.ncinput_v or None,
                                 w_path=config.data.ncinput_w or None,
                                 mode_cfg=config.mode, **time_kw)
    except Exception as exc:
        print(f"error: could not open dataset {config.data.ncinput!r}: {exc}",
              file=sys.stderr)
        return 1

    try:
        # the reader's mesh-relative checks, as a run makes them (a reader
        # without the hook: nothing happens)
        try:
            _prepare_dataset(reader, mesh, ds)
        except Exception as exc:
            print(f"error: the dataset {config.data.ncinput!r} does not "
                  f"fit the mesh: {exc}", file=sys.stderr)
            return 1
        if "time" in ds.dims:
            n_time = ds.sizes["time"]
            print(f"  Time range: {n_time} steps")
            if "time" in ds.coords:
                t0 = ds["time"].values[0]
                t1 = ds["time"].values[-1]
                print(f"    {t0}  →  {t1}")
            record_times = _describe_record_times(
                ds, getattr(config.data, "time_offset", None))
            if record_times is not None:
                print(f"    record times {record_times}")
    finally:
        ds.close()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point — installed as ``kelp`` console script."""
    parser = argparse.ArgumentParser(
        prog="kelp",
        description="Lagrangian particle tracker for unstructured ocean meshes",
    )
    parser.add_argument(
        "--version", action="version",
        version=f"kelp {__version__}",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run tracking from a YAML config")
    p_run.add_argument("config", help="path to YAML config file")
    p_run.add_argument(
        "-o", "--output", help="trajectory output path (.nc or .zarr)"
    )
    p_run.add_argument(
        "-v", "--verbose", action="store_true", help="verbose progress",
    )
    p_run.add_argument(
        "--checkpoint-freq", type=int, default=None,
        help="write a restart checkpoint every N steps "
             "(default: use config; 0 = disabled)",
    )
    # Use ``store_const`` (default None) instead of ``store_true``
    # (default False) so the YAML's ``numerics.temporal_interp`` is
    # respected when the flag is absent.  ``run_tracking`` only falls
    # back to the config value when this kwarg is None.
    p_run.add_argument(
        "--temporal-interp", action="store_const", const=True, default=None,
        help="enable linear temporal interpolation between file snapshots "
             "(default: use config)",
    )
    p_run.add_argument(
        "--seed", type=int, default=None,
        help="override numerics.diffusion.seed (RNG seed for diffusion)",
    )
    p_run.add_argument(
        "--flush-every", type=int, default=None,
        help="NetCDF only: buffer N record points, then open/append/close so "
             "the file is readable mid-run (default: use config; 0 = keep "
             "file open the whole run)",
    )
    p_run.set_defaults(func=_cmd_run)

    p_info = sub.add_parser(
        "info", help="show mesh stats and time range for a config",
    )
    p_info.add_argument("config", help="path to YAML config file")
    p_info.set_defaults(func=_cmd_info)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
