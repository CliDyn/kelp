# Instructions for AI agents

This project uses `CLAUDE.md` as the single source of truth for LLM development instructions;
the per-module notes it points to live in `docs/dev/`.

**Read `CLAUDE.md` before doing anything.** It contains:

- What this project is (Lagrangian particle tracker for ocean meshes)
- Repository layout (the import graph is in `docs/dev/README.md`)
- Where the per-module notes are (`docs/dev/`: an index, then notes per module with
  function signatures, data shapes and traps; read them before changing a module)
- Conventions to follow when adding code (Numba, type hints, immutable state, etc.)
- Test patterns (synthetic data, no external files)
- What is and isn't implemented yet

All code conventions, naming patterns, and architectural decisions are documented there.
