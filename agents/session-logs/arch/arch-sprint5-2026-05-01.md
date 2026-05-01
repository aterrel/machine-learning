---
session: arch/sprint5-2026-05-01
agent: Software Architect
date: 2026-05-01
sprint: 5
---

# Software Architect Session — Sprint 5

## Task
Create ARCH-003: Multi-Backend Design Pattern for REQ-0007 (Numba + CuPy variants) and REQ-0008 (comparison demo + cuML).

## Decisions Made
1. Side-by-side files per demo (not subdirectories) — keeps each demo dir self-explanatory
2. All variants: numpy input → numpy output (same interface contract as existing demos)
3. Optional imports: try/except at module level, _BACKEND_AVAILABLE sentinel
4. Numba: warmup handled by BenchmarkRunner(warmup=1) — first call compiles JIT
5. CuPy: cp.cuda.Stream.null.synchronize() before stopping wall-clock timers
6. Benchmark runner: --backend flag dispatches; --backend all prints comparison table
7. Comparison demo (08_comparison) reuses existing algorithm functions via importlib; does NOT re-implement
8. cuML backends: always written, guarded by ImportError, return numpy arrays

## Output
ARCH-003.md created.
