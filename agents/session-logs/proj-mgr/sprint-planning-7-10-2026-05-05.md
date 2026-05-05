<!-- Agent: Project Manager -->
<!-- Session-Type: Sprint Planning -->
<!-- Date: 2026-05-05 -->
<!-- Timestamp: 2026-05-05T00:00:00Z -->
<!-- Agents-Activated: Software Architect, Product Manager, Tech Lead -->
<!-- Skills-Used: /plan -->
<!-- Outcome: Success -->

# Session Log — Project Manager — Sprint Planning 7-10 — 2026-05-05

## Session Summary

Multi-agent planning session covering Sprints 7 through 10. Three sub-agents (Software Architect, Product Manager, Tech Lead) reviewed existing design documents, created new ARCH and REQ documents for backlog items, performed technical risk review, and produced detailed task breakdowns. PROJECT_STATUS.md updated to reflect all sprint plans and new documents.

---

## Work Completed

### Sub-Agent A: Software Architect

**ARCH-004 Verdict: Conditional Approval**
- Design is sound and complete; one gap identified: `sm_version: str` field must be added to `DeviceSpec` in Sprint 7 (required by ARCH-005/Sprint 8). Exception type discrepancy between `from_name()` (KeyError) and `from_device()` (ValueError) is correct behavior — programmer must implement both.

**ARCH-005 Verdict: Conditional Approval**
- Comprehensive design; two documentation gaps: `_MMA_LATENCY` dict values not explicitly listed (sourced from ptx-tracer-research.md), and sm_86 ArchSpec uses placeholder `...` values. Neither blocks implementation.

**New ARCH documents created:**
- `agents/architecture/ARCH-006.md` — CI/CD Pipeline (GitHub Actions): two-workflow design (CPU-only CI on every push; GPU CI on manual trigger / self-hosted runner)
- `agents/architecture/ARCH-007.md` — Jupyter Notebook Coverage: notebooks as presentation wrappers around existing demo modules; GPU-guarded cells; `nbconvert --execute` test strategy

**Sprint 7 task list (8 files, dependency-ordered):**
1. `src/kernel_model/__init__.py`
2. `src/kernel_model/device_spec.py` (includes `sm_version` field)
3. `src/kernel_model/occupancy.py`
4. `src/kernel_model/roofline.py`
5. Update `__init__.py` re-exports
6. `demos/09_kernel_model/__init__.py`
7. `demos/09_kernel_model/main.py`
8. `tests/test_kernel_model.py`

**Sprint 8 task list (9 files, dependency-ordered):**
1. `src/kernel_model/_taxonomy.py`
2. `src/kernel_model/_arch_table.py`
3. `src/kernel_model/ptx_tracer.py`
4. Update `src/kernel_model/__init__.py`
5. `demos/10_ptx_tracer/__init__.py`
6. `demos/10_ptx_tracer/ptx_fixtures/vector_add.ptx`
7. `demos/10_ptx_tracer/ptx_fixtures/gemm_mma.ptx`
8. `demos/10_ptx_tracer/main.py`
9. `tests/test_ptx_tracer.py`

---

### Sub-Agent B: Product Manager

**REQ-0012 created** — CI/CD Pipeline (GitHub Actions): P1, Sprint 9
- Two workflows: `ci.yml` (lint + CPU tests, every push), `gpu-ci.yml` (full GPU tests, manual trigger)
- Acceptance criteria: CI passes in < 5 min; lint violation causes failure; GPU workflow manually triggerable

**REQ-0013 created** — Jupyter Notebook Coverage (demos 01 + 02): P1, Sprint 9
- Notebooks as wrappers around existing demo modules
- GPU-guarded cells; must run to completion on CPU-only via `nbconvert --execute`
- Committed with cleared output

**REQ-0014 created** — User-Facing README Documentation: P1, Sprint 10
- Top-level README with quickstart, demo table (01-10), prerequisites, CI badge, test instructions
- Under 300 lines; all relative links valid; renders on GitHub
- Deferred to Sprint 10 (after CI badge and notebooks available)

**Rationale for no REQ for physical GPU validation**: GPU validation is a quality-assurance activity, not a product feature. It is tracked as a Sprint 10 deliverable in PROJECT_STATUS.md but does not need a formal REQ document.

---

### Sub-Agent C: Tech Lead — Risk Review

**Sprint 7 (ARCH-004) — Go/No-Go: GO**

Key findings:
1. Missing `sm_version` field in ARCH-004 — must be added in `device_spec.py` (Sprint 7 carries this forward for Sprint 8)
2. Test coverage: 11 CPU + 1 GPU test is adequate; all edge cases covered
3. API consistency: `OccupancyModel(device).compute(...)` pattern matches existing src/ conventions
4. Python 3.11 compatibility: all type hints are valid
5. No security concerns: pure analytical code, no shell calls, no user eval

**Sprint 8 (ARCH-005) — Go/No-Go: GO (contingent on Sprint 7)**

Key findings:
1. `_MMA_LATENCY` values not explicit in ARCH-005 — programmer must consult ptx-tracer-research.md
2. sm_86 ArchSpec values left as `...` — programmer must fill from research doc
3. Test coverage: 13 CPU test cases is thorough, including performance benchmark
4. `trace_file()` should use `pathlib.Path.read_text()` for consistency (minor)
5. Hard dependency on Sprint 7 `DeviceSpec.sm_version` — Sprint 8 cannot start until Sprint 7 is complete

**Recommended implementation order for Sprint 7:**
device_spec.py → occupancy.py → roofline.py → __init__.py → demos/09_kernel_model/main.py → tests/test_kernel_model.py

---

### Project Manager: Synthesis and Updates

- Updated `PROJECT_STATUS.md`:
  - Sprint 7: expanded deliverables table with all 8 files in dependency order; added Definition of Done
  - Sprint 8: expanded deliverables table with all 9 files; added Definition of Done
  - Sprint 9 (new): CI/CD + Jupyter notebooks; 4 deliverables; Definition of Done
  - Sprint 10 (new): README + GPU validation; 3 deliverables; Definition of Done
  - Architecture document status table added
  - ARCH-004/005 Conditional Approval verdicts recorded
  - Backlog table updated with REQ refs and sprint slots
  - Key File Locations updated to include ARCH-006/007 and REQ-0010–0014

---

## Decisions Made

| Decision | Rationale | Alternatives Considered |
|----------|-----------|------------------------|
| ARCH-004/005: Conditional Approval (not Changes Required) | Both documents are implementation-ready; gaps are minor clarifications, not design flaws | Changes Required would delay Sprint 7 start |
| `sm_version` added to Sprint 7 scope | ARCH-005 requires it; better to implement once in Sprint 7 than retrofit in Sprint 8 | Separate Sprint 7.5 for field addition (wasteful) |
| CI/CD split into two workflows | GPU tests cannot run on GitHub-hosted runners; two-tier is the industry-standard pattern | Docker with CUDA (adds infrastructure cost and complexity) |
| Notebooks as wrappers, not reimplementations | Single source of truth for demo logic; no maintenance burden from code duplication | Self-contained notebooks (would diverge from demo modules) |
| README in Sprint 10, not Sprint 9 | README should reference all 10 demos; CI badge and notebooks should exist first | README in Sprint 9 (would need updating after Sprint 9 completes) |
| No REQ for physical GPU validation | Validation is QA activity, not product feature; PROJECT_STATUS tracking is sufficient | REQ-0015 for GPU validation (over-engineered) |

## Blockers / Open Questions

- [ ] `_MMA_LATENCY` dict values for bottleneck classification in ARCH-005: programmer must consult ptx-tracer-research.md before implementing `ptx_tracer.py`
- [ ] GitHub repository URL needed for README CI badge and clone instructions (Sprint 10)
- [ ] Self-hosted GPU runner availability for `gpu-ci.yml` (Sprint 9) — may be documentation-only initially
- [ ] B100 and RTX 5090 specs in DeviceSpec table should be verified against official NVIDIA datasheets before shipping Sprint 7

## Next Steps

- Programmer begins Sprint 7 implementation starting with `device_spec.py`
- Programmer must include `sm_version: str` field in DeviceSpec (required by Sprint 8)
- Sprint 8 blocked until Sprint 7 Tech Lead review passes
- Sprint 9 (CI + Notebooks) begins after Sprint 8 closes
- Sprint 10 (README + GPU validation) begins after Sprint 9 closes

## Files Changed

- `PROJECT_STATUS.md` (modified — Sprint 7-10 plans, ARCH verdicts, new REQ/ARCH index)
- `agents/architecture/ARCH-006.md` (created — CI/CD Pipeline architecture)
- `agents/architecture/ARCH-007.md` (created — Jupyter Notebook architecture)
- `agents/requirements/REQ-0012.md` (created — CI/CD Pipeline requirement)
- `agents/requirements/REQ-0013.md` (created — Jupyter Notebook requirement)
- `agents/requirements/REQ-0014.md` (created — README documentation requirement)
- `agents/session-logs/proj-mgr/sprint-planning-7-10-2026-05-05.md` (created — this file)
