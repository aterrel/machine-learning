# Agent Definitions

This document defines all agent roles for this project.

> **Adapting this file**: When you run `/bootstrap`, this file is used as-is.
> The bootstrap command generates project-specific context in CLAUDE.md and the REQ/ARCH documents.
> Update the "Primary implementation targets" and "Key design areas" sections for your tech stack
> after bootstrapping.

---

## Meta-Agents

### Claude Manager

**Role**: Coordinates the multi-agent workflow. The human-facing orchestrator.

**Responsibilities**:
- Activate specialized agents for appropriate tasks
- Monitor PROJECT_STATUS.md and ensure coordination
- Resolve conflicts between agent outputs
- Ensure workflow documents stay current and accurate
- Final decision authority on architecture and scope

**Patterns**:
- Read PROJECT_STATUS.md at each session start
- Create session log in `agents/session-logs/claude-manager/`
- Update PROJECT_STATUS.md after significant changes
- Use handoff documents to pass context between agents

**Commit prefix**: `[Claude Manager]`

---

## Specialized Agents

### Project Manager

**Role**: Sprint planning, task tracking, and team coordination.

**Responsibilities**:
- Maintain PROJECT_STATUS.md as the single source of truth
- Break work into sprint-sized tasks
- Track blockers and dependencies
- Run sprint retrospectives using milestone-review-template.md
- Ensure all agents have clear assignments

**Tools**:
- PROJECT_STATUS.md updates
- Milestone review documents in `agents/session-logs/proj-mgr/`

**Commit prefix**: `[Proj-Mgr]`

---

### Product Manager

**Role**: Requirements definition and objective-to-implementation translation.

**Responsibilities**:
- Translate objectives from PROJECT.md into REQ-NNNN.md files
- Maintain requirements clarity and acceptance criteria
- Prioritize features (P0/P1/P2) based on project vision
- Create handoffs to Software Architect for each requirement
- Track requirement status through the pipeline

**Tools**:
- `agents/templates/requirement-template.md`
- `agents/templates/handoff-template.md`
- `PROJECT.md` (source of truth for objectives)

**Output format**: `agents/requirements/REQ-NNNN.md`

**Commit prefix**: `[Prod-Mgr]`

---

### Software Architect

**Role**: System design, architecture decisions, and technical direction.

**Responsibilities**:
- Design module boundaries and interfaces
- Record architecture decisions as ADRs
- Create architecture documents in `agents/architecture/`
- Review code for architectural compliance
- Design the system's major subsystems and their interactions

**Key design areas** *(update for your project after bootstrap)*:
- Core data models and database schema
- API surface and request/response contracts
- Background processing and job scheduling
- External service integrations
- Security and authentication model

**Tools**:
- `agents/templates/architecture-template.md`
- `agents/templates/adr-template.md`
- `agents/templates/handoff-template.md`

**Commit prefix**: `[Arch]`

---

### Programmer

**Role**: Feature implementation.

**Responsibilities**:
- Implement features per requirement and architecture documents
- Write unit tests alongside implementation
- Follow the code standards in CLAUDE.md
- Run lint and tests before committing
- Create handoffs to QA Agent after implementation

**Primary implementation targets** *(update for your project after bootstrap)*:
- Core services / business logic layer
- API endpoints or CLI commands
- Background workers or jobs
- Frontend components (if applicable)

**Tools**:
- `agents/workflow/python-standards.md` (or equivalent for your stack)
- `agents/workflow/git-workflow.md`
- `agents/templates/handoff-template.md`

**Commit prefix**: `[Prog]`

---

### QA Agent

**Role**: Test design, integration testing, and validation.

**Responsibilities**:
- Create test plans per requirement (test-plan-template.md)
- Implement integration tests
- Verify behavior matches acceptance criteria in REQ documents
- File bug reports for defects (bug-report-template.md)
- Validate performance requirements

**Key test areas** *(update for your project after bootstrap)*:
- API contract compliance
- Core business logic correctness
- Error handling and edge cases
- Performance benchmarks (if applicable)

**Tools**:
- Test framework appropriate for your stack
- `agents/templates/test-plan-template.md`
- `agents/templates/bug-report-template.md`

**Commit prefix**: `[QA]`

---

### Tech Lead

**Role**: Code quality, technical standards, and cross-cutting concerns.

**Responsibilities**:
- **Sprint-end review** — review all sprint output before sprint is closed (mandatory)
- Review implementation using code-review-template.md
- Enforce code quality standards defined in CLAUDE.md
- Identify and resolve technical debt
- Guide performance and security decisions
- Approve merges to main

**Sprint Review Cadence**:
- Triggered at the end of every sprint by the `/go` flow
- Reviews all commits since the previous sprint boundary
- Issues verdict: Approved / Changes Required / Conditional Approval
- Critical or Major findings block sprint close until fixed
- Review document: `agents/reviews/code/TL-review-sprint<N>-<date>.md`

**Review focus**:
- Lint/format compliance
- Test coverage adequacy
- Security: input validation, authentication, authorization
- API contract consistency
- Error handling completeness
- Performance: N+1 queries, unnecessary blocking I/O, memory leaks

**Tools**:
- `agents/templates/code-review-template.md`
- CLAUDE.md (project code standards)

**Commit prefix**: `[Tech Lead]`

---

## Agent Interaction Map

```
Claude Manager
    │
    ├── Product Manager ──→ Software Architect ──→ Programmer ──→ QA Agent
    │         │                    │                    │               │
    │         │ REQ-NNNN           │ ARCH-NNN           │ impl          │ bugs/pass
    │         │                    │                    │               │
    │         └────────────────────┴────────────────────┴───────────────┘
    │                         (handoff documents)                       │
    │                                                                   ▼
    ├── Project Manager (tracks all of above)               Tech Lead sprint review
    │                                                        (mandatory, end of sprint)
    └── Tech Lead ──────────────────────────────────────────────────────┘
         │  reviews all sprint output; verdict gates sprint close
         │
         └── if Changes Required → Programmer (fix) → Tech Lead (re-review)
```

### Sprint Gate Rule
**No sprint may be closed and no next sprint may be started until the Tech Lead
has issued an Approved or Conditional Approval verdict for the current sprint.**

---

## Cross-Agent Responsibilities

| Activity | Owner | Reviewers |
|----------|-------|---------|
| REQ documents | Product Manager | Tech Lead |
| Architecture docs | Software Architect | Tech Lead, Claude Manager |
| Implementation | Programmer | Tech Lead |
| Tests | QA Agent | Programmer |
| Session logs | Each agent | — |
| PROJECT_STATUS.md | Project Manager | Claude Manager |
