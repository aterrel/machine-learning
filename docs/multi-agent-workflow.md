# Multi-Agent Workflow — Philosophy and Guide

---

## Why Multi-Agent?

Software development involves fundamentally different modes of thinking:

- **Product thinking**: What should we build? For whom? What matters most?
- **Architectural thinking**: How should it be structured? What are the trade-offs?
- **Implementation thinking**: How do we write this code correctly and efficiently?
- **Quality thinking**: How do we verify this is correct and maintainable?
- **Coordination thinking**: What are we building next? What's blocking us?

A single "do everything" AI agent tends to conflate these modes — it'll start coding before
the requirements are clear, or make architectural decisions implicitly in implementation code,
or forget to test because it's focused on getting the feature working.

The multi-agent workflow **separates concerns by role** and makes communication between roles
explicit (through REQ documents, ARCH documents, and handoffs). Each agent has a clear mandate,
clear inputs, and clear outputs.

---

## The Document Pipeline

The core insight: **documents are the API between agents**.

```
PROJECT.md           ← Human fills this in (objectives, tech stack, constraints)
    │
    ▼
REQ-NNNN.md          ← Product Manager translates objectives into testable requirements
    │                   (acceptance criteria, API design, out of scope)
    ▼
ARCH-NNN.md          ← Software Architect designs the solution
    │                   (module structure, data flow, key decisions, testing strategy)
    ▼
Implementation       ← Programmer writes code following REQ + ARCH
    │
    ▼
Test Plan            ← QA Agent verifies acceptance criteria are met
    │
    ▼
Code Review          ← Tech Lead issues sprint verdict
```

Each document in this pipeline answers a question:
- **REQ**: What are we building and how will we know it's done?
- **ARCH**: How should it be built and why?
- **Handoff**: What did I do, what do you need to do next?
- **Code Review**: Is this ready to ship?

---

## The Sprint Cycle

### Sprint Kickoff

The Project Manager (via `/go`) evaluates:
- What's done? (git history)
- What's pending? (todo.md)
- What's blocked? (todo.md + PROJECT_STATUS.md)

Then activates agents in order:
1. Product Manager (if REQs are missing)
2. Software Architect (if ARCH docs are missing for ready REQs)
3. Programmer (if implementation is ready to start)
4. QA Agent (in parallel with Programmer, or after)

### Sprint Execution

Agents work in sequence or parallel based on dependencies:

```
Product Manager ──→ Architect ──→ Programmer
                                      │
                          QA Agent ←──┘  (after implementation)
```

**No agent starts work without reading the handoff document addressed to them.**

### Sprint End

The Tech Lead reviews **everything** committed since the last sprint boundary.
This is non-negotiable — it's the quality gate.

Verdicts:
- **Approved**: Sprint closes. Next sprint opens.
- **Changes Required**: Programmer fixes Critical/Major issues. TL re-reviews. Then sprint closes.
- **Conditional Approval**: Minor issues tracked for next sprint. Sprint closes.

**The sprint gate rule exists because**: without it, technical debt accumulates invisibly.
The Tech Lead review forces a quality checkpoint every sprint, not "eventually when we have time."

---

## What Makes Good REQ Documents

A good REQ document is one where:
1. A new agent who hasn't been in the conversation can understand what to build from the REQ alone
2. Every acceptance criterion is testable: "User can X and receives Y" — not "X works correctly"
3. The Out of Scope section is explicit — prevents scope creep
4. The API/Interface Design section gives the Programmer a head start
5. Dependencies between REQs are called out explicitly

A bad REQ document is vague: "Users should be able to manage their accounts." There's no way to know when this is done.

---

## What Makes Good ARCH Documents

A good ARCH document:
1. Explains module boundaries — what each module IS and IS NOT responsible for
2. Shows data flow end-to-end for the main use cases
3. Records decisions AND what was rejected and why
4. Provides specific implementation notes: class names, method signatures, file paths
5. Has a testing strategy — how will QA verify this component?

A bad ARCH document is a vague diagram with boxes and arrows but no specifics.

---

## Commit Conventions

Every commit prefix tells you which agent made it and why:

| Prefix | Agent | Typical commits |
|--------|-------|----------------|
| `[Claude Manager]` | Meta-orchestrator | Bootstrap, workflow changes, project setup |
| `[Proj-Mgr]` | Project Manager | STATUS updates, sprint open/close |
| `[Prod-Mgr]` | Product Manager | REQ documents, requirements changes |
| `[Arch]` | Software Architect | ARCH documents, ADRs |
| `[Prog]` | Programmer | Feature implementation, bug fixes |
| `[QA]` | QA Agent | Tests, bug reports |
| `[Tech Lead]` | Tech Lead | Code reviews, standards enforcement |

This makes `git log` a readable history of agent decisions, not just code changes:
```
[Tech Lead] Sprint 2 code review: Conditional Approval
[Prog] REQ-0003: complete priority classification service
[QA] REQ-0003: test plan and integration tests
[Arch] ARCH-002: priority engine design
[Prod-Mgr] REQ-0003: priority classification requirements
```

---

## Complexity-Driven Process

Not every feature needs the full pipeline. Match the process to the complexity:

### Simple (1-2 days)
Single endpoint or small utility function. No architecture changes needed.

```
Programmer → QA Agent → Tech Lead review → Merge
```

### Medium (3-5 days)
New service module or significant feature. May need one ARCH decision.

```
Product Manager → Architect → Programmer → QA Agent → Tech Lead review → Merge
```

### Complex (1+ week)
Cross-cutting changes, new subsystem, performance or security implications.

```
Product Manager → Architect (with ADR) → Programmer → QA Agent → Tech Lead sprint review → Milestone review → Merge
```

Claude Manager decides complexity classification. Default: when in doubt, do the Medium process.

---

## Handling Blockers

When work is blocked:

1. Mark the task `[!]` in todo.md with a reason:
   ```
   [!] [Prog] Implement Gmail sync (blocked — waiting for OAuth credentials from user)
   ```

2. Add to PROJECT_STATUS.md Blockers section:
   ```
   | Blocker | Impact | Owner | Resolution |
   |---------|--------|-------|-----------|
   | Gmail API credentials not configured | Blocks REQ-0002 | User | User must create Google Cloud project |
   ```

3. Continue with any unblocked tasks. Don't let one blocker stop all progress.

4. When the blocker is resolved, update both files and re-run `/go`.

---

## Memory and Context

Agents don't have persistent memory across sessions. This is intentional and handled through documents:

- `PROJECT_STATUS.md` — current state of the project
- `agents/todo.md` — current task list
- `agents/handoffs/` — inter-agent context passing
- `agents/session-logs/` — historical record of decisions
- REQ + ARCH documents — the single source of truth for features

At the start of every session, the agent reads these documents to reconstruct context.
**Documents replace memory.**

---

## Adapting the Workflow

The workflow is designed to be modified. Common adaptations:

**Fewer agents** (solo projects): Skip the Project Manager and have Claude Manager track status directly.

**More agents**: Add a DevOps Agent for infrastructure, a Security Agent for pen testing, a Documentation Agent for user docs.

**Different cadence**: Change sprint length in patterns.md. Add daily standup as a slash command.

**Different tech stack**: Replace python-standards.md with your language's standards. Update the lint/test commands in CLAUDE.md.

**Stricter gates**: Require QA Agent sign-off before Tech Lead review. Add performance benchmarks as a required step.

The key invariant to preserve: **the Tech Lead sprint review is mandatory**. Everything else is negotiable.
