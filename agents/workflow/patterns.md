# Workflow Patterns

---

## Daily Agent Activation Pattern

### Session Start
1. **Read `agents/todo.md`** — find your pending tasks (marked `[ ]` for your role)
2. Read `PROJECT_STATUS.md` for current state
3. Run `git log --oneline -5` to see recent activity
4. Check `agents/handoffs/` for documents addressed to you
5. Check `agents/reviews/` for feedback on your previous work
6. **Create a git worktree** for your task (from main repo root):
   ```bash
   git worktree add agent-worktrees/<agent>-<TASK-ID> -b <agent>/<TASK-ID>-<slug>
   ```
7. Create session log: `agents/session-logs/[your-role]/YYYY-MM-DD-NNN.md`
8. Mark task `[>]` in-progress in `agents/todo.md`

### During Session — Commit at Every Logical Checkpoint
- After completing each discrete sub-task or file group, commit immediately:
  ```bash
  git add <specific-files>
  git commit -m "[Agent-Prefix] TASK-ID: description of checkpoint"
  ```
- Use `WIP:` prefix for incomplete checkpoints you need to save before context switch
- **Never accumulate more than ~5 file changes without committing**

### Session End
1. Finalize and commit session log
2. **Update `agents/todo.md`** — mark tasks done (`[x]`), add new tasks produced
3. Update PROJECT_STATUS.md if assignments changed
4. Create handoff document for the next agent (if needed)
5. **Push your branch**: `git push -u origin <branch-name>`
6. Open PR if work is complete and review-ready

---

## Feature Development Workflow

### Step 1: Requirements (Product Manager)
1. Read relevant section of `PROJECT.md`
2. Create `agents/requirements/REQ-NNNN.md` using requirement-template.md
3. Define acceptance criteria — each must be specific and testable
4. Create handoff → Software Architect

### Step 2: Architecture (Software Architect)
1. Read the REQ document
2. Design the solution in `agents/architecture/ARCH-NNN.md`
3. Record decisions as ADRs if needed (adr-template.md)
4. Create handoff → Programmer

### Step 3: Implementation (Programmer)
1. Read REQ + ARCH documents and any handoffs
2. Create worktree + branch:
   ```bash
   git worktree add agent-worktrees/prog-REQ-NNNN -b prog/REQ-NNNN-short-description
   ```
3. Implement following CLAUDE.md code standards
4. Write unit tests alongside code
5. **Commit at each logical checkpoint**
6. Run lint + tests before final commit
7. Push and create handoff → QA Agent

### Step 4: Testing (QA Agent)
1. Read REQ + implementation
2. Create test plan in `agents/session-logs/qa-agent/test-plan-REQ-NNNN.md`
3. Write integration tests
4. File bugs if found (bug-report-template.md)
5. Create handoff → Tech Lead (passing) or → Programmer (failing)

### Step 5: Review (Tech Lead)
1. Review code against CLAUDE.md standards
2. Complete code-review-template.md
3. Either: approve → merge, or request changes → back to Programmer

### Step 6: Merge
1. All CI checks pass (lint + tests)
2. Tech Lead has issued **Approved** verdict
3. Project Manager confirms task complete in PROJECT_STATUS.md

---

## Sprint Review Pattern

**At the end of every sprint**, before the Project Manager closes the sprint and opens the next one, the Tech Lead must perform a sprint-wide code review.

### When to Trigger
- After all P0 sprint tasks are committed to main
- Before the Project Manager runs the sprint retrospective / next sprint kickoff
- The `/go` flow automatically includes this step

### Tech Lead Sprint Review Steps
1. Run `git log --oneline` since last sprint boundary
2. Review all new/modified source files
3. Review all new/modified test files
4. Create review document: `agents/reviews/code/TL-review-sprint<N>-<date>.md`
5. Issue one of three verdicts:
   - **Approved** — no blocking issues; sprint may close
   - **Changes Required** — Critical/Major issues; Programmer must fix before close
   - **Conditional Approval** — Minor issues only; sprint may close, tracked for next sprint
6. If **Changes Required**: create handoff to Programmer
7. Commit: `[Tech Lead] TL-review: Sprint N code review`

### Severity Definitions
| Severity | Definition | Blocks Sprint Close? |
|----------|-----------|---------------------|
| Critical | Incorrect behavior, data corruption, crash, security vulnerability | Yes |
| Major | Logic error, missing error handling, broken contract | Yes |
| Minor | Style, naming, missing docstring | No |
| Nit | Formatting preference, cosmetic | No |

---

## Feature Complexity Classification

### Simple (1-2 day estimated)
- Single function/endpoint, no architecture changes, existing test patterns apply
- **Process**: Programmer → QA → **Tech Lead review** → Merge (skip Architecture step)

### Medium (3-5 day estimated)
- New module or significant class change, may need architecture decision
- **Process**: Full workflow (all 6 steps, including Tech Lead review)

### Complex (1+ week estimated)
- Cross-cutting architectural change, multiple modules affected, performance implications
- **Process**: Full workflow + ADR + Claude Manager involvement + **Tech Lead sprint review** + milestone review after

---

## Agent Handoff Pattern

Handoff documents live in `agents/handoffs/`. See handoff-template.md for format.

**Filename**: `HANDOFF-[FROM]-to-[TO]-[REQ-or-topic]-[date].md`
**Example**: `HANDOFF-prod-mgr-to-arch-REQ-0001-2026-02-24.md`

**Handoff must include**:
1. What was completed (summary)
2. What the next agent needs to do (clear task)
3. Relevant file paths
4. Open questions / decisions needed
5. Acceptance criteria from the REQ document

---

## Best Practices

### Session Management
- Keep sessions focused — one feature or task per session
- Document blockers immediately in PROJECT_STATUS.md
- Don't start new work without reading handoffs addressed to you

### Documentation
- Session logs must be created before ending a session
- Architecture decisions must be recorded as ADRs before implementation begins
- REQ documents must have clear acceptance criteria

### Code Quality
- Lint must pass before any commit (see CLAUDE.md for commands)
- Tests must pass before any commit
- New public functions need docstrings / JSDoc
- Type hints or type annotations on all public interfaces

### Git Hygiene
- Feature branches only — never commit directly to main
- One logical change per commit
- Commit message starts with agent prefix (see git-workflow.md)
- Reference REQ number in commit when relevant

---

## Troubleshooting

### "I don't know what to work on"
1. Read PROJECT_STATUS.md → Sprint Backlog
2. Check `agents/handoffs/` for documents addressed to you
3. Check with Claude Manager

### "The tests are failing"
1. Run tests with verbose flag for full output
2. Check if a dependency changed
3. If implementation issue: create bug report, assign to Programmer
4. If test issue: fix the test and document why

### "I need to make an architecture decision"
1. Create an ADR using `agents/templates/adr-template.md`
2. Discuss options in the ADR document
3. Tag Software Architect + Tech Lead for review before proceeding

### "I'm blocked on an external dependency"
1. Document the blocker in PROJECT_STATUS.md Blockers section
2. Mark your task `[!]` in todo.md with explanation
3. Identify any other tasks that can proceed while blocked
4. Notify Claude Manager via session log
