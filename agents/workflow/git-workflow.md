# Git Workflow

---

## Git Worktrees for Agent Isolation

Each agent works in a **dedicated git worktree** so multiple agents can work in
parallel without interfering with each other's filesystem state.

Worktrees live under `.git/worktrees/` (managed by git) and are checked out to
a separate directory under `agent-worktrees/` at the project root.

### Worktree Lifecycle

```bash
# 1. Create a worktree at session start
git worktree add agent-worktrees/<agent>-<task-id> -b <agent>/<task-id>-<slug>
# e.g.:
git worktree add agent-worktrees/prod-mgr-REQ-0002 -b prod-mgr/REQ-0002-gmail-sync

# 2. Agent does ALL work inside that directory
cd agent-worktrees/prod-mgr-REQ-0002
# ... read, edit, write files ...

# 3. Commit checkpoints regularly (see Commit Cadence below)
git add <specific-files>
git commit -m "[Prod-Mgr] REQ-0002: add acceptance criteria"

# 4. Push branch when session is done
git push -u origin prod-mgr/REQ-0002-gmail-sync

# 5. Open PR for review / merge to main
gh pr create --title "[Prod-Mgr] REQ-0002: Gmail sync requirements" ...

# 6. Remove worktree after merge
git worktree remove agent-worktrees/prod-mgr-REQ-0002
git branch -d prod-mgr/REQ-0002-gmail-sync
```

### Worktree Naming Convention

| Agent | Worktree dir | Branch |
|-------|-------------|--------|
| Product Manager | `agent-worktrees/prod-mgr-<TASK>` | `prod-mgr/<TASK>-<slug>` |
| Software Architect | `agent-worktrees/arch-<TASK>` | `arch/<TASK>-<slug>` |
| Programmer | `agent-worktrees/prog-<TASK>` | `prog/<TASK>-<slug>` |
| QA Agent | `agent-worktrees/qa-<TASK>` | `qa/<TASK>-<slug>` |
| Tech Lead | `agent-worktrees/tech-lead-<TASK>` | `tech-lead/<TASK>-<slug>` |
| Project Manager | `agent-worktrees/proj-mgr-<TASK>` | `proj-mgr/<TASK>-<slug>` |
| Claude Manager | `agent-worktrees/claude-mgr-<TASK>` | `claude-mgr/<TASK>-<slug>` |

### Commit Cadence

Commit **at every logical checkpoint**, not just at session end:

| Trigger | Example commit message |
|---------|----------------------|
| After each document section | `[Prod-Mgr] REQ-0002: add functional requirements` |
| After resolving each blocker | `[Arch] ARCH-001: resolve data model open questions` |
| After any new file created | `[Prog] Add account service skeleton` |
| Before switching context | `[Prog] WIP: email sync — fetch logic done, parser WIP` |
| At session end | `[Prog] REQ-0002: complete gmail sync implementation` |

**Rule**: Never end a session with uncommitted work. A `WIP:` prefix is
acceptable for checkpoint commits that are not yet reviewable.

---

## Branch Strategy

| Branch Type | Pattern | Purpose | Example |
|-------------|---------|---------|---------|
| feature | `prog/REQ-NNNN-description` | New features | `prog/REQ-0001-account-management` |
| fix | `prog/BUG-NNN-description` | Bug fixes | `prog/BUG-001-auth-token-refresh` |
| arch | `arch/ARCH-NNN-description` | Architecture | `arch/ARCH-001-system-design` |
| test | `qa/description` | Test infrastructure | `qa/api-integration-tests` |
| docs | `prod-mgr/description` | Documentation/REQs | `prod-mgr/REQ-0003-priority-engine` |

**Rules**:
- Never commit directly to `main`
- All branches are created from current `main`
- Branches should be short-lived (max one sprint)
- Each branch lives in its own worktree under `agent-worktrees/`

---

## Agent Commit Prefixes

| Agent | Prefix |
|-------|--------|
| Claude Manager | `[Claude Manager]` |
| Project Manager | `[Proj-Mgr]` |
| Product Manager | `[Prod-Mgr]` |
| Software Architect | `[Arch]` |
| Programmer | `[Prog]` |
| QA Agent | `[QA]` |
| Tech Lead | `[Tech-Lead]` |

---

## Commit Message Format

```
[Agent-Prefix] Short description (≤72 chars)

Optional longer description explaining why, not what.
Reference REQ-NNNN if implementing a tracked requirement.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```

**Examples**:
```
[Prog] REQ-0001: implement account service with OAuth token storage

Adds AccountService with create, get, list, delete methods.
Tokens encrypted with Fernet before DB storage (see ARCH-001).
Tests in tests/services/test_account_service.py.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```

```
[QA] REQ-0002: test plan and API integration tests for sync endpoint

Covers: initial sync trigger, incremental sync, error handling.
All tests pass against test database with mocked Gmail API.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```

---

## Merge Requirements

Before merging a feature branch to `main`:

1. **Tests pass**: Test command from CLAUDE.md exits 0
2. **Lint clean**: Lint command from CLAUDE.md exits 0
3. **Code review**: Tech Lead approval on PR
4. **Architecture compliance**: For arch/ branches, Software Architect must approve
5. **No merge conflicts**: Branch is up to date with main

---

## Git Safety Protocol

**NEVER**:
- `git push --force` to main
- `git reset --hard` without saving current work
- `git commit --amend` on published commits
- Skip pre-commit hooks (`--no-verify`)
- Commit secrets, `.env` files, or credentials

**ALWAYS**:
- Stage specific files (`git add src/specific_file.py`)
- Review `git diff --staged` before committing
- Write descriptive commit messages with agent prefix

---

## Common Git Workflows

### Start agent session with a worktree
```bash
# From main repo root — create worktree + branch in one step
git worktree add agent-worktrees/prog-REQ-0001 -b prog/REQ-0001-feature-name

# Work inside the worktree
cd agent-worktrees/prog-REQ-0001
```

### Checkpoint commit during session
```bash
# Stage specific files (never git add -A)
git add src/services/account_service.py
git diff --staged   # review before committing
git commit -m "$(cat <<'EOF'
[Prog] REQ-0001: add AccountService with create and get methods

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

### Final commit + push at session end
```bash
git add src/services/account_service.py tests/services/test_account_service.py
git commit -m "$(cat <<'EOF'
[Prog] REQ-0001: complete account service implementation

Implements create, get, list, update, delete for accounts.
OAuth token encryption/decryption via Fernet.
Tests: 15 passing, all AC from REQ-0001 verified.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
git push -u origin prog/REQ-0001-feature-name
```

### Open PR
```bash
gh pr create --title "[Prog] REQ-0001: account service implementation" --body "..."
```

### Merge and clean up worktree
```bash
# After PR is merged
git worktree remove agent-worktrees/prog-REQ-0001
git branch -d prog/REQ-0001-feature-name
```
