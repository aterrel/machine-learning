# Session Log Standards

---

## When to Create a Session Log

Every agent session must produce a session log. Create it **before** starting work,
update during, and finalize before committing at session end.

A session = one focused work block by one agent on one task or related set of tasks.

---

## Standard Format

```markdown
<!-- Agent: [Agent Name] -->
<!-- Session-Type: [Feature Implementation | Architecture Design | Requirements | Code Review | Sprint Planning | Bug Fix] -->
<!-- Date: YYYY-MM-DD -->
<!-- Timestamp: YYYY-MM-DDTHH:MM:SSZ -->
<!-- Agents-Activated: [list of agents activated this session] -->
<!-- Skills-Used: [slash commands used, e.g., /go, /plan] -->
<!-- Outcome: [Success | Partial | Blocked] -->

# Session Log — [Agent Name] — YYYY-MM-DD-NNN

## Session Summary

[1-3 sentences. What was the goal? What was accomplished?]

## Work Completed

### [Task 1 Name]
- [Specific thing done]
- [Files created or modified]
- [Decisions made]

### [Task 2 Name]
- ...

## Decisions Made

| Decision | Rationale | Alternatives Considered |
|----------|-----------|------------------------|
| [Decision] | [Why] | [What else was considered] |

## Next Steps

- [What the next agent needs to do]
- [Any blockers to resolve]
- [Handoffs created]

## Files Changed

- `path/to/file.py` (created/modified)
- `agents/handoffs/HANDOFF-...md` (created)
```

---

## File Naming

```
agents/session-logs/[agent-role]/YYYY-MM-DD-NNN.md
```

Where `NNN` is a zero-padded sequence number for that day (001, 002, etc.).

**Agent role directory names**:
- `claude-manager/`
- `proj-mgr/`
- `prod-mgr/`
- `arch/`
- `programmer/`
- `qa-agent/`
- `tech-lead/`

**Examples**:
- `agents/session-logs/programmer/2026-03-01-001.md`
- `agents/session-logs/arch/2026-03-01-001.md`
- `agents/session-logs/qa-agent/2026-03-02-001.md`

---

## What to Include

**Always include**:
- What task you were working on (REQ-NNNN, ARCH-NNN, or description)
- What was completed vs what was left
- Any decisions made and why
- Files created or significantly modified
- Handoffs created

**Include when relevant**:
- Blockers encountered
- Open questions that need resolution
- Architecture concerns flagged
- Bugs discovered
- Performance observations

**Do NOT include**:
- Full file contents (reference paths instead)
- Long code excerpts (summarize instead)
- Implementation details better captured in the code itself

---

## Example Session Log

```markdown
<!-- Agent: Programmer -->
<!-- Session-Type: Feature Implementation -->
<!-- Date: 2026-03-05 -->
<!-- Timestamp: 2026-03-05T14:30:00Z -->
<!-- Agents-Activated: Programmer -->
<!-- Skills-Used: none -->
<!-- Outcome: Success -->

# Session Log — Programmer — 2026-03-05-001

## Session Summary

Implemented the AccountService (REQ-0001) including CRUD operations and OAuth token
encryption. All 12 tests passing. Created handoff to QA Agent.

## Work Completed

### Account Service Implementation
- Created `backend/app/services/account_service.py` with AccountService class
- Methods: create_account, get_account, list_accounts, update_account, delete_account
- OAuth token encryption using Fernet (SECRET_KEY from environment)

### API Endpoint Implementation
- Completed `backend/app/api/endpoints/accounts.py` (was a stub)
- All 5 endpoints implemented: GET/POST /accounts, GET/PATCH/DELETE /accounts/{id}
- Input validation via Pydantic schemas

### Tests
- `backend/tests/services/test_account_service.py`: 12 tests, all passing
- `backend/tests/api/test_accounts.py`: 8 tests, all passing

## Decisions Made

| Decision | Rationale | Alternatives Considered |
|----------|-----------|------------------------|
| Fernet symmetric encryption for tokens | Simple, fast, key rotation possible | AES-GCM (more complex), no encryption (rejected — security req) |
| Soft delete via deleted_at column | Allows audit trail | Hard delete (rejected — data recovery concern) |

## Next Steps

- QA Agent: write additional edge case tests for token refresh failure scenarios
- Programmer (next session): implement Gmail OAuth flow (auth.py endpoint stubs)
- Created: `agents/handoffs/HANDOFF-prog-to-qa-REQ-0001-2026-03-05.md`

## Files Changed

- `backend/app/services/account_service.py` (created)
- `backend/app/api/endpoints/accounts.py` (implemented from stub)
- `backend/tests/services/test_account_service.py` (created)
- `backend/tests/api/test_accounts.py` (created)
- `agents/session-logs/programmer/2026-03-05-001.md` (this file)
- `agents/handoffs/HANDOFF-prog-to-qa-REQ-0001-2026-03-05.md` (created)
```
