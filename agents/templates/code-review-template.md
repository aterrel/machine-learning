# Code Review — [Sprint N / Feature Name]

**Reviewer**: Tech Lead
**Date**: YYYY-MM-DD
**Scope**: [Sprint N / PR / Feature description]
**Verdict**: Approved | Changes Required | Conditional Approval

---

## Verdict

**[VERDICT]**

[One paragraph explaining the verdict. If Changes Required, state what must be
fixed before the sprint can close. If Conditional Approval, state what should
be addressed in the next sprint.]

---

## Scope

Commits reviewed:
```
[git log --oneline output showing reviewed commits]
```

Files reviewed:
- `[file1.py]`
- `[file2.py]`
- `[test_file.py]`

---

## Findings

### Critical (must fix — blocks sprint close)

- [ ] **CRIT-1**: [File path, line] — [Description of issue]
  - **Problem**: [What's wrong]
  - **Fix required**: [What to do]

### Major (must fix — blocks sprint close)

- [ ] **MAJ-1**: [File path, line] — [Description of issue]
  - **Problem**: [What's wrong]
  - **Fix required**: [What to do]

### Minor (should fix — does not block sprint close)

- [ ] **MIN-1**: [File path, line] — [Improvement suggestion]

### Nit (optional — cosmetic or preference)

- **NIT-1**: [File path] — [Minor style note]

---

## Positive Observations

[What was done well. Acknowledge good patterns, clever solutions, thorough tests.]

- [Good thing 1]
- [Good thing 2]

---

## Checklist

| Category | Status | Notes |
|----------|--------|-------|
| Lint/format compliance | Pass / Fail | |
| Test coverage adequate | Pass / Fail | |
| Type hints complete | Pass / Fail | |
| Docstrings on public APIs | Pass / Fail | |
| Security: input validation | Pass / Fail | |
| Security: no hardcoded secrets | Pass / Fail | |
| Error handling complete | Pass / Fail | |
| No N+1 queries | Pass / Fail | |
| API contract matches REQ | Pass / Fail | |

---

## Actions Required

If verdict is **Changes Required**:

- [ ] [Programmer] Fix CRIT-1: [short description]
- [ ] [Programmer] Fix MAJ-1: [short description]
- [ ] [Tech Lead] Re-review after fixes

If verdict is **Conditional Approval**:

- [ ] [Programmer] Address MIN-1 in Sprint N+1 (tracked in todo.md)

---

*Review by: Tech Lead*
*Handoff to: [Programmer if Changes Required / Project Manager if Approved]*
