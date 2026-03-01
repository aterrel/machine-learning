# REQ-NNNN: [Requirement Title]

**Status**: Draft | Active | Implemented | Verified
**Priority**: P0 | P1 | P2
**Objective Reference**: [Objective N from PROJECT.md]
**Created**: YYYY-MM-DD
**Author**: Product Manager
**Assigned To**: [Agent or Unassigned]

---

## Summary

[1-3 sentence description of what this requirement covers and why it matters.]

---

## Background

[Context from PROJECT.md. Why does this feature exist? What user problem does it solve?]

---

## Requirements

### Functional Requirements

| ID | Requirement | Priority |
|----|-------------|---------|
| REQ-NNNN-F1 | [Functional requirement 1] | P0 |
| REQ-NNNN-F2 | [Functional requirement 2] | P0 |
| REQ-NNNN-F3 | [Optional: lower priority requirement] | P1 |

### Non-Functional Requirements

| ID | Requirement | Priority |
|----|-------------|---------|
| REQ-NNNN-NF1 | [Performance, reliability, or security requirement] | P0 |

---

## Acceptance Criteria

All criteria must pass for this requirement to be marked **Implemented**:

- [ ] AC-1: [Specific, testable condition — "User can X and receives Y"]
- [ ] AC-2: [Specific, testable condition]
- [ ] AC-3: [Performance criterion if applicable — e.g., "completes in < N seconds"]

---

## API / Interface Design (if applicable)

```
# Describe the expected interface — REST endpoints, CLI flags, function signatures, etc.
# Use the notation appropriate for your stack.

# REST example:
POST   /api/v1/resource          # Create
GET    /api/v1/resource          # List
GET    /api/v1/resource/{id}     # Get one
PATCH  /api/v1/resource/{id}     # Update
DELETE /api/v1/resource/{id}     # Delete

# Function signature example:
async def process(input: InputModel) -> OutputModel: ...
```

---

## Out of Scope

- [What is explicitly NOT included in this requirement]

---

## Dependencies

| Dependency | Type | Status |
|-----------|------|--------|
| [Other REQ or ARCH or external] | requires | [status] |

---

## Open Questions

- [ ] [Question that needs resolution before implementation]

---

## Notes

[Additional context, links to external docs, references to external APIs, etc.]
