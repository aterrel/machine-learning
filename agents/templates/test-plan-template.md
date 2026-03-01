# Test Plan — REQ-NNNN: [Requirement Title]

**Author**: QA Agent
**Date**: YYYY-MM-DD
**REQ Reference**: agents/requirements/REQ-NNNN.md
**Implementation**: [Branch or commit being tested]

---

## Scope

[What is being tested? Which acceptance criteria from the REQ document does this plan cover?]

---

## Test Environment

- Runtime: [e.g., Python 3.12, Node 20]
- Database: [e.g., PostgreSQL 15, SQLite in-memory]
- External services: [e.g., Mocked via httpx-mock / real test account / none]
- Test command: `[the command to run these tests]`

---

## Test Cases

### Happy Path

| ID | Scenario | Steps | Expected Result |
|----|----------|-------|----------------|
| TC-001 | [Normal use case] | 1. [Step] 2. [Step] | [What happens] |
| TC-002 | [Another normal case] | 1. [Step] | [What happens] |

### Error Cases

| ID | Scenario | Steps | Expected Result |
|----|----------|-------|----------------|
| TC-010 | [Invalid input] | 1. Send X | Returns 400 with message Y |
| TC-011 | [Not found] | 1. Request ID 99999 | Returns 404 |
| TC-012 | [Unauthorized] | 1. No token | Returns 401 |

### Edge Cases

| ID | Scenario | Steps | Expected Result |
|----|----------|-------|----------------|
| TC-020 | [Empty collection] | 1. List with no items | Returns [] with 200 |
| TC-021 | [Boundary condition] | 1. [Setup] | [Result] |

---

## Acceptance Criteria Coverage

| AC from REQ | Test Case(s) | Status |
|-------------|-------------|--------|
| AC-1: [text] | TC-001, TC-002 | Pending |
| AC-2: [text] | TC-010, TC-011 | Pending |
| AC-3: [text] | TC-020 | Pending |

---

## Out of Scope

- [What this test plan explicitly does NOT cover]

---

## Bugs Found

[Fill in during test execution]

| Bug ID | Description | Severity | Status |
|--------|-------------|----------|--------|
| BUG-001 | [Description] | Critical/Major/Minor | Open |
