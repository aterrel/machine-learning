# ARCH-NNN: [Architecture Document Title]

**Status**: Draft | Approved | Superseded
**Created**: YYYY-MM-DD
**Author**: Software Architect
**REQ Reference**: [REQ-NNNN, REQ-NNNN]
**Approved By**: [Tech Lead, Claude Manager]

---

## Overview

[2-4 sentence summary of what this document covers and what architectural problem it solves.]

---

## Context

[Why does this architecture decision need to be made? What are the constraints?
What does the existing system provide that we build on?]

---

## Design

### Module Structure

```
[project root]/
├── [module/]
│   ├── [submodule.py]     # Description of responsibility
│   └── [submodule2.py]    # Description of responsibility
└── [other-module/]
    └── ...
```

### Key Classes / Interfaces

[Describe the main classes, functions, or interfaces and their responsibilities.
Include method signatures for the most important ones.]

```python
class ExampleService:
    async def main_operation(self, input: InputType) -> OutputType: ...
    async def helper_operation(self, ...): ...
```

### Data Flow

[Describe how data moves through the system for the main use case.]

```
Input → [Component A] → [Component B] → [Database] → [Component C] → Output
```

---

## Key Design Decisions

1. **[Decision 1]**: [What was decided and why. What alternatives were rejected.]

2. **[Decision 2]**: [What was decided and why. What alternatives were rejected.]

---

## Consequences

### Positive
- [What this design enables or makes easier]

### Negative / Trade-offs
- [What this design makes harder or forecloses]

### Risks
- [Risks and mitigations]

---

## Testing Strategy

[How will this component be tested? Unit tests? Integration tests? Mocks needed?]

---

## Implementation Notes for Programmer

[Specific guidance: which files to create, what to implement first, known gotchas,
references to external documentation.]

---

## ADRs

- [ADR-NNN: title of related architecture decision record]
