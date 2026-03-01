You are activating a planning session. Read the topic or question provided after the /plan command (or ask the user for one if none given).

---

## PLANNING SESSION SEQUENCE

### Step 1: Understand the Ask

What is the planning question? Examples:
- "Should we add real-time notifications or polling?"
- "How should we model multi-tenancy?"
- "What's the best way to implement search?"
- "We need to add feature X — how complex is it?"

Classify the ask:
- **Feature Validation** → Is this worth building? What's the scope?
- **Technical Feasibility** → Can we do this with our stack? What are the trade-offs?
- **Architecture Decision** → Which approach should we take?
- **Scope Definition** → What exactly does this mean to build?

---

### Step 2: Product Manager Analysis

Research the user value and requirements:
1. Read `PROJECT.md` for vision and constraints
2. Read relevant existing `agents/requirements/` documents
3. Identify: Who benefits? What's the user story? What's the acceptance criteria?
4. Evaluate: Does this fit the project vision? What priority is it?
5. Check: Is this already captured in a REQ? If yes, reference it.

Output a Product Manager perspective:
- User value statement
- Suggested priority (P0/P1/P2)
- Draft acceptance criteria (3-5 items)
- Scope recommendation (what's in/out)

---

### Step 3: Technical Feasibility Analysis

Research technical approach:
1. Read relevant `agents/architecture/` documents
2. Consider the tech stack from `CLAUDE.md`
3. Evaluate: How does this fit existing architecture? What changes are needed?
4. Complexity classification (from agents/workflow/patterns.md):
   - **Simple** (1-2 days): Single function/endpoint, no architecture changes
   - **Medium** (3-5 days): New module, possible architecture decision needed
   - **Complex** (1+ week): Cross-cutting change, new subsystem, ADR required

Output a Technical Feasibility perspective:
- Complexity classification with rationale
- Approach options (2-3 options max)
- Recommended approach with trade-offs
- What architecture documents would need updating
- Risks and unknowns

---

### Step 4: Recommendation

Synthesize both perspectives into a clear recommendation:

```
PLANNING RECOMMENDATION: [Feature/Question Name]

Decision: [Proceed / Defer / Needs more info]

Why: [2-3 sentences]

If proceeding:
  Complexity: Simple / Medium / Complex
  Suggested Sprint: Sprint N or Backlog
  Next step: Create REQ-00NN + ARCH-00N, then /go

Trade-offs accepted:
  [What we gain]
  [What we give up or defer]

Open questions before starting:
  - [Question 1]
  - [Question 2]
```

---

### Step 5: Optional — Create Artifacts

If the user confirms they want to proceed, offer to:
- Create a REQ document for the feature
- Create an ARCH document (if Medium or Complex)
- Add tasks to `agents/todo.md` for the next sprint
- Update `PROJECT_STATUS.md` with the new feature

Ask: "Should I create the REQ document now, or are there open questions to resolve first?"
