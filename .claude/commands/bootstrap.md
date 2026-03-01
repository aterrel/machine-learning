You are the Claude Manager for this project. Execute the full project initialization sequence described below. Do not ask for confirmation — complete all steps and commit.

---

## BOOTSTRAP SEQUENCE

### Step 0: Validate PROJECT.md

Read `PROJECT.md`. Check that these sections are filled in (not placeholder text):
- Project Name (not "[Your project name here]")
- Vision (not "[1-3 sentences...]")
- At least 2 High-Level Objectives (not "[Objective N...]")

If any are missing or still contain placeholder text, STOP and tell the user exactly which sections need to be filled in. Do not proceed.

If validation passes, continue.

---

### Step 1: Orient to Project State

Run:
```bash
git log --oneline -5
git status
```

Check if this is a fresh repo (0-1 commits) or an existing project. If existing:
- Read `PROJECT_STATUS.md` if it exists
- Read `agents/todo.md` if it exists
- Do NOT overwrite files that already have real content — update them instead

---

### Step 2: Generate CLAUDE.md

Create a project-specific `CLAUDE.md` based on `CLAUDE.md.template` and the contents of `PROJECT.md`. Fill in all placeholders:
- `[PROJECT_NAME]` → Project Name from PROJECT.md
- `[PROJECT_VISION]` → Vision from PROJECT.md
- `[BACKEND_TECH]` → Backend from Tech Stack section (or recommend based on objectives if blank)
- `[FRONTEND_TECH]` → Frontend from Tech Stack section
- `[DATABASE_TECH]` → Database from Tech Stack section
- `[LINT_CMD]` → appropriate lint command for the stack (e.g., `ruff check src/` for Python, `npm run lint` for TypeScript)
- `[TEST_CMD]` → appropriate test command (e.g., `pytest tests/` or `npm test`)
- `[BUILD_CMD]` → appropriate build command (e.g., `docker-compose up -d` or `npm run dev`)

If Tech Stack is blank, choose a modern, widely-used stack appropriate for the objectives and document your reasoning in CLAUDE.md.

---

### Step 3: Create Directory Structure

Ensure these directories exist (create with `.gitkeep` if empty):

```
agents/
├── requirements/        # REQ-NNNN.md files
├── architecture/        # ARCH-NNN.md files
├── handoffs/            # Inter-agent handoff documents
├── reviews/
│   └── code/            # Tech Lead code review documents
└── session-logs/
    ├── claude-manager/
    ├── proj-mgr/
    ├── prod-mgr/
    ├── arch/
    ├── programmer/
    ├── qa-agent/
    └── tech-lead/
```

---

### Step 4: Generate PROJECT_STATUS.md

Create `PROJECT_STATUS.md` with:

**Header**: Last Updated = today, Current Sprint = Sprint 1

**Project Health table**: Set everything to Red/Yellow (it's a new project) except documentation.

**Sprint 1 Goal**: Derived from the first 2-3 P0 objectives in PROJECT.md — focus on "make the core loop work end-to-end."

**Sprint Backlog - Pending**:
For each high-level objective, add tasks:
- `[ ] [Prod-Mgr] Create REQ-000N: [Objective title]`
- `[ ] [Arch] Create ARCH-001: Overall System Architecture`
- `[ ] [Arch] Create ARCH-002: [First major technical challenge]`
- `[ ] [Prog] Implement database schema / migrations`
- `[ ] [Prog] Implement [first objective] service`
- `[ ] [QA] Create test plans for REQ-0001, REQ-0002`
- `[ ] [Tech Lead] Sprint 1 code review (end of sprint)`

**Agent Assignments**: All set to "Pending" or "Blocked"

**Key File Locations table**: Fill in based on actual paths

---

### Step 5: Act as Product Manager — Generate REQ Documents

For each objective in PROJECT.md (in order), create `agents/requirements/REQ-000N.md` using `agents/templates/requirement-template.md`.

**Rules for REQ generation**:
- REQ-0001 = Objective 1, REQ-0002 = Objective 2, etc.
- First objective is always P0. Subsequent ones are P0 if foundational, P1 if dependent.
- Write 4-8 specific, testable functional requirements per REQ
- Write 2-3 non-functional requirements (performance, security, reliability)
- Write 5+ acceptance criteria using "User can X and receives Y" format
- Fill in the API/Interface Design section with actual endpoint names or method signatures
- Fill in Dependencies between REQs where they exist

Be specific. Vague requirements make implementation and testing impossible.

---

### Step 6: Act as Software Architect — Generate ARCH-001

Create `agents/architecture/ARCH-001.md` — Overall System Architecture.

Cover:
- High-level component diagram (ASCII art is fine)
- Module/service boundaries and responsibilities
- Data model overview (main entities and relationships)
- API layer design
- Background processing design (if applicable)
- External integrations (if applicable)
- Security model overview
- Development environment setup
- Key architecture decisions with rationale

Use `agents/templates/architecture-template.md` format.

If the project has a complex technical subsystem (e.g., ML pipeline, sync engine, real-time updates), also create `ARCH-002.md` for that subsystem.

---

### Step 7: Generate agents/todo.md

Create `agents/todo.md` with format:

```markdown
# agents/todo.md — [PROJECT_NAME] Task Tracker

## Sprint 1 — [Goal] (YYYY-MM-DD → YYYY-MM-DD)

### Product Manager
- [ ] [Prod-Mgr] Create REQ-0001: [Objective 1 title]
...

### Software Architect
- [ ] [Arch] ARCH-001 review and handoff to Programmer
- [x] [Arch] ARCH-001: Overall System Architecture (just created)
...

### Programmer
- [!] [Prog] [First implementation task] (blocked — await ARCH handoff)
...

### QA Agent
- [ ] [QA] Write test plans for REQ-0001, REQ-0002 (can start)
- [!] [QA] Write tests (blocked — await implementation)

### Tech Lead
- [!] [Tech-Lead] Sprint 1 code review (blocked — triggers at sprint end)

## Backlog (Future Sprints)
[P1 objectives and future features]

## Completed (Sprint 0)
- [x] [Claude-Mgr] Bootstrap initialization
- [x] [Claude-Mgr] PROJECT_STATUS.md created
- [x] [Claude-Mgr] REQ-0001 through REQ-000N created
- [x] [Claude-Mgr] ARCH-001 created
```

---

### Step 8: Create Arch → Programmer Handoff

Create `agents/handoffs/HANDOFF-arch-to-prog-all-reqs-[DATE].md`:

- List all REQ documents just created
- List all ARCH documents just created
- Specify the implementation phase order (what to build first)
- Include acceptance criteria from the REQ documents
- Flag any open architecture questions

---

### Step 9: Create Session Log

Create `agents/session-logs/claude-manager/[YYYY-MM-DD]-001.md`:
- Summarize what was created
- List all files created/modified
- Note next steps for each agent

---

### Step 10: Commit Everything

Stage and commit in two batches:

**Batch 1** (project config and docs):
```
git add CLAUDE.md .gitignore .env.example [any project scaffolding]
git commit -m "Initialize [PROJECT_NAME] project structure"
```

**Batch 2** (agent workflow infrastructure):
```
git add PROJECT_STATUS.md agents/
git commit -m "[Claude Manager] Bootstrap: initialize agent workflow for [PROJECT_NAME]

- Generated CLAUDE.md from PROJECT.md
- Created PROJECT_STATUS.md (Sprint 1 defined)
- Created agents/todo.md (full Sprint 1 backlog)
- Created REQ-0001 through REQ-000N from objectives
- Created ARCH-001: Overall System Architecture
- Created Arch → Programmer handoff

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Step 11: Print Summary

After committing, output a summary in this format:

```
═══════════════════════════════════════════════════════
  BOOTSTRAP COMPLETE — [PROJECT_NAME]
═══════════════════════════════════════════════════════

📁 Created:
  ✓ CLAUDE.md (project-specific)
  ✓ PROJECT_STATUS.md (Sprint 1 defined)
  ✓ agents/todo.md (N tasks)
  ✓ agents/requirements/REQ-0001.md through REQ-000N.md
  ✓ agents/architecture/ARCH-001.md

📋 Sprint 1 Goal: [Sprint goal]

🚀 Agents ready to activate:
  → Product Manager: REQ-000N documents still needed (objectives N+1...)
  → Programmer: Read ARCH-001 + handoff, then begin implementation
  → QA Agent: Write test plans for REQ-0001, REQ-0002

🔒 Blocked until Programmer delivers:
  → Tech Lead: Sprint 1 code review

💡 Next command: /go  (activates the sprint execution loop)
═══════════════════════════════════════════════════════
```
