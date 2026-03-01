You are the Claude Manager activating the sprint execution loop. Follow this sequence exactly.

---

## SPRINT EXECUTION SEQUENCE

### Step 1: Orient

Read in order:
1. `PROJECT_STATUS.md` — current sprint, pending/completed tasks, blockers
2. `agents/todo.md` — find all pending `[ ]` and in-progress `[>]` tasks
3. `git log --oneline -10` — see recent activity
4. `agents/handoffs/` — check for any pending handoffs

Identify: What sprint are we in? What's done? What's blocked? What's ready to start?

---

### Step 2: Correct the Record

Fix any inconsistencies between todo.md, PROJECT_STATUS.md, and the actual git history:

- Mark tasks `[x]` if git history shows they're committed
- Unblock `[!]` tasks if their dependencies are now met
- Move completed sprint items to the Completed section
- Add any tasks discovered in handoffs that aren't in todo.md

Commit corrections:
```
[Proj-Mgr] Sync todo.md and PROJECT_STATUS.md with current state
```

---

### Step 3: Evaluate Sprint Health

Determine which of these states we're in:

**A. Sprint In Progress** — P0 tasks are pending or in-progress. Activate sprint agents.

**B. Sprint Complete, Needs Tech Lead Review** — All P0 tasks are done but no TL review yet. Activate Tech Lead.

**C. Sprint Complete + TL Approved** — Ready to close sprint and open next one. Run retrospective, update PROJECT_STATUS.md, open Sprint N+1.

**D. Blocked** — Dependencies on external input (credentials, external API access, etc.). Document blocker in PROJECT_STATUS.md, note what's needed, stop.

---

### Step 4A: Activate Sprint Agents (if State A)

Based on todo.md, activate the appropriate agents in this order:

**Always check first**: Product Manager
- If any REQ documents are missing for planned sprint features, create them now
- Creates REQ documents, then hands off to Software Architect

**Then**: Software Architect
- If any ARCH documents are missing for features with REQs, create them now
- Creates ARCH documents, then creates handoff to Programmer

**Then**: Programmer
- Read REQ + ARCH documents and handoffs addressed to Programmer
- Implement the next unblocked task from todo.md
- Run tests before committing

**Parallel**: QA Agent
- Write test plans for any REQ that has been implemented
- Write tests for completed implementations

**Scope per activation**: Focus on P0 tasks only. P1 tasks are backlog.

For each agent activated:
1. Read their handoffs
2. Create a session log before starting
3. Commit work with correct prefix ([Prod-Mgr], [Arch], [Prog], [QA])
4. Update todo.md marking tasks done
5. Create handoff for next agent

---

### Step 4B: Activate Tech Lead (if State B)

Tech Lead sprint review:
1. `git log --oneline` since last sprint boundary
2. Review all new/modified source files
3. Create `agents/reviews/code/TL-review-sprint[N]-[DATE].md`
4. Issue verdict: Approved / Changes Required / Conditional Approval
5. If Changes Required: create handoff to Programmer and fix before closing
6. Commit: `[Tech Lead] TL-review: Sprint N code review`

**Sprint may NOT close without Tech Lead Approved or Conditional Approval.**

---

### Step 4C: Close Sprint and Open Next (if State C)

1. Update PROJECT_STATUS.md:
   - Move all completed tasks to Completed section
   - Add sprint to Sprint History
   - Define Sprint N+1 goal and pending tasks

2. Run milestone review: create `agents/session-logs/proj-mgr/sprint-[N]-retro-[DATE].md`

3. Commit: `[Proj-Mgr] Close Sprint N, open Sprint N+1`

4. Then activate Sprint N+1 agents (go to Step 4A for new sprint)

---

### Step 5: Update State

After all agent work:
1. Update `agents/todo.md` — all changes reflected
2. Update `PROJECT_STATUS.md` — sprint status, agent assignments, blockers
3. Commit if not already committed in agent steps

---

### Step 6: Summary

Output:
- What was completed this session
- Current sprint status (N tasks done, M remaining, K blocked)
- Which agents were activated and what they produced
- What the next `/go` invocation will pick up
