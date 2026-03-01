# agents/

This directory contains all multi-agent workflow artifacts for this project.

## Structure

```
agents/
├── todo.md              # Current task list — all agents read this first
├── workflow/            # How agents work: roles, patterns, git conventions, code standards
│   ├── agents.md        # Agent role definitions and interaction map
│   ├── patterns.md      # Daily activation patterns and feature workflows
│   ├── git-workflow.md  # Git worktrees, branching, commit conventions
│   ├── python-standards.md  # Code quality standards
│   └── session-logs.md  # How to write session logs
├── templates/           # Document templates — copy to create new artifacts
│   ├── requirement-template.md
│   ├── architecture-template.md
│   ├── adr-template.md
│   ├── handoff-template.md
│   ├── code-review-template.md
│   ├── test-plan-template.md
│   ├── bug-report-template.md
│   ├── milestone-review-template.md
│   └── session-log-template.md
├── requirements/        # REQ-NNNN.md — product requirements (created by Product Manager)
├── architecture/        # ARCH-NNN.md — design documents (created by Software Architect)
├── handoffs/            # HANDOFF-from-to-topic-date.md — inter-agent context
├── reviews/
│   └── code/            # TL-review-sprint-N-date.md — Tech Lead code reviews
└── session-logs/        # Per-agent session logs
    ├── claude-manager/
    ├── proj-mgr/
    ├── prod-mgr/
    ├── arch/
    ├── programmer/
    ├── qa-agent/
    └── tech-lead/
```

## Quick Reference

| Question | Read this |
|----------|-----------|
| What should I work on? | `agents/todo.md` |
| What's the project status? | `PROJECT_STATUS.md` |
| What are my responsibilities? | `agents/workflow/agents.md` |
| How do I format my work? | `agents/templates/[relevant-template].md` |
| What has the architect designed? | `agents/architecture/` |
| What are the requirements? | `agents/requirements/` |
| Who handed me work? | `agents/handoffs/` |
| What did TL say about my code? | `agents/reviews/code/` |
