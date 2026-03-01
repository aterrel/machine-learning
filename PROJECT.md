# PROJECT.md — Fill This In Before Running /bootstrap

This is the **only file you need to fill in** to initialize a multi-agent project.
Once complete, run `/bootstrap` in Claude Code and the system will generate everything else.

---

## Project Name

[Your project name here]

---

## Vision

[1-3 sentences. What is this? Who is it for? What problem does it solve?]

Example: "A personal AI assistant that helps software developers manage their email across 10+ accounts,
reducing time spent on email by 50% through intelligent prioritization and automated junk removal."

---

## High-Level Objectives

List the major user-facing outcomes you want to achieve. These become your initial REQ documents.
Write outcomes, not implementations. Use plain language. 3-8 objectives is ideal.

1. [Objective 1 — e.g., "Users can connect and manage multiple email accounts from one UI"]
2. [Objective 2 — e.g., "The system automatically prioritizes emails by importance"]
3. [Objective 3 — e.g., "Users can search emails across all accounts with natural language"]
4. [Objective 4]
5. [Objective 5]

---

## Tech Stack

Leave blank if you want the system to recommend a stack based on your objectives.

- **Backend language/framework**: [e.g., Python/FastAPI, TypeScript/Express, Go/Gin, Ruby/Rails]
- **Frontend**: [e.g., React/TypeScript, Vue, Next.js, none (API only), CLI only]
- **Database**: [e.g., PostgreSQL, MongoDB, SQLite, Redis, none]
- **Background jobs**: [e.g., Celery, BullMQ, Sidekiq, none]
- **Deployment target**: [e.g., Docker, Kubernetes, Vercel, AWS Lambda, local only]
- **Special infrastructure**: [e.g., pgvector, Elasticsearch, S3, none]

---

## Constraints

List anything the system must work within. These affect architecture decisions.

- [ ] [e.g., "Must work completely offline — no cloud AI inference"]
- [ ] [e.g., "Single developer — keep complexity low, avoid microservices"]
- [ ] [e.g., "Must handle 1M+ records — performance is critical"]
- [ ] [e.g., "Privacy-first — all data stays local"]
- [ ] [e.g., "Open source — avoid proprietary dependencies where possible"]

---

## Success Metrics

How will you know this project is successful? These become acceptance criteria.

- [e.g., "Reduces time spent on email by 50%"]
- [e.g., "95%+ accuracy on priority classification"]
- [e.g., "Sub-second search across 100k+ emails"]
- [e.g., "Setup takes < 10 minutes for a new user"]

---

## Out of Scope (optional but recommended)

Be explicit about what this project will NOT do. Prevents scope creep.

- [e.g., "Email composition and sending (read-only for now)"]
- [e.g., "Mobile app (web only in v1)"]
- [e.g., "Team/shared mailbox features"]

---

## Additional Context (optional)

Any other information the system should know:
- Existing codebase? Link to it or describe its state.
- Technical decisions already made?
- External APIs or services this will integrate with?
- Similar existing products to reference?
