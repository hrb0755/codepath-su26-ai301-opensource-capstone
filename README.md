# codepath-ai301-opensource-capstone
# Contribution 1: Add a PostgreSQL connector for Moss (usemoss/moss #168)

**Contribution Number:** 1  
**Student:** Ruobing Han
**Issue:** https://github.com/usemoss/moss/issues/168  
**Status:** Phase I Complete

---

## Why I Chose This Issue

I chose issue #168, "Add moss-connector-postgres," in
[usemoss/moss](https://github.com/usemoss/moss) because it is a clean, well-scoped
first contribution in an emerging AI-infrastructure project that I can complete
confidently while still learning something real. Moss is a local, embedded
retrieval/search layer for production AI — sub-10ms hybrid (semantic + keyword)
search with no separate vector database — i.e. the retrieval substrate that RAG and
agent systems depend on. It is young (created Oct 2025), YC-backed, and actively
developed, which is exactly the kind of promising new project I joined this program
to contribute to.

I'm interested in this because:
1. Retrieval / RAG infrastructure is adjacent to my target areas (ML infra and
   agent/LLM development), so the work builds directly toward where I want to grow.
2. The scope is bounded and pattern-based: Moss already ships several connectors, so
   this issue is "add one more" — copy the connector template, implement iteration
   over a PostgreSQL source via psycopg (v3), and add tests and a short README.
   There is a clear "done" and an existing pattern to mirror.
3. It is pure Python and lives in an isolated connector package, so I can be
   productive without first having to understand the entire codebase — a good fit
   for my first open-source PR.
4. I want to learn how a production retrieval layer ingests and syncs external data
   sources, and writing a connector is the most direct way to learn that.

What "fixed" looks like: Moss gains a working PostgreSQL connector that pulls rows
from a Postgres database into Moss so they can be indexed and searched, following
the existing connector interface, with tests and documentation. I have reached out
on the issue to confirm it is unclaimed and to check the contribution requirements
(the issue mentions a short demo video and a CLA).

---

## Understanding the Issue

### Problem Description

Moss is an embedded retrieval/search layer that indexes data through "connectors,"
each of which pulls records from a specific source into Moss. Today there is no
connector for PostgreSQL, so users cannot index data that lives in a Postgres
database. This issue asks for a new `moss-connector-postgres` package that reads
rows from PostgreSQL (via psycopg) and feeds them into Moss, mirroring the existing
connectors, with tests and a README.

### Expected Behavior

[What should happen?]

### Current Behavior

[What actually happens?]

### Affected Components

[Which parts of the codebase are involved?]

---

## Reproduction Process

### Environment Setup

[Notes on setting up your local development environment - challenges you faced, how you solved them]

### Steps to Reproduce

1. [Step 1]
2. [Step 2]
3. [Observed result]

### Reproduction Evidence

- **Commit showing reproduction:** [Link to commit in your fork]
- **Screenshots/logs:** [If applicable]
- **My findings:** [What you discovered during reproduction]

---

## Solution Approach

### Analysis

[Your analysis of the root cause - what's causing the issue?]

### Proposed Solution

[High-level description of your fix approach]

### Implementation Plan

Using UMPIRE framework (adapted):

**Understand:** [Restate the problem]

**Match:** [What similar patterns/solutions exist in the codebase?]

**Plan:** [Step-by-step implementation plan]
1. [Modify file X to do Y]
2. [Add function Z]
3. [Update tests]

**Implement:** [Link to your branch/commits as you work]

**Review:** [Self-review checklist - does it follow the project's contribution guidelines?]

**Evaluate:** [How will you verify it works?]

---

## Testing Strategy

### Unit Tests

- [ ] Test case 1: [Description]
- [ ] Test case 2: [Description]
- [ ] Test case 3: [Description]

### Integration Tests

- [ ] Integration scenario 1
- [ ] Integration scenario 2

### Manual Testing

[What you tested manually and results]

---

## Implementation Notes

### Week [X] Progress

[What you built this week, challenges faced, decisions made]

### Week [Y] Progress

[Continue documenting as you work]

### Code Changes

- **Files modified:** [List]
- **Key commits:** [Links to important commits]
- **Approach decisions:** [Why you chose certain approaches]

---

## Pull Request

**PR Link:** [GitHub PR URL when submitted]

**PR Description:** [Draft or final PR description - much of the content above can be adapted]

**Maintainer Feedback:**
- [Date]: [Summary of feedback received]
- [Date]: [How you addressed it]

**Status:** [Awaiting review / Iterating / Approved / Merged]

---

## Learnings & Reflections

### Technical Skills Gained

[What you learned technically]

### Challenges Overcome

[What was hard and how you solved it]

### What I'd Do Differently Next Time

[Reflection on your process]

---

## Resources Used

- [Link to helpful documentation]
- [Tutorial or Stack Overflow post that helped]
- [GitHub issues or discussions that helped]
