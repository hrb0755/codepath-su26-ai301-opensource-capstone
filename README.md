# codepath-ai301-opensource-capstone
# Contribution 1: Support testing Megatron code as an inference engine (radixark/miles #400)

**Contribution Number:** 1  
**Student:** Ruobing Han
**Issue:** https://github.com/radixark/miles/issues/400  
**Status:** Phase I Complete

---

## Why I Chose This Issue

I chose issue #400, "Support testing Megatron code as an inference engine," in
[radixark/miles](https://github.com/radixark/miles) because it sits exactly at the
intersection of my background and where I want to grow: ML systems and LLM
post-training. Miles is an emerging, fast-moving RL post-training framework (backed
by InfiXAI, Ant Group, and the SGLang RL team) that couples SGLang for rollout with
Megatron-LM for training — the kind of large, promising new project I joined this
program to contribute to.

I'm interested in this because:
1. Post-training / RL infrastructure is my primary target area. I recently
   implemented the qwen3-next attention stack (linear attention + DeltaNet), so I'm
   comfortable reading LLM internals.
2. The issue is an additive enhancement with a contained surface: expose Megatron's
   existing inference path so it can be used as an inference engine for
   testing/evaluation, rather than a from-scratch rewrite.
3. The contribution can be validated on a single GPU with a small model, so it fits
   the hardware I can reliably access (a 3090 cluster) without needing scarce
   large-GPU time.
4. I want to learn how modern RLHF stacks wire rollout (SGLang) and training
   (Megatron) together, and this issue forces me to understand that boundary.

What "fixed" looks like: Megatron's inference can be driven through a standard
(OpenAI-compatible) interface and validated against evals, giving contributors a
Megatron-native inference path for testing. I have reached out on the issue to
confirm it is unclaimed (the only prior interest went quiet ~5 months ago) and to
pin down the acceptance criteria with the maintainers.

---

## Understanding the Issue

### Problem Description

Miles can train models with Megatron-LM and serve rollouts with SGLang, but there
is currently no supported way to use Megatron itself as an inference engine. This
makes it harder to validate Megatron-side behavior and run evaluations without
routing through SGLang. The issue asks to wire Megatron's existing inference code
behind a standard inference interface so it can be exercised directly for testing.

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
