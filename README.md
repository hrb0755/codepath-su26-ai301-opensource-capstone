# codepath-ai301-opensource-capstone
# Contribution 1: Support testing Megatron code as an inference engine (radixark/miles #400)

**Contribution Number:** 1  
**Student:** Ruobing Han
**Issue:** https://github.com/radixark/miles/issues/400  
**Status:** Phase III In Progress (Build)

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
   the hardware I can reliably access without needing scarce large-GPU time.
4. I want to learn how modern RLHF stacks wire rollout (SGLang) and training
   (Megatron) together, and this issue forces me to understand that boundary.

What "fixed" looks like: Megatron's inference can be driven through a standard
(OpenAI-compatible) interface and validated against evals, giving contributors a
Megatron-native inference path for testing. I have reached out on the issue to
confirm it is unclaimed (the only prior interest went quiet ~5 months ago) and to
pin down the acceptance criteria with the maintainers.

---

## Understanding the Issue

> **Note on issue type.** #400 is labeled `enhancement` + `help wanted`, not a bug.
> "Reproducing" it means showing that the requested capability is *absent*, not
> triggering a crash.

### Problem Description

Miles can train models with Megatron-LM and serve rollouts with SGLang, but there
is currently no supported way to use Megatron itself as an inference engine. This
makes it harder to validate Megatron-side behavior and run evaluations without
routing through SGLang. The issue asks to wire Megatron's existing inference code
behind a standard inference interface so it can be exercised directly for testing.

Maintainer scope signals from the issue thread:
- `gongyisheng` pointed at Megatron-LM's own inference test
  (`tests/unit_tests/test_inference.py`) and asked "will it be enough to use?" —
  i.e. reuse Megatron's existing inference stack rather than build a new one.
- `fzyzcjy` (author) clarified the target: "a reasonably fast one w/ openai
  endpoint etc, i.e. the normal requirements."

### Expected Behavior

A contributor should be able to take a model that miles built with Megatron-LM and
**generate text from it directly** (autoregressive decoding) for testing/evaluation
— ideally behind an OpenAI-compatible endpoint — without standing up a separate
SGLang server.

### Current Behavior

All generation in miles is delegated to SGLang. The Megatron side only does
training and no-grad forward passes (for log-probs). There is no autoregressive
generation, no sampling loop, and no Megatron inference engine wired up. The
OpenAI-compatible API is SGLang's own server; miles only puts a load-balancing
proxy in front of it.

### Affected Components

- `miles/rollout/generate_hub/` — the low-level `generate()` path (SGLang HTTP).
- `miles/backends/megatron_utils/` — Megatron model setup + `forward_only()`; the
  natural home for a new inference path.
- `miles_plugins/models/hf_attention.py` — already imports
  `megatron.core.inference.contexts.BaseInferenceContext` as a *type*, so the model
  layers are already inference-context-aware (a useful building block).
- `tests/e2e/megatron/` — where a new Megatron inference test would live.
- `miles/router/` — SGLang reverse proxy (pattern to reuse for an endpoint later).

---

## Reproduction Process

### Environment Setup

The project is Docker-first (`radixark/miles:dev`), but the GPU server I work on
has no container runtime, and installing a system Docker daemon on a shared,
multi-tenant training box is not acceptable. I therefore reproduced the runtime in
a conda env (`mega`, Python 3.12) by mirroring the project's `docker/Dockerfile`.

The one thing that actually matters is ABI alignment — miles pins patched, prebuilt
CUDA-13 kernels, so the torch version has to match them exactly (the docs call a
mismatch the "#1 source of bug reports"). The recipe that worked:

1. `torch==2.11.0` + cu130 (matches sglang v0.5.12's pin; the cu130 wheels are
   `manylinux_2_28`).
2. Prebuilt `cp312` wheels installed `--no-deps` from the
   `yueming-yuan/miles-wheels@cu130-x86_64-v0.5.12` release: `flash-attn` 2.7.4,
   `apex`, `transformer_engine` (+ `transformer_engine_torch`), plus
   `transformer_engine_cu13==2.12.0` (the compiled libs, from PyPI) and `onnxscript`.
3. Megatron-LM `radixark/Megatron-LM@miles-main` (megatron-core 0.16.0rc0),
   `pip install -e . --no-deps`.
4. miles: `pip install -r requirements.txt` with torch pinned via a `--constraint`
   file (so `torchft-nightly` can't pull a different torch), then re-force
   `numpy<2`, then `pip install -e . --no-deps`.

Challenges and how I solved them:
- **No Docker on a shared box** → mirrored the Dockerfile into the `mega` conda env.
- **ABI mismatch risk** → pinned torch to exactly `2.11.0/cu130` to match the wheels;
  `flash-attn`, `transformer_engine` (incl. FP8), and `apex` all verified on an H100.
- **`torchft-nightly` tried to move torch** → installed requirements with a torch
  `--constraint`.
- **transitive deps bumped numpy to 2.x** → re-pinned `numpy<2` afterward.
- **JuiceFS is slow for many-small-file installs** → torch/TE installs take a while;
  expect minutes, not seconds.

Result: `import miles` works from any directory, and a single-GPU smoke test drives
Megatron's own inference engine end-to-end with **no SGLang installed** (see
Reproduction Evidence). SGLang is intentionally omitted — it is not needed for
Megatron-native inference.

### Steps to Reproduce

Reproducing this enhancement means showing, on demand, that miles has no supported
way to run Megatron as an inference engine. A dependency-free script does this by
inspecting the source (`w2/repro/reproduce_issue_400.py`, no GPU required):

1. From this repo, run:
   ```bash
   python w2/repro/reproduce_issue_400.py --miles-root /path/to/miles
   ```
2. The script confirms three conditions, all of which hold today:
   1. **Generation goes only through SGLang** — the generate path posts to SGLang's
      HTTP `/generate` endpoint (`miles/rollout/generate_hub/single_turn.py:20,43`,
      `.../multi_turn.py:31`).
   2. **No Megatron inference engine is used** — none of
      `StaticInferenceEngine` / `DynamicInferenceEngine` /
      `TextGenerationController` / `GPTInferenceWrapper` appears anywhere; the only
      `megatron.core.inference` reference is a type-only import
      (`miles_plugins/models/hf_attention.py:6`).
   3. **No CLI flag selects Megatron as the inference engine** — only generic
      python-path overrides exist (`miles/utils/arguments.py:284,447`), and both
      default to SGLang.
3. **Expected (desired):** a supported way to generate text from miles' Megatron
   model. **Actual:** none — generation requires SGLang.
4. The check is static source inspection, so it reproduces identically on every run.

Manual cross-check (no script):
```bash
grep -rn "core.inference\|TextGenerationController\|InferenceEngine" miles/ miles_plugins/
grep -n "/generate" miles/rollout/generate_hub/single_turn.py
```

### Reproduction Evidence

- **Working branch (my fork):** https://github.com/hrb0755/miles/tree/fix-issue-400
- **Reproduction script:** `w2/repro/reproduce_issue_400.py` → prints
  `GAP REPRODUCED` (exit 0).
- **Findings:** miles' generation is SGLang-only; the Megatron side has model setup
  + `forward_only()` (`miles/backends/megatron_utils/model.py:193-325`) but no
  autoregressive generation.
- **Feasibility, verified:** I confirmed the fix is *integration, not a rewrite* by
  running Megatron-LM's own inference stack standalone on one H100 —
  `GPTInferenceWrapper` → `TextGenerationController` → `StaticInferenceEngine`
  generating tokens from a small model, **with no SGLang**
  (`w2/repro/megatron_inference_smoke.py`, requires the full `mega` env). This is
  the exact API the contribution will wire miles' model into.

---

## Solution Approach

### Analysis

The root cause is a missing capability, not a defect: miles never constructs or
drives Megatron-LM's inference engine. Generation is implemented exclusively as an
HTTP call into SGLang (`miles/rollout/generate_hub/single_turn.py:20,43`), and the
Megatron actor only ever runs a no-grad forward for log-probs
(`miles/backends/megatron_utils/model.py:193-325`) — there is no sampling loop, KV
cache, or engine wrapper on the Megatron model.

### Proposed Solution

Wire miles' already-built Megatron model into Megatron-LM's existing inference stack
(`megatron.core.inference`: `GPTInferenceWrapper` → `TextGenerationController` →
`StaticInferenceEngine`) and add a small single-GPU test that drives it. Keep the
first version testable on one GPU with a small model (TP=PP=1, basic sampling);
defer distributed (TP/PP) generation and the OpenAI-compatible endpoint to
follow-ups, confirming acceptance criteria with maintainers first.

### Implementation Plan

Using the UMPIRE framework:

**Understand:** miles can train with Megatron-LM but can only generate text through
SGLang. There is no supported path to drive miles' Megatron model as an inference
engine for testing/eval. #400 asks to add one (ideally fast, with an OpenAI
endpoint).

**Match:**
- Megatron-LM ships a complete inference stack in `megatron.core.inference`
  (`GPTInferenceWrapper`, `TextGenerationController`, `StaticInferenceEngine`) plus
  a reference test `tests/unit_tests/test_inference.py` — the maintainer pointed
  here, and I have run this stack directly (see Reproduction Evidence).
- miles already builds the Megatron GPT model and parallel state
  (`miles/backends/megatron_utils/model.py`, `initialize.py`, `parallel.py`) and
  runs no-grad forwards (`forward_only`, `model.py:193-325`).
- miles' attention already accepts a `BaseInferenceContext`
  (`miles_plugins/models/hf_attention.py:180`).
- The SGLang integration shows the rollout/endpoint seam to reuse later
  (`miles/ray/rollout/rollout_server.py`, `miles/router/router.py`, and the
  `--custom-generate-function-path` hook at `miles/utils/arguments.py:447`).
- `tests/e2e/sglang/test_chat_input_ids_equivalence.py` is a template for
  validating a new engine's output by equivalence.

**Plan (minimal, test-first):**
1. Add `miles/backends/megatron_utils/inference.py` that, given miles' built model
   + parallel state, constructs `GPTInferenceWrapper` → `TextGenerationController`
   → `StaticInferenceEngine`.
2. Expose a minimal `generate(prompts, sampling_params)` helper returning
   tokens/text (greedy + basic temperature/top-p first).
3. Add a single-GPU, tiny-model test under `tests/e2e/megatron/` (mirroring
   upstream `test_inference.py`) that generates via the Megatron engine and
   validates the output. Register it with `register_cuda_ci(...)`.
4. Keep v1 at TP=1, PP=1, small model; document distributed generation and the
   OpenAI-compatible endpoint as explicit follow-ups.

**Implement:** Phase III, on branch
https://github.com/hrb0755/miles/tree/fix-issue-400 (placeholder until commits land).

**Review:** Self-review against `docs/developer/contributor-guide.md`:
conventional-commit messages; `pre-commit run --all-files` (ruff/black/isort, line
length 119); use `ParallelState` instead of direct `mpu.get_*`; use
`load_hf_config` / `load_tokenizer` instead of bare `AutoConfig`/`AutoTokenizer`.
Run the PR checklist (pre-commit green, tests added, `python3 train.py --help`
parses if any flag is added).

**Evaluate:** the new single-GPU test passes; generated tokens match a reference
within tolerance; `pytest tests/fast -x` stays green; optionally an equivalence
check vs SGLang following `test_chat_input_ids_equivalence.py`.

---

## Testing Strategy

### Unit Tests

- [x] CPU (`tests/fast/backends/megatron_utils/test_megatron_inference_adapter.py`):
      8 unit tests for the HF→Megatron tokenizer adapter and the tokenizer
      dispatch helper — tokenize/detokenize/bos/eos/eod mapping, `skip_special_tokens`
      forwarding, pass-through of Megatron-style tokenizers, and error cases.
      All 8 pass locally (run with `--noconftest`, see Challenges). No GPU/Megatron
      needed (the module defers Megatron imports), so these run in the CPU CI stage.

### Integration Tests

- [x] GPU (`tests/fast-gpu/test_megatron_inference.py`): builds a tiny GPTModel
      in-process (TP=PP=1), drives it through `build_inference_engine` + `generate`,
      and asserts it returns the requested number of in-range tokens. Registered
      with `register_cuda_ci(..., suite="stage-b-2-gpu-h200", labels=["megatron"])`.
      Written and lint-clean; **GPU validation pending** (the shared cluster's GPUs
      were fully occupied by another training job — see Challenges). The identical
      engine path was already proven on an H100 in Phase II
      (`w2/repro/megatron_inference_smoke.py`).
- [ ] (Follow-up) Output-equivalence cross-check vs a reference (HF or SGLang).

### Manual Testing

The Phase II single-GPU smoke test (`w2/repro/megatron_inference_smoke.py`) drove
the same `megatron.core.inference` stack end-to-end on an H100. Phase III formalizes
that into the in-repo test above, routed through the new `inference.py` module.

---

## Implementation Notes

### Week 2 Progress

- Synced fork to upstream (was 15 commits behind): web "Sync fork" for the remote,
  `git merge --ff-only upstream/main` locally; created branch `fix-issue-400`.
- Mapped the architecture: generation is SGLang-only; Megatron does training +
  `forward_only` log-probs; no Megatron generation path exists.
- Wrote a deterministic, dependency-free reproduction script
  (`w2/repro/reproduce_issue_400.py`) that proves the gap (exit 0).
- Built a working runtime in the `mega` conda env (Docker wasn't available),
  mirroring the project Dockerfile with torch 2.11.0/cu130 + prebuilt cu13 wheels +
  Megatron-LM `miles-main` + miles. Documented every pin and fix above.
- Validated the plan by running Megatron's `megatron.core.inference` stack
  standalone on one H100 (`w2/repro/megatron_inference_smoke.py`) — confirming the
  contribution is integration, not building an engine.
- Drafted the UMPIRE solution plan (minimal, test-first scope).

### Week 3 Progress (Phase III: Build)

**What I built:**
- Added `miles/backends/megatron_utils/inference.py` — drives a miles-built
  Megatron `GPTModel` through Megatron-LM's own inference stack
  (`GPTInferenceWrapper` → `TextGenerationController` → `StaticInferenceEngine`):
  - `build_inference_engine(model, tokenizer, ...)` — unwraps the actor's
    `list[DDP]` to the underlying `GPTModel`, derives `padded_vocab_size` /
    `params_dtype`, assembles `InferenceWrapperConfig`, and returns a ready engine.
  - `generate(engine, prompts, ...)` — thin wrapper over `engine.generate`.
  - `HFTokenizerInferenceAdapter` — bridges miles' HuggingFace tokenizer
    (`encode`/`decode`/`bos_token_id`/`eos_token_id`) to the
    `tokenize`/`detokenize`/`bos`/`eod` interface the controller expects.
  - Megatron imports are deferred into functions so the adapter is unit-testable
    on CPU without Megatron installed.
- Added CPU unit tests (8) and a single-GPU end-to-end generation test.

**Challenges faced:**
- The repo's root `tests/conftest.py` transitively `import sglang`, which I had
  deliberately not installed (it isn't needed for Megatron-native inference). Any
  plain `pytest` therefore fails at conftest import. Fix: run the CPU test with
  `pytest --noconftest -o addopts=""` (it uses no conftest fixtures); the GPU test
  also runs standalone via `python3` (the e2e convention). CI has sglang, so this
  is a local-env-only workaround.
- The HF tokenizer interface doesn't match what `TextGenerationController` calls
  (`bos`/`eod`/`tokenize`/`detokenize`), so I wrote the small adapter above.
- The shared cluster's 8 GPUs were fully occupied by another training job, so the
  in-repo GPU test couldn't be run yet; the same engine path was already validated
  on an H100 in Phase II.

**Self-review:** `ruff` / `black` / `isort` clean on all three files; the
`ban-mpu-get` and `ban-bare-auto-loaders` pre-commit hooks pass (no `mpu.get_*`,
no bare `AutoConfig`/`AutoTokenizer`). Per-repo git identity set so commits are
correctly attributed.

### Code Changes

- **Files added:**
  - `miles/backends/megatron_utils/inference.py` (engine + tokenizer adapter)
  - `tests/fast/backends/megatron_utils/test_megatron_inference_adapter.py` (CPU)
  - `tests/fast-gpu/test_megatron_inference.py` (single-GPU e2e)
- **Branch:** https://github.com/hrb0755/miles/tree/fix-issue-400
- **Key commit:** `feat(megatron): add Megatron inference engine for testing (#400)`
  (`fe079e7ca`).
- **Approach decisions:** kept the first PR focused on a standalone, testable
  engine builder rather than also wiring the actor/CLI/OpenAI-endpoint, which are
  noted as follow-ups; reused miles' training tokenizer via an adapter for
  consistency; scoped v1 to TP=PP=1.

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

- Issue #400: https://github.com/radixark/miles/issues/400
- Megatron-LM inference stack: `megatron.core.inference` and its
  `tests/unit_tests/test_inference.py`
- miles contributor guide: `docs/developer/contributor-guide.md`
- miles Dockerfile / CI (canonical environment recipe): `docker/Dockerfile`,
  `.github/workflows/_run-ci.yml`
- SGLang equivalence-test pattern:
  `tests/e2e/sglang/test_chat_input_ids_equivalence.py`
