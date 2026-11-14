# AGENTS.md — Coding Agent Contract

This file is the authoritative operating contract for coding agents working in this repository.

## Mission

Implement `embedfit`, a production-quality Python framework for few-shot adaptation of pretrained visual embeddings.

The primary pattern is:

```text
pretrained encoder
→ supervised embedding adaptation
→ frozen embeddings
→ lightweight classifier
```

The implementation must additionally expose embeddings for retrieval and risk-oriented evaluation.

## Required reading order

Before modifying code, read:

1. `CONTEXT.md`
2. `SPEC.md`
3. `ARCHITECTURE.md`
4. `IMPLEMENTATION_PLAN.md`
5. `ACCEPTANCE_CRITERIA.md`
6. `TEST_STRATEGY.md`
7. `SECURITY_PRIVACY.md`
8. all ADRs in `docs/adrs/`

Do not begin by writing large amounts of code without reconciling the task with these documents.

## Execution rules

### 1. Work one milestone at a time

Follow `IMPLEMENTATION_PLAN.md`.

Do not skip directly to advanced features such as hard-negative mining, ROI fusion, or domain-specific pretraining before the frozen baseline and SupCon paths are correct.

### 2. Tests ship with features

For every feature:

```text
implementation
→ unit tests
→ integration test where applicable
→ docs/example update
```

A feature is not complete until tests pass.

### 3. Offline CI is mandatory

Core tests must not download pretrained checkpoints.

Use `DummyBackbone` and synthetic image fixtures.

External-model tests must be explicitly marked.

### 4. Keep the core domain-neutral

Do not place use-case-specific behavior inside:

- backbones
- losses
- heads
- generic samplers
- model serialization

Use-case-specific material belongs in:

- examples
- documentation
- generic risk metrics
- optional relationship configuration

### 5. Prevent leakage

Never use validation/test examples for:

- representation training
- hard-negative mining
- threshold selection on test
- early stopping on test
- model selection on test

Respect group IDs and temporal boundaries.

### 6. Do not infer ambiguous semantics

Require explicit configuration for:

- positive class
- provider when a model name is ambiguous
- relationship rules
- multilabel behavior
- risk score direction

### 7. Public API quality

Every public function/class must have:

- type annotations
- docstring
- documented input semantics
- documented output semantics
- useful validation errors

### 8. No sensitive-data logging

Never log raw images or metadata values by default.

Logs may contain:

- sample counts
- dimensions
- opaque sample IDs if explicitly supplied
- aggregate metrics

### 9. Keep modules composable

Avoid monoliths. Prefer modules under ~500 lines when practical.

### 10. Preserve reproducibility

Every stochastic subsystem must accept/inherit a seed.

## Commands expected to work

```bash
ruff check .
pytest -q
python -m build
```

If mypy is enabled for a milestone:

```bash
mypy src/embedfit
```

## Decision policy

When the specification is silent:

1. choose the simplest design that preserves extension points;
2. avoid new runtime dependencies;
3. document the decision in an ADR if it changes architecture or public API;
4. update `SPEC.md` if behavior becomes normative.

## Prohibited shortcuts

Do not:

- mock away the central training behavior in integration tests;
- expose a test-only implementation as production code;
- silently fall back to another model provider;
- silently skip bad images;
- use accuracy as the only rare-event metric;
- hard-code ImageNet normalization globally;
- make network access part of default tests;
- add a heavy vector database to v0.1;
- invent use-case labels in core code.

## Definition of done

A milestone is done only when:

- implementation exists;
- relevant tests pass;
- docs are updated;
- acceptance criteria for the milestone pass;
- `TASKS.md` is updated;
- no untracked architectural change exists.
