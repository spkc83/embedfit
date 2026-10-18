# Implementation Plan

Implement milestones sequentially. Each milestone should end in a green test suite.

## Milestone 0 — Repository foundation

Deliver:

- `pyproject.toml`
- package imports
- lint/test configuration
- seed/device helpers
- base exceptions
- `DummyBackbone`
- synthetic image fixtures

Exit criteria:

```bash
pytest -q
ruff check .
python -m build
```

## Milestone 1 — Backbone abstraction and encoding

Deliver:

- `ImageBackbone`
- `TorchvisionBackbone`
- provider registry/resolver
- `EmbedFitModel`
- `encode`
- freeze/unfreeze
- preprocessing ownership

Tests:

- shape
- normalization
- freeze state
- single/batched images
- malformed image error

Do not add timm/HF until core abstraction works.

## Milestone 2 — Lightweight heads + frozen baseline

Deliver:

- `ClassificationHead`
- logistic-regression head
- label encoder/mapping
- `fit_head`
- `predict`
- `predict_proba`
- frozen Trainer mode

This milestone should produce the first actually useful model.

## Milestone 3 — Standard metrics + risk metrics

Deliver:

- classification metrics
- PR-AUC
- recall at FPR
- precision/recall at review rate
- precision/recall @ K
- dollar recall
- threshold selection

All metric functions must have deterministic unit tests with manually checkable examples.

## Milestone 4 — Data splitting and leakage protection

Deliver:

- few-shot sampler
- grouped split helper
- temporal split helper
- duplicate ID overlap checker
- clear documentation

## Milestone 5 — Supervised contrastive training

Deliver:

- balanced class batch sampler
- SupCon loss
- differentiable training encoder path
- optimizer/scheduler
- representation-phase checkpoints
- phase transition to logistic-regression head

This is the core research capability.

## Milestone 6 — Pairwise contrastive training

Deliver:

- `ImagePair`
- bounded pair sampler
- balanced positives/negatives
- cosine pair loss
- pair training loop

## Milestone 7 — Additional backbone providers

Deliver:

- timm adapter
- Hugging Face vision adapter
- pooling policies
- serialization metadata

Network-dependent tests must remain opt-in.

## Milestone 8 — Serialization

Deliver:

- `save_pretrained`
- local `from_pretrained`
- head persistence
- exact label-order persistence
- numerical equivalence test

## Milestone 9 — Retrieval

Deliver:

- brute-force cosine index
- top-k search
- exclude-self
- metadata passthrough
- retrieval metrics

## Milestone 10 — Few-shot evaluator

Deliver:

- N-per-class sampling
- repeated seeds
- frozen / CE (linear probe then fine-tune, ADR 0012) / SupCon benchmark protocols
- tidy results
- aggregate mean/std

## Milestone 11 — Hard-negative mining

Only after pair/SupCon paths are reliable.

Deliver:

- nearest different-class negative mining
- no-gradient refresh
- train-only scope
- configurable refresh schedule

## Milestone 12 — Documentation and hardening

Deliver:

- quickstart
- financial-risk synthetic example
- leakage guide
- benchmark guide
- API docs
- error-message audit
- privacy/logging audit

## Required in v0.2

- triplet learning (`SPEC.md` §3: deferred from v0.1, required by v0.2)

## Optional v0.2 items

- calibration methods
- prototype head
- FAISS backend
- multilabel classification
- ROI/multi-crop representation
- self-supervised domain adaptation
