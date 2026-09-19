# Task Board

The coding agent should update this file as implementation progresses.

Legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` complete
- `[!]` blocked

## M0 Foundation

- [x] Create bootstrap `pyproject.toml` (agent must expand dependencies/config)
- [x] Add package metadata and optional dependency groups
- [x] Add ruff configuration
- [x] Add pytest configuration
- [x] Implement `set_seed`
- [x] Implement device resolver
- [x] Add `EmbedFitError`
- [x] Add `DummyBackbone`
- [x] Add synthetic-image fixture helpers
- [x] Add import smoke test
- [x] Ensure wheel/sdist build

## M1 Backbone + model

- [x] Define `ImageBackbone`
- [x] Implement backbone registry
- [x] Implement `TorchvisionBackbone`
- [x] Implement provider resolution
- [x] Implement `EmbedFitModel`
- [x] Implement `encode`
- [x] Implement embedding normalization
- [x] Implement freeze/unfreeze
- [x] Tests for malformed inputs
- [x] Tests for grayscale/RGB handling

## M2 Frozen baseline

- [x] Define `ClassificationHead`
- [x] Implement logistic regression head
- [x] Implement label mapping
- [x] Implement `fit_head`
- [x] Implement `predict`
- [x] Implement `predict_proba`
- [x] Implement `TrainingArguments`
- [x] Implement frozen `Trainer`
- [x] End-to-end synthetic test

## M3 Metrics

- [x] Standard classification metrics
- [x] PR-AUC
- [x] Recall at fixed FPR
- [x] Precision at review rate
- [x] Recall at review rate
- [x] Precision@K / Recall@K
- [x] Dollar recall
- [x] Threshold selection
- [x] Positive-class semantics validation

## M4 Data safety

- [ ] N-per-class sampling
- [ ] Grouped split
- [ ] Temporal split
- [ ] Split overlap checker
- [ ] Documentation for leakage

## M5 SupCon

- [ ] Balanced class batch sampler
- [ ] SupCon loss
- [ ] Unit tests for loss
- [ ] Representation optimizer
- [ ] Training loop
- [ ] Validation loss
- [ ] Best checkpoint restore
- [ ] Freeze + head fit transition
- [ ] Integration test demonstrating encoder weights change

## M6 Pairs

- [ ] `ImagePair`
- [ ] Pair sampler
- [ ] Bounded generation
- [ ] Pair loss
- [ ] Pair trainer
- [ ] Relationship rule abstraction
- [ ] Tests

## M7 Providers

- [ ] timm adapter
- [ ] HF Transformers adapter
- [ ] pooling configuration
- [ ] opt-in external checkpoint tests

## M8 Serialization

- [ ] config schema
- [ ] backbone save/load
- [ ] head save/load
- [ ] labels
- [ ] metadata
- [ ] round-trip equivalence

## M9 Retrieval

- [ ] brute-force cosine backend
- [ ] retrieval result type
- [ ] top-k
- [ ] exclude self
- [ ] retrieval metrics
- [ ] leakage-safe exclusion by group

## M10 Few-shot benchmark

- [ ] evaluator
- [ ] multiple shots
- [ ] repeated seeds
- [ ] tidy records
- [ ] aggregation
- [ ] benchmark frozen + CE + SupCon

## M11 Hard negatives

- [ ] miner
- [ ] refresh schedule
- [ ] train-only guard
- [ ] hard-vs-random test

## M12 Docs/release

- [ ] README quickstart
- [ ] examples
- [ ] domain-risk guide
- [ ] privacy/logging review
- [ ] changelog
- [ ] v0.1 release checklist
