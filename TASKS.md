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

- [ ] Define `ImageBackbone`
- [ ] Implement backbone registry
- [ ] Implement `TorchvisionBackbone`
- [ ] Implement provider resolution
- [ ] Implement `EmbedFitModel`
- [ ] Implement `encode`
- [ ] Implement embedding normalization
- [ ] Implement freeze/unfreeze
- [ ] Tests for malformed inputs
- [ ] Tests for grayscale/RGB handling

## M2 Frozen baseline

- [ ] Define `ClassificationHead`
- [ ] Implement logistic regression head
- [ ] Implement label mapping
- [ ] Implement `fit_head`
- [ ] Implement `predict`
- [ ] Implement `predict_proba`
- [ ] Implement `TrainingArguments`
- [ ] Implement frozen `Trainer`
- [ ] End-to-end synthetic test

## M3 Metrics

- [ ] Standard classification metrics
- [ ] PR-AUC
- [ ] Recall at fixed FPR
- [ ] Precision at review rate
- [ ] Recall at review rate
- [ ] Precision@K / Recall@K
- [ ] Dollar recall
- [ ] Threshold selection
- [ ] Positive-class semantics validation

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
