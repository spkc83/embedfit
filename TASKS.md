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

- [x] N-per-class sampling
- [x] Grouped split
- [x] Temporal split
- [x] Split overlap checker
- [x] Documentation for leakage

## M5 SupCon

- [x] Balanced class batch sampler
- [x] SupCon loss
- [x] Unit tests for loss
- [x] Representation optimizer
- [x] Training loop
- [x] Validation loss
- [x] Best checkpoint restore
- [x] Freeze + head fit transition
- [x] Integration test demonstrating encoder weights change

## M6 Pairs

- [x] `ImagePair`
- [x] Pair sampler
- [x] Bounded generation
- [x] Pair loss
- [x] Pair trainer
- [x] Relationship rule abstraction
- [x] Tests

## M7 Providers

- [ ] timm adapter
- [ ] HF Transformers adapter
- [ ] pooling configuration
- [ ] opt-in external checkpoint tests

## M8 Serialization

- [x] config schema
- [x] backbone save/load
- [x] head save/load
- [x] labels
- [x] metadata
- [x] round-trip equivalence

## M9 Retrieval

- [x] brute-force cosine backend
- [x] retrieval result type
- [x] top-k
- [x] exclude self
- [x] retrieval metrics
- [x] leakage-safe exclusion by group

## M10 Few-shot benchmark

- [x] evaluator
- [x] multiple shots
- [x] repeated seeds
- [x] tidy records
- [x] aggregation
- [x] benchmark frozen + CE + SupCon

## M11 Hard negatives

- [x] miner
- [x] refresh schedule
- [x] train-only guard
- [x] hard-vs-random test

## M12 Docs/release

- [x] README quickstart
- [x] examples
- [x] domain-risk guide
- [ ] privacy/logging review
- [x] changelog
- [ ] v0.1 release checklist
