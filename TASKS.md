# Task Board

The coding agent should update this file as implementation progresses.

Legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` complete
- `[!]` blocked

## M0 Foundation

- [x] Create `pyproject.toml`
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
- [x] Hugging Face `datasets` and pandas adapters (`ImageDataset.from_hf_dataset`, `ImageDataset.from_pandas`)
- [x] Corrupt-image `skip` policy with a reported count (`on_corrupt_image`, `num_skipped`)

## M5 SupCon

- [x] Balanced class batch sampler
- [x] SupCon loss
- [x] Unit tests for loss
- [x] Representation optimizer
- [x] Training loop
- [x] Validation loss
- [x] Best checkpoint restore
- [x] Representation checkpoints on disk, best and last epoch (`checkpoint_dir`, ADR 0016)
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

- [x] timm adapter
- [x] HF Transformers adapter
- [x] pooling configuration
- [x] opt-in external checkpoint tests

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

- [x] README quickstart, run offline by `tests/integration/test_readme.py`
- [x] examples
- [x] domain-risk guide
- [x] privacy/logging review
- [x] changelog
- [x] v0.1 release checklist (`plans/V0_1_RELEASE.md`), version 0.1.0

## After M12: v0.1 scope, baselines, and real-image benchmarks

- [x] `end_to_end` as linear probe then fine-tune (LP-FT, ADR 0012)
- [x] batch-norm statistics fixed during representation training, `freeze_batch_norm` (ADR 0013)
- [x] partial unfreezing of the last backbone blocks, `unfreeze_last_blocks` (ADR 0015)
- [x] augmentation profiles `none` / `natural` / `document` and the augmentation-pipeline registry (ADR 0014)
- [x] batch-sampler and metric registries (`register_sampler`, `register_metric`; SPEC §26)
- [x] EuroSAT few-shot benchmark with DINOv2-small (`benchmarks/few_shot.py`, `docs/benchmarks/eurosat-dinov2-small.md`)
- [x] EuroSAT few-shot benchmark with ResNet-50 (`docs/benchmarks/eurosat-resnet50.md`)
- [x] RVL-CDIP document-image benchmark with DINOv2-small (`docs/benchmarks/rvl-cdip-dinov2-small.md`)
- [ ] further real-image benchmarks (`docs/research/EXPERIMENT_MATRIX.md`)
