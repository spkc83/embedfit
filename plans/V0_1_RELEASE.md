# v0.1 Release Checklist

Status legend: `[x]` verified on `main`, `[ ]` open.

## Functional

- [x] frozen + LR — `Trainer(training_mode="frozen")`; `tests/integration/test_frozen_baseline.py`
- [x] SupCon + LR — `training_mode="supcon"`; `tests/integration/test_supcon_training.py`
- [x] pair contrastive path — `training_mode="pair"` with `RelationshipRule`; `tests/integration/test_pair_training.py`
- [x] hard-negative mining for pair training — on the synthetic task it adds variance and does not improve accuracy
- [x] torchvision provider — offline tests plus an opt-in pretrained ResNet-18 test
- [x] timm provider — preprocessing compared with `timm.data.create_transform` across `crop_pct` values and crop modes, plus an opt-in `resnet10t.c3_in1k` test
- [x] Hugging Face provider — preprocessing compared with Transformers image processors, unsupported settings raise, plus an opt-in `facebook/dinov2-small` test
- [x] classification metrics
- [x] risk metrics, calibration, and threshold selection
- [x] retrieval with leakage-safe exclusions and ranking metrics
- [x] save/load — pickle-free bundles, atomic saves, consistency checks on load
- [x] few-shot evaluator, `end_to_end` baseline, `LinearHead` — every mode ends with a converged head (ADR 0011); `end_to_end` is LP-FT (ADR 0012)
- [x] batch-norm statistics fixed during representation training — `TrainingArguments.freeze_batch_norm`, default `True` (ADR 0013); `tests/integration/test_batch_norm_training.py`

## Quality

- [x] offline CPU tests green — no GPU, and internet sockets are blocked for every test not marked `network`
- [x] coverage target met — 98% overall; every module in `TEST_STRATEGY.md`'s list is above 85%
- [x] lint green — `ruff check`, `ruff format --check`, and `mypy` with `disallow_untyped_defs`
- [x] package build green — wheel and sdist, Apache-2.0 metadata, `py.typed`
- [x] examples execute — six scripts run by `tests/integration/test_examples.py`
- [x] no sensitive-data logging paths — structural audit of log calls and exception messages, no chained file-path causes; mechanisms listed in `SECURITY_PRIVACY.md`
- [x] README quickstart verified — every Python block in `README.md` runs in order, offline on the `dummy/tiny` backbone, in `tests/integration/test_readme.py`
- [x] public API documented with types and docstrings; behaviour pinned by regression tests

## Research readiness

- [x] frozen baseline benchmark
- [x] CE baseline benchmark — `end_to_end`, scored with a converged head like every other mode (ADR 0011); it starts from a linear probe (ADR 0012); on the synthetic task (`examples/06_few_shot_benchmark.py` with seeds 1 to 5 instead of its default three; 15 epochs, batch size 12, and one learning rate of 1e-2 for every method, at which the first epoch disturbs the probe, ADR 0012) it trails SupCon at 4 and 8 shots (0.86 and 0.97 against 0.96 and 1.00) and matches it at 16
- [x] SupCon benchmark
- [x] repeated-seed support — `FewShotEvaluator(shots, seeds)` with `aggregate`
- [x] grouped split support
- [x] temporal split support

Real-image results: EuroSAT with DINOv2-small at 4, 8, and 16 shots ([results](../docs/benchmarks/eurosat-dinov2-small.md)). There, `end_to_end` beat every other mode on all nine few-shot samples and neither contrastive mode separated from `frozen`. With ResNet-50 on the same splits ([results](../docs/benchmarks/eurosat-resnet50.md)), `end_to_end` again had the highest mean at every shot count, and both contrastive modes improved on `frozen`. On RVL-CDIP document pages with DINOv2-small ([results](../docs/benchmarks/rvl-cdip-dinov2-small.md); `frozen`, `end_to_end`, and `supcon` only), `end_to_end` was best on all nine few-shot samples and `supcon` improved on `frozen` at 16 shots only. ADR 0012 (the earlier, randomly initialised `end_to_end` classifier on EuroSAT with DINOv2-small) and ADR 0013 (ResNet-50 on EuroSAT with and without fixed batch-norm statistics) also report single-seed real-image figures. Apart from these, the accuracy figures in the docs and tests come from synthetic images on an untrained demo backbone and say nothing about real data. Not yet benchmarked: other backbones and datasets, 32 to 128 shots, `pair` on documents, hard negatives, retrieval, and out-of-time evaluation (`docs/research/EXPERIMENT_MATRIX.md`).

## Release hygiene

- [ ] version set to 0.1.0 — currently `0.1.0.dev0` in `src/embedfit/_version.py`
- [x] changelog — `CHANGELOG.md`
- [x] license decision — Apache-2.0

## Specified for v0.1 but not built

These are specified for v0.1 in `SPEC.md` or the design and research docs, but are not built. Each needs a decision: build it, or defer it in `SPEC.md`.

- [x] Augmentation profiles `none` / `natural` / `document` (§13): `TrainingArguments.augmentation`, `embedfit.augmentation` (ADR 0014). Evidence: `tests/unit/test_augmentation.py` (profile contract, `document` never mirrors over 200 seeds, crop and rotation bounds), `tests/integration/test_augmented_training.py` (only training draws augmented in every trained mode, identity pipeline equals `none`, reproducible and independent of `num_workers`).
- [x] Dataset adapters for Hugging Face `datasets` and pandas DataFrames (§6): `ImageDataset.from_hf_dataset` and `ImageDataset.from_pandas`. Evidence: `tests/unit/test_hf_dataset_adapter.py`, `tests/unit/test_pandas_adapter.py`.
- [x] Corrupt-image policy `skip` with a reported count (`docs/design/DATA_MODEL.md`): `on_corrupt_image="skip"` and `ImageDataset.num_skipped`. Evidence: `tests/unit/test_dataset.py::TestSkipCorruptImages`, `tests/unit/test_image.py`, `tests/unit/test_privacy_audit.py`.
- [x] Partial unfreezing for the end-to-end baseline (`docs/research/BASELINES.md`) — `TrainingArguments.unfreeze_last_blocks`, for every mode that trains the backbone, with provider-defined blocks (ADR 0015); `tests/unit/test_backbone_blocks.py`, `tests/integration/test_partial_unfreezing.py`, and block tests per provider.
- [x] Representation checkpoints on disk — `TrainingArguments.checkpoint_dir` writes the best and the last epoch as model bundles (ADR 0016); `tests/integration/test_representation_checkpoints.py`.
- [x] Sampler registry (§26): `register_sampler` and `TrainingArguments.sampler` for `supcon` and `end_to_end` batches, checked for every epoch before training. Evidence: `tests/unit/test_sampler_registry.py`, `tests/integration/test_registries.py`.
- [x] Metric registry (§26): `register_metric`, `Trainer.evaluate(metrics=...)`, and `FewShotEvaluator.run(metrics=...)`. Evidence: `tests/unit/evaluation/test_metric_registry.py`, `tests/integration/test_registries.py`.
- [x] Augmentation pipeline registry (§26): `register_augmentation` and `TrainingArguments.augmentation`, with `"none"` reserved. Evidence: `tests/unit/test_augmentation_registry.py`, `tests/integration/test_augmented_training.py` (a registered pipeline used in every trained mode and under a registered custom sampler).

## Deferred beyond v0.1

Recorded as deferred in `SPEC.md` or ADR 0001; not part of the v0.1 release.

- `triplet` training mode — required by v0.2 (§3). `TrainingArguments` accepts the mode and `Trainer` rejects it as unavailable.
- Multilabel classification — optional for v0.2 (§2). v0.1 is single-label.
- Prototype / nearest-centroid head — optional for v0.2 (§9). A custom head can be registered with `register_head` (`docs/research/BASELINES.md`).
- PyPI package-name check — not needed for v0.1; the first publication is planned on Hugging Face, with a blog post. `embedfit` has not been checked against PyPI; that check comes before any PyPI upload (ADR 0001).
