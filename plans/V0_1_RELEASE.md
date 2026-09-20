# v0.1 Release Checklist

Status legend: `[x]` verified on `main`, `[ ]` open. Every code milestone was reviewed independently (brute-force cross-checks and mutation tests) before it was merged.

## Functional

- [x] frozen + LR — `Trainer(training_mode="frozen")`; `tests/integration/test_frozen_baseline.py`
- [x] SupCon + LR — `training_mode="supcon"`; `tests/integration/test_supcon_training.py`
- [x] pair contrastive path — `training_mode="pair"` with `RelationshipRule`; `tests/integration/test_pair_training.py`
- [x] hard-negative mining for pair training — reviewed; on the synthetic task it adds variance and does not improve accuracy
- [x] torchvision provider — offline tests plus an opt-in pretrained ResNet-18 test
- [x] timm provider — reviewed; preprocessing compared with `timm.data.create_transform` across `crop_pct` values and crop modes, plus an opt-in `resnet10t.c3_in1k` test
- [x] Hugging Face provider — reviewed; preprocessing compared with Transformers image processors, unsupported settings raise, plus an opt-in `facebook/dinov2-small` test
- [x] classification metrics
- [x] risk metrics, calibration, and threshold selection
- [x] retrieval with leakage-safe exclusions and ranking metrics
- [x] save/load — pickle-free bundles, atomic saves, consistency checks on load
- [x] few-shot evaluator, `end_to_end` baseline, `LinearHead` — reviewed; every mode ends with a converged head (ADR 0011)

## Quality

- [x] offline CPU tests green — no GPU, and internet sockets are blocked for every test not marked `network`
- [x] coverage target met — 98% overall; every module in `TEST_STRATEGY.md`'s list is above 85%
- [x] lint green — `ruff check`, `ruff format --check`, and `mypy` with `disallow_untyped_defs`
- [x] package build green — wheel and sdist, Apache-2.0 metadata, `py.typed`
- [x] examples execute — six scripts run by `tests/integration/test_examples.py` (`docs/m12-examples`)
- [x] no sensitive-data logging paths — structural audit of log calls and exception messages, no chained file-path causes; mechanisms listed in `SECURITY_PRIVACY.md`
- [ ] README quickstart verified — the snippets use placeholder variables (`train_images`, `test_images`); the runnable equivalents are the examples
- [x] public API reviewed — every milestone, and the examples, guides, and privacy audit, had an independent review, and the findings are fixed with regression tests

## Research readiness

- [x] frozen baseline benchmark
- [x] CE baseline benchmark — `end_to_end`, scored with a converged head like every other mode (ADR 0011); on the synthetic task it matches SupCon from 8 shots up, and at 4 shots the ordering varies by seed
- [x] SupCon benchmark
- [x] repeated-seed support — `FewShotEvaluator(shots, seeds)` with `aggregate`
- [x] grouped split support
- [x] temporal split support

No benchmark on real images has been run. Every accuracy figure in the docs and tests comes from synthetic bar images on an untrained demo backbone and says nothing about real data; the experiments in `docs/research/EXPERIMENT_MATRIX.md` are still to be done.

## Release hygiene

- [ ] version set to 0.1.0 — currently `0.1.0.dev0` in `src/embedfit/_version.py`
- [x] changelog — `CHANGELOG.md` (`docs/m12-examples`)
- [x] license decision — Apache-2.0
- [ ] final package-name availability check — `embedfit` has not been checked against PyPI (ADR 0001 keeps the name provisional)

## Specified for v0.1 but not built

These appear in `SPEC.md` without a milestone in `IMPLEMENTATION_PLAN.md`. Each needs a decision: build it, or defer it in `SPEC.md`.

- [ ] Augmentation profiles `none` / `natural` / `document` (§13). Training currently applies no task augmentation.
- [ ] Dataset adapters for Hugging Face `datasets` and pandas DataFrames (§6). `ImageDataset` accepts sequences of records and map-style torch datasets.
- [ ] Corrupt-image policy `skip` with a reported count (`docs/design/DATA_MODEL.md`). A corrupt image always raises.
- [ ] `triplet` training mode (§3: optional for v0.1, required by v0.2). `TrainingArguments` accepts the mode and `Trainer` rejects it as unavailable.
- [ ] Multilabel classification (§2, stretch).
- [ ] Partial unfreezing for the end-to-end baseline (`docs/research/BASELINES.md`). Training always unfreezes the whole backbone.
- [ ] Representation checkpoints on disk. The best epoch is kept in memory and restored; nothing is written until `save_model`.
- [ ] Registries for samplers, metrics, and augmentation pipelines (§26). Backbones, heads, and losses have registries; custom samplers and metrics are used by calling them directly.
- [ ] Prototype / nearest-centroid head (§9, optional).
