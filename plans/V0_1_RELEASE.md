# v0.1 Release Checklist

Status legend: `[x]` verified, `[~]` done but awaiting independent review and merge to `main`, `[ ]` open.

## Functional

- [x] frozen + LR — `Trainer(training_mode="frozen")`; `tests/integration/test_frozen_baseline.py`
- [x] SupCon + LR — `training_mode="supcon"`; `tests/integration/test_supcon_training.py`
- [x] pair contrastive path — `training_mode="pair"` with `RelationshipRule`; `tests/integration/test_pair_training.py`
- [~] hard-negative mining for pair training — `feat/m11-hard-negatives`
- [x] torchvision provider — offline tests plus an opt-in pretrained ResNet-18 test
- [~] timm provider — `feat/m7-providers`; offline tests plus an opt-in `resnet10t.c3_in1k` test
- [~] Hugging Face provider — `feat/m7-providers`; offline tests plus an opt-in `facebook/dinov2-small` test
- [x] classification metrics
- [x] risk metrics, calibration, and threshold selection
- [x] retrieval with leakage-safe exclusions and ranking metrics
- [x] save/load — pickle-free bundles, atomic saves, consistency checks on load
- [~] few-shot evaluator, `end_to_end` baseline, `LinearHead` — `feat/m10-few-shot-evaluator`

## Quality

- [x] offline CPU tests green — 899 tests on the branch stack, 702 on `main`; no network, no GPU
- [x] coverage target met — 98% overall; every module in `TEST_STRATEGY.md`'s list is above 85%
- [x] lint green — `ruff check`, `ruff format --check`, and `mypy` with `disallow_untyped_defs`
- [x] package build green — wheel and sdist, Apache-2.0 metadata, `py.typed`
- [~] examples execute — six scripts run by `tests/integration/test_examples.py` (`docs/m12-examples`)
- [~] no sensitive-data logging paths — audit and structural guard test (`chore/m12-privacy-audit`); mechanisms listed in `SECURITY_PRIVACY.md`
- [ ] README quickstart verified — the snippets use placeholder variables (`train_images`, `test_images`); the runnable equivalents are the examples
- [ ] public API reviewed — every milestone on `main` had an independent review; M7, M10, M11, and the examples have not had theirs yet

## Research readiness

- [x] frozen baseline benchmark
- [~] CE baseline benchmark — `end_to_end`; a temperature-scaled classifier was measured and rejected because it lowered few-shot accuracy
- [x] SupCon benchmark
- [~] repeated-seed support — `FewShotEvaluator(shots, seeds)` with `aggregate`
- [x] grouped split support
- [x] temporal split support

No benchmark on real images has been run. Every accuracy figure in the docs and tests comes from synthetic bar images on an untrained demo backbone and says nothing about real data; the experiments in `docs/research/EXPERIMENT_MATRIX.md` are still to be done.

## Release hygiene

- [ ] version set to 0.1.0 — currently `0.1.0.dev0` in `src/embedfit/_version.py`
- [~] changelog — `CHANGELOG.md` (`docs/m12-examples`)
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

## Merge order for the pending branches

`feat/m10-few-shot-evaluator` → `docs/m12-examples` → `feat/m11-hard-negatives` → `chore/m12-privacy-audit` → `feat/m7-providers` → `chore/v0.1-release-checklist`. Each is stacked on the previous one, so merging the last one brings in all of them.
