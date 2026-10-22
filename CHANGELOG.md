# Changelog

All notable changes to EmbedFit are recorded here, following [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `TrainingArguments.max_steps`: a representation-training budget in optimizer steps for every trained mode. It replaces `num_epochs`, which is then ignored; training stops after exactly that many steps, part-way through an epoch if need be, and the learning-rate schedule is sized on it. A final partial epoch is validated and checkpointed like any other (ADR 0017).
- `TrainingArguments.projection_dim`: `supcon` computes its loss on a projection head `Linear(D, D) → ReLU → Linear(D, projection_dim)` that is trained with the backbone and discarded afterwards; the head and retrieval keep using the backbone embedding (ADR 0018).
- `TrainingArguments.contrastive_views`: `supcon` augments every drawn image this many times, each view with its own seed, and passes the views to `SupConLoss` as `[B, V, D]`; needs an augmentation (ADR 0018).
- `benchmarks/few_shot.py --max-steps`, `--projection-dim`, and `--contrastive-views`.

## [0.1.0] - 2026-10-18

The first release.

### Added

- `EmbedFitModel`: `from_pretrained` for `<provider>/<name>` identifiers and saved directories, `encode`, `fit_head`, `predict`, `predict_proba`, `risk_score`, `freeze_backbone` / `unfreeze_backbone`, `save_pretrained`.
- Backbone providers `torchvision`, `timm`, `hf` (Hugging Face Transformers), and `dummy`, the `ImageBackbone` contract, and a provider registry. Adapters own resizing, interpolation, and normalisation, and transformer encoders offer `provider_default`, `cls`, and `mean` pooling.
- `Trainer` with the training modes `frozen`, `supcon`, `pair`, and the `end_to_end` cross-entropy baseline (linear probe, then fine-tuning; ADR 0012), each ending with a head fitted to convergence; `TrainingArguments` validated on creation, with batch-norm running statistics kept fixed during representation training by default (`freeze_batch_norm`, ADR 0013); best-epoch selection on validation data; `evaluate` and `save_model`.
- Partial unfreezing: `TrainingArguments.unfreeze_last_blocks` trains only the backbone's last N blocks in every mode that trains it, with blocks defined by each provider through `ImageBackbone.parameter_blocks()`; `unfreeze_backbone(last_blocks=N)` (ADR 0015).
- Representation checkpoints: `TrainingArguments.checkpoint_dir` writes the best and the last epoch as model bundles with an unfitted head, atomically, with an index in `checkpoints.json` (ADR 0016).
- Task augmentation for representation training: `TrainingArguments.augmentation` with the profiles `none`, `natural`, and `document`, and an augmentation-pipeline registry (`embedfit.augmentation`). It runs before the backbone's preprocessing, on training draws only, seeded per draw (ADR 0014); `benchmarks/few_shot.py --augmentation`.
- Hard-negative mining for `pair` training (`hard_negative_fraction`, `hard_negative_refresh_epochs`), restricted to training data.
- Losses `SupConLoss` and `CosinePairLoss`, with a loss registry (`register_loss`, `create_loss`); no `TrainingArguments` field selects a registered loss, and training builds `supcon` and `cosine_pair` by name. Samplers `BalancedClassBatchSampler`, `PairSampler` with `RelationshipRule`, and `sample_few_shot`.
- Heads `LogisticRegressionHead` and `LinearHead` (with `from_module` and `to_module`), with a head registry.
- Batch-sampler registry (`register_sampler`, built-ins `balanced` and `random`) selected with `TrainingArguments.sampler` for `supcon` and `end_to_end` training; every sampler except the built-in `balanced`, which is valid by construction, has its batches checked for every epoch when the `Trainer` is created (the built-in `random` included).
- Metric registry (`register_metric`) for positive-class score metrics, added to results with `Trainer.evaluate(metrics=...)` and `FewShotEvaluator.run(metrics=...)`; built-in metric names are reserved.
- `embedfit.evaluation`: classification metrics, ROC-AUC, PR-AUC, recall at a fixed FPR, precision and recall at K and at a review rate, dollar recall, Brier score, calibration bins, ECE, and `select_threshold`.
- `embedfit.data`: `ImageDataset`, `ImageSample`, `LabelMapping`, image conversion, stratified, grouped, and temporal splits, and `find_split_overlap`.
- `RetrievalIndex` with a brute-force cosine backend, exclusion by id, by the query's own id, and by group, and ranking metrics.
- `FewShotEvaluator` for methods × shots × seeds benchmarks with tidy records.
- Pickle-free model bundles with atomic saving and consistency checks on load.
- `benchmarks/few_shot.py`, a few-shot benchmark on public datasets (EuroSAT, RVL-CDIP documents), with results in `docs/benchmarks/`.
- Runnable examples on synthetic data, and guides on data leakage, benchmarking, and rare-event risk workflows.
- Dataset adapters `ImageDataset.from_pandas` and `ImageDataset.from_hf_dataset` (the `pandas` and `datasets` extras), with a column mapping and `metadata_columns`.
- Encoded image file bytes as an accepted image form (ADR 0007 amendment).
- Corrupt-image policy `on_corrupt_image="skip"`, which drops records whose image file or bytes cannot be read when the dataset is built and reports the count in `ImageDataset.num_skipped`.

### Not yet available

- The `triplet` training mode (deferred to v0.2).
- Multilabel classification and a prototype / nearest-centroid head (deferred beyond v0.1).
