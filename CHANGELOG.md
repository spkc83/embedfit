# Changelog

All notable changes to EmbedFit are recorded here, following [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `EmbedFitModel`: `from_pretrained` for `<provider>/<name>` identifiers and saved directories, `encode`, `fit_head`, `predict`, `predict_proba`, `risk_score`, `freeze_backbone` / `unfreeze_backbone`, `save_pretrained`.
- Backbone providers `torchvision`, `timm`, `hf` (Hugging Face Transformers), and `dummy`, the `ImageBackbone` contract, and a provider registry. Adapters own resizing, interpolation, and normalisation, and transformer encoders offer `provider_default`, `cls`, and `mean` pooling.
- `Trainer` with the training modes `frozen`, `supcon`, `pair`, and the `end_to_end` cross-entropy baseline (linear probe, then fine-tuning; ADR 0012), each ending with a head fitted to convergence; `TrainingArguments` validated on creation, with batch-norm running statistics kept fixed during representation training by default (`freeze_batch_norm`, ADR 0013); best-epoch selection on validation data; `evaluate` and `save_model`.
- Partial unfreezing: `TrainingArguments.unfreeze_last_blocks` trains only the backbone's last N blocks in every mode that trains it, with blocks defined by each provider through `ImageBackbone.parameter_blocks()`; `unfreeze_backbone(last_blocks=N)` (ADR 0015).
- Representation checkpoints: `TrainingArguments.checkpoint_dir` writes the best and the last epoch as model bundles with an unfitted head, atomically, with an index in `checkpoints.json` (ADR 0016).
- Task augmentation for representation training: `TrainingArguments.augmentation` with the profiles `none`, `natural`, and `document`, and an augmentation-pipeline registry (`embedfit.augmentation`). It runs before the backbone's preprocessing, on training draws only, seeded per draw (ADR 0014); `benchmarks/few_shot.py --augmentation`.
- Hard-negative mining for `pair` training (`hard_negative_fraction`, `hard_negative_refresh_epochs`), restricted to training data.
- Losses `SupConLoss` and `CosinePairLoss`; samplers `BalancedClassBatchSampler`, `PairSampler` with `RelationshipRule`, and `sample_few_shot`.
- Heads `LogisticRegressionHead` and `LinearHead` (with `from_module` and `to_module`), with a head registry.
- Batch-sampler registry (`register_sampler`, built-ins `balanced` and `random`) selected with `TrainingArguments.sampler` for `supcon` and `end_to_end` training; every epoch's batches are checked when the `Trainer` is created.
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
