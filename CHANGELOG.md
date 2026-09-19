# Changelog

All notable changes to EmbedFit are recorded here, following [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `EmbedFitModel`: `from_pretrained` for `<provider>/<name>` identifiers and saved directories, `encode`, `fit_head`, `predict`, `predict_proba`, `risk_score`, `freeze_backbone` / `unfreeze_backbone`, `save_pretrained`.
- Backbone providers `torchvision` and `dummy`, the `ImageBackbone` contract, and a provider registry. Adapters own resizing, interpolation, and normalisation.
- `Trainer` with the training modes `frozen`, `supcon`, `pair`, and the `end_to_end` cross-entropy baseline; `TrainingArguments` validated on creation; best-epoch selection on validation data; `evaluate` and `save_model`.
- Losses `SupConLoss` and `CosinePairLoss`; samplers `BalancedClassBatchSampler`, `PairSampler` with `RelationshipRule`, and `sample_few_shot`.
- Heads `LogisticRegressionHead` and `LinearHead`, with a head registry.
- `embedfit.evaluation`: classification metrics, ROC-AUC, PR-AUC, recall at a fixed FPR, precision and recall at K and at a review rate, dollar recall, Brier score, calibration bins, ECE, and `select_threshold`.
- `embedfit.data`: `ImageDataset`, `ImageSample`, `LabelMapping`, image conversion, stratified, grouped, and temporal splits, and `find_split_overlap`.
- `RetrievalIndex` with a brute-force cosine backend, exclusion by id, by the query's own id, and by group, and ranking metrics.
- `FewShotEvaluator` for methods × shots × seeds benchmarks with tidy records.
- Pickle-free model bundles with atomic saving and consistency checks on load.
- Runnable examples on synthetic data, and guides on data leakage and benchmarking.

### Not yet available

- `timm` and Hugging Face backbone providers and configurable pooling.
- Hard-negative mining, the `triplet` training mode, and augmentation profiles.
