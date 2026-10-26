# Changelog

All notable changes to EmbedFit are recorded here, following [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `TrainingArguments.max_steps`: a representation-training budget in optimizer steps for every trained mode. It replaces `num_epochs`, which is then ignored; training stops after exactly that many steps, part-way through an epoch if need be, and the learning-rate schedule is sized on it. A final partial epoch is validated and checkpointed like any other (ADR 0017).
- `TrainingArguments.projection_dim`: `supcon` computes its loss on a projection head `Linear(D, D) → ReLU → Linear(D, projection_dim)` that is trained with the backbone and discarded afterwards; the head and retrieval keep using the backbone embedding (ADR 0018).
- `TrainingArguments.contrastive_views`: `supcon` augments every drawn image this many times, each view with its own seed, and passes the views to `SupConLoss` as `[B, V, D]`; needs an augmentation (ADR 0018).
- `benchmarks/few_shot.py --max-steps`, `--projection-dim`, and `--contrastive-views`.
- `TrainingArguments.head_learning_rate` and `TrainingArguments.layer_lr_decay`: per-part learning rates for representation training. The first sets the rate of the `end_to_end` classifier and the `supcon` projection head; the second trains backbone block `i` of `n` at `learning_rate × decay ** (n − 1 − i)`, the stem at `decay ** n` times it, and the tail at `learning_rate`, with the blocks of partial unfreezing. Both default to one rate for every parameter, as before, and are recorded with saved models and checkpoints (ADR 0020).
- `benchmarks/few_shot.py --head-learning-rate` and `--layer-lr-decay` (and `benchmarks/embedding_quality.py`), per method as `METHOD=VALUE`, part of the resume configuration.
- `benchmarks/embedding_quality.py`: an embedding-quality benchmark with the data and training options of `benchmarks/few_shot.py`. The `open_set` task holds two or more classes out of training and measures retrieval, nearest-centroid accuracy, and novelty detection (against the training images and against a held-out reference) on them; the `rare_positive` task trains on a few positives among many negatives and ranks the evaluated split by the head's probability and by similarity to the known positives (`docs/guides/benchmarking.md`).
- `benchmarks/few_shot.py --interpolate` and `benchmarks/embedding_quality.py --interpolate`: WiSE-FT weight interpolation. After training, every trained method is also scored with its backbone at `(1 − α) × pretrained + α × trained` for each given α in [0, 1), with a new head fitted; at α 0 the `supcon` and `pair` scores equal the `frozen` ones, and `end_to_end` is exactly its linear probe. Records gain an `alpha` key (`null` for the trained model).
- `benchmarks/few_shot.py --temperature` (`supcon`), `--device` (default `cpu`; part of the resume configuration), and `--disable-cudnn`, which runs CUDA convolutions without cuDNN for GPUs the installed cuDNN does not support.
- `benchmarks/few_shot.py --dataset` (and `benchmarks/embedding_quality.py`) takes `dtd` (Describable Textures), `pets` (Oxford-IIIT Pets), `flowers` (Oxford Flowers-102), and `pcam` (PatchCamelyon), downloaded at run time; their pool, validation, and test splits come from the official ones (`docs/guides/benchmarking.md#datasets`). For these, `--pool-per-class` and `--eval-per-class` are caps per class.
- `benchmarks/few_shot.py --max-image-side` (and `benchmarks/embedding_quality.py`): decode every loaded image once and keep it in memory with its shorter side shrunk to at most N, instead of decoding and augmenting the full-resolution source at every training draw. Off by default; recorded in the results file's configuration. Pool duplicates and repeated evaluation images are found on the original pixels, and an N below the backbone's resize edge is refused before any data is loaded.
- `benchmarks/few_shot.py --heads`: score every method with each named head (`logistic_regression`, `linear`), fitted on the training embeddings of the final backbone and of each interpolated one, after one training run per method, shot count, and seed. Records gain a `head` key; without the option every method keeps its own head, as before (ADR 0019).
- The environment record of `benchmarks/few_shot.py` and `benchmarks/embedding_quality.py` results files holds the git commit and whether the checkout had uncommitted changes (`git_dirty`; both `null` outside a git checkout), and the versions of torchvision, Transformers, timm, and scikit-learn.

### Changed

- `benchmarks/few_shot.py` and `benchmarks/embedding_quality.py` refuse a shot count or seed given twice, a shot count below 1, and a negative seed. They seed the random number generators with the group's seed before they build each model. A backbone with randomly initialised parts (for example a newly added layer) is therefore initialised differently from before: do not resume a results file of such a backbone written before this change, since its old and new records would not be comparable. Fully pretrained backbones are not affected.
- `benchmarks/embedding_quality.py` checks before any training that the pool covers the largest shot count for every seed, not only the first.
- `benchmarks/few_shot.py` exits with a clear message when the installed torchvision does not list a dataset's image files and labels as expected, or lists different numbers of them.
- ADR 0011 no longer says that every mode is scored with the same head: `end_to_end` gets a `LinearHead`, the others logistic regression (ADR 0019).
- `benchmarks/embedding_quality.py`: the `open_set` error for too few classes states how many classes the task needs.

### Fixed

- `benchmarks/few_shot.py` and `benchmarks/embedding_quality.py` switch NNPACK off where it cannot run, so that convolutions on the CPU, or on CUDA with `--disable-cudnn`, no longer log "Could not initialize NNPACK" on every call. Results do not change: NNPACK is never used there.

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
