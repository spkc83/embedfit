# EmbedFit

> **Status:** pre-alpha — see [What the package provides](#what-the-package-provides)  
> **Primary modality for v0.1:** images  
> **Reference use case:** few-shot check-fraud / deposit-risk classification and similarity retrieval  
> **Package name:** `embedfit` ([naming notes](PROJECT_NAMING.md))

EmbedFit is a SetFit-inspired framework for adapting pretrained embedding models to supervised tasks with very small labeled datasets.

The v0.1 implementation focuses on vision:

```text
pretrained vision encoder
        ↓
few-shot representation adaptation
        ↓
task-specific image embeddings
        ↓
lightweight classifier / retrieval / anomaly signals
```

The initial financial-services use case is check deposit risk, including fraud and RDI/returned-item patterns, but **the core library must remain domain-neutral**.

## Install

EmbedFit is not on PyPI. Install it from source:

```bash
git clone https://github.com/spkc83/embedfit.git
cd embedfit
pip install -e .                  # core: torch, numpy, scikit-learn, pillow
pip install -e ".[torchvision]"   # or another extra for pretrained backbones: timm, hf, all
```

## What the package provides

The design is specified in [`SPEC.md`](SPEC.md) and [`ARCHITECTURE.md`](ARCHITECTURE.md), with decisions recorded in [`docs/adrs/`](docs/adrs/). Contributors, human or agent, start from [`AGENTS.md`](AGENTS.md); open work is listed in [`TASKS.md`](TASKS.md), and the v0.1 release checklist is [`plans/V0_1_RELEASE.md`](plans/V0_1_RELEASE.md).

The snippets below run in order, offline. The first one builds stand-in data from synthetic images; replace it with your own labelled images:

```python
from embedfit import EmbedFitModel, ImageDataset, Trainer, TrainingArguments
from embedfit.testing import make_bar_images


def stand_in_scans(per_class, seed):  # two classes of synthetic images
    images, kinds = make_bar_images(per_class, num_classes=2, seed=seed)
    return images, ["legitimate" if kind == "vertical" else "risk" for kind in kinds]


train_images, train_labels = stand_in_scans(16, seed=0)
val_images, val_labels = stand_in_scans(16, seed=1)
test_images, test_labels = stand_in_scans(16, seed=2)
train = ImageDataset.from_lists(train_images, train_labels)
validation = ImageDataset.from_lists(val_images, val_labels)
test = ImageDataset.from_lists(test_images, test_labels)
```

```python
labels = ["legitimate", "risk"]
model = EmbedFitModel.from_pretrained(
    "dummy/tiny",                 # untrained offline CNN for this demo; for real images use e.g.
                                  # "torchvision/resnet18" (needs the torchvision extra)
    labels=labels,                # class order is as written, never sorted
    positive_label="risk",
)
embeddings = model.encode(test_images[:2])     # numpy array, shape [2, model.embedding_dim]

trainer = Trainer(model, TrainingArguments(training_mode="frozen"), train)
trainer.train()

model.predict(test_images)         # labels
model.predict_proba(test_images)   # [N, 2]; columns follow model.labels
model.risk_score(test_images)      # probability of positive_label
```

`FewShotEvaluator` compares methods at several training-set sizes over repeated seeds, with every method trained on the same few-shot sample and scored on the same held-out test set. On the stand-in data and the untrained `dummy/tiny` backbone the scores mean nothing; the block only shows the API:

```python
from embedfit import FewShotEvaluator

methods = ["frozen", "end_to_end", "supcon"]
evaluator = FewShotEvaluator(shots=[4, 8], seeds=[13, 42])   # e.g. shots=[4, 8, 16, 32]
records = evaluator.run(
    train,                                 # the pool few-shot samples are drawn from
    test_dataset=test,
    methods=methods,
    model_factory=lambda: EmbedFitModel.from_pretrained("dummy/tiny", labels=labels),
    training_arguments={m: TrainingArguments(training_mode=m, num_workers=0) for m in methods},
)                                      # tidy records: method, backbone, shots, seed, metric, value
summary = evaluator.aggregate(records)   # mean and standard deviation over seeds
```

A trained model is saved to a directory and loaded back without its training data:

```python
trainer.save_model("models/risk-v1")                       # or model.save_pretrained(path)
model = EmbedFitModel.from_pretrained("models/risk-v1")    # same labels, order, head, preprocessing
```

Metrics work on plain arrays, so they apply to any model's scores:

```python
from embedfit.evaluation import pr_auc, recall_at_fpr, recall_at_review_rate, select_threshold

scores = model.risk_score(val_images)
pr_auc(val_labels, scores, positive_label="risk")
recall_at_fpr(val_labels, scores, positive_label="risk", target_fpr=0.01)
recall_at_review_rate(val_labels, scores, positive_label="risk", review_rate=0.02)
select_threshold(val_labels, scores, positive_label="risk", objective="target_fpr", target=0.01)
```

`training_mode="frozen"` leaves the backbone untouched and fits a logistic-regression head on its embeddings. It is the baseline that the representation-tuning modes are measured against.

`training_mode="supcon"` first adapts the backbone with a supervised contrastive loss on class-balanced batches, then freezes it and fits the same head:

```python
args = TrainingArguments(
    training_mode="supcon", num_epochs=5, batch_size=32, learning_rate=2e-5,
    num_workers=0,                  # load in the main process; faster on small datasets
)
trainer = Trainer(model, args, train, eval_dataset=validation)   # validation picks the best epoch
output = trainer.train()            # output.train_losses, output.eval_losses, output.best_epoch
trainer.evaluate(test)
```

`training_mode="pair"` adapts the backbone on sampled positive and negative image pairs with a cosine pair loss instead. `relationship_rules` can declare pairs similar or dissimilar from any sample field, without the library knowing what the field means:

```python
from embedfit.data import ImageSample
from embedfit.sampling import RelationshipRule

train_groups = [f"group-{i // 4}" for i in range(len(train_images))]   # e.g. a shared account
train = ImageDataset([                     # from_lists cannot set group_id
    ImageSample(image, label, group_id=group)
    for image, label, group in zip(train_images, train_labels, train_groups)
])
rule = RelationshipRule(field="group_id", relation="same", pair_type="positive", weight=2.0)
args = TrainingArguments(training_mode="pair", num_workers=0)
Trainer(model, args, train, relationship_rules=[rule]).train()
```

`encode` accepts PIL images, numpy arrays, torch tensors, file paths, and encoded image file bytes, and returns one L2-normalised row per image. Resolution, interpolation mode, and normalisation constants come from the selected checkpoint, never from a global default; pass `image_size=` to run a convolutional backbone at higher resolution.

Public components:

| Component | Location | Purpose |
| --- | --- | --- |
| `EmbedFitModel` | `embedfit` | Backbone plus head: `from_pretrained` (identifier or saved directory), `save_pretrained`, `encode`, `fit_head`, `predict`, `predict_proba`, `risk_score`, `freeze_backbone` / `unfreeze_backbone` (whole backbone or its last blocks), differentiable `forward` |
| `Trainer`, `TrainingArguments` | `embedfit` | Two-phase training (representation, then head) with `train`, `predict`, `evaluate`, and `save_model`. Modes `frozen`, `supcon`, `pair`, and the `end_to_end` cross-entropy baseline; optional partial unfreezing (`unfreeze_last_blocks`), representation checkpoints on disk (`checkpoint_dir`), a budget in optimizer steps (`max_steps`), a SupCon projection head and augmented views (`projection_dim`, `contrastive_views`), and joint cross-entropy and SupCon training in `end_to_end` (`contrastive_weight`); arguments are validated on creation |
| `FewShotEvaluator` | `embedfit` | Benchmark of methods × shots × seeds as tidy records, with `aggregate` for mean and standard deviation |
| `ImageDataset`; `ImageSample` | `embedfit`; `embedfit.data` | Sample records (`image`, `label`, optional `sample_id`, `group_id`, `timestamp`, `amount`, `metadata`); file paths and encoded bytes stay undecoded until encoding; `from_pandas` and `from_hf_dataset` adapters; `on_corrupt_image="skip"` drops unreadable images |
| `classification_metrics`, `confusion_matrix`, `roc_auc`, `pr_auc` | `embedfit.evaluation` | Standard metrics; macro averages cover every declared class |
| `recall_at_fpr`, `precision_at_k`, `recall_at_k`, `precision_at_review_rate`, `recall_at_review_rate`, `dollar_recall`, `dollar_recall_at_review_rate` | `embedfit.evaluation` | Rare-event ranking metrics with an explicit positive label and order-independent tie handling |
| `brier_score`, `calibration_bins` (returns `CalibrationBins`), `expected_calibration_error` | `embedfit.evaluation` | Calibration of positive-class probabilities |
| `select_threshold` (returns `ThresholdResult`) | `embedfit.evaluation` | Threshold for max F1 or a target precision, recall, FPR, or review rate; for validation data only |
| `stratified_split`, `grouped_split`, `temporal_split`, `find_split_overlap` | `embedfit.data` | Reproducible splits by class, group, or time, and a check that no `sample_id` or `group_id` spans splits ([leakage guide](docs/guides/leakage.md)) |
| `sample_few_shot` | `embedfit.sampling` | N samples per class, without replacement |
| `register_sampler`, `create_sampler`, `available_samplers`, `BatchSampler` | `embedfit.sampling` | Registry of the batch samplers selectable with `TrainingArguments.sampler` for `supcon` and `end_to_end` (built-ins `balanced` and `random`); every sampler except the built-in `balanced`, which is valid by construction, is checked for every epoch when the `Trainer` is created |
| `register_metric`, `get_metric`, `available_metrics` | `embedfit.evaluation` | Registry of positive-class score metrics that `Trainer.evaluate(metrics=...)` and `FewShotEvaluator.run(metrics=...)` add to their results |
| `LabelMapping` | `embedfit.data` | Ordered label ↔ class-index mapping that distinguishes `0` from `False` |
| `SupConLoss`, `CosinePairLoss`, `create_loss`, `register_loss`, `available_losses` | `embedfit.losses` | Supervised contrastive loss over `[B, D]` or multi-view `[B, V, D]` embeddings, weighted cosine loss over embedding pairs, and a loss registry |
| `NaturalAugmentation`, `DocumentAugmentation`, `create_augmentation`, `register_augmentation`, `available_augmentations` | `embedfit.augmentation` | Task augmentation of training images (`TrainingArguments.augmentation`: `"none"`, `"natural"`, `"document"`, or a registered pipeline), applied before the backbone's own preprocessing and only to representation-training draws ([design](docs/design/API.md#augmentation)) |
| `PairSampler`, `ImagePair`, `RelationshipRule` | `embedfit.sampling` | Bounded, balanced positive/negative pairs per epoch; rules set pair targets and weights from arbitrary fields; optional hard-negative mining from training embeddings |
| `BalancedClassBatchSampler` | `embedfit.sampling` | Batches of up to `samples_per_class` images from each of `classes_per_batch` classes, never a duplicate image, reproducible per seed and epoch |
| `ClassificationHead`, `LogisticRegressionHead`, `LinearHead`, `create_head`, `load_head`, `register_head` | `embedfit.heads` | Classifiers over `[N, D]` embeddings and class indices, constructible by name |
| `RetrievalIndex` | `embedfit` | Cosine nearest-neighbour search over embeddings, with ids, metadata passthrough, and leakage-safe exclusion by id or group ([design](docs/design/RETRIEVAL.md#implemented-semantics)) |
| `RetrievalResult`, `RetrievalBackend`, `BruteForceCosineBackend` | `embedfit.retrieval` | One neighbour (`id`, `score`, `rank`, `metadata`); the backend contract future FAISS or pgvector adapters implement; the exhaustive v0.1 backend |
| `retrieval_precision_at_k`, `retrieval_recall_at_k`, `mean_reciprocal_rank`, `mean_average_precision` | `embedfit.retrieval` | Ranking quality of retrieved results against known-relevant ids, without calling a model |
| `ImageBackbone` | `embedfit.backbones` | Contract for provider adapters, which own preprocessing, feature extraction, and their own serialization |
| `TimmBackbone`, `HFBackbone` | `embedfit.backbones` | `timm/<name>` and `hf/<repo-id>` encoders (for example `timm/convnext_tiny.fb_in22k_ft_in1k`, `hf/facebook/dinov2-base`), each reproducing its checkpoint's preprocessing (unsupported settings raise rather than being ignored), with `pooling="provider_default"`, `"cls"`, or `"mean"` |
| `TorchvisionBackbone` | `embedfit.backbones` | torchvision classification models with the linear classifier removed (`torchvision/<name>`); models without one, such as SqueezeNet, are rejected |
| `create_backbone`, `load_backbone`, `register_backbone_provider` | `embedfit.backbones` | Resolve `<provider>/<name>` identifiers (the provider is never guessed) and reload saved backbones |
| `to_rgb_tensor` | `embedfit.data` | Converts any supported image form to a float RGB tensor in [0, 1] |
| `set_seed` | `embedfit.utils` | Seeds Python, NumPy, and torch (CPU + CUDA); optional deterministic mode |
| `resolve_device` | `embedfit.utils` | `"auto"` resolves CUDA → MPS → CPU; an explicit device never falls back silently |
| `EmbedFitError`, `ConfigurationError`, `ImageInputError`, `BackboneProviderError`, `HeadNotFittedError`, `UndefinedMetricError`, `TrainingDivergedError`, `DataLeakageError`, `IndexNotFittedError` | `embedfit.exceptions` | Error hierarchy; messages describe structure, never image or metadata content |
| `DummyBackbone` | `embedfit.backbones` | Tiny trainable CNN with no checkpoint, for offline tests and examples |
| `make_bar_images` | `embedfit.testing` | Deterministic synthetic image classes with learnable structure |

The `triplet` training mode is deferred to v0.2; `Trainer` rejects an unavailable `training_mode` when it is created. v0.1 handles single-label classification only; multilabel classification and a prototype / nearest-centroid head are deferred.

## Examples and guides

- [`examples/`](examples/README.md): six runnable scripts on synthetic images, from the frozen baseline to a rare-event risk workflow and a few-shot benchmark.
- [Avoiding data leakage](docs/guides/leakage.md): choosing splits and checking them.
- [Benchmarking few-shot methods](docs/guides/benchmarking.md): what makes a comparison fair, and how `benchmarks/embedding_quality.py` scores the embeddings on unseen classes and rare positives.
- [Rare-event risk workflows](docs/guides/rare-event-risk.md): splits, positive class, metrics, thresholds, and retrieval for rare, costly classes.
- [EuroSAT with DINOv2-small](docs/benchmarks/eurosat-dinov2-small.md): a few-shot benchmark on real images, with the script to reproduce it in [`benchmarks/`](benchmarks/few_shot.py).
- [EuroSAT with ResNet-50](docs/benchmarks/eurosat-resnet50.md): the same protocol with a batch-norm convolutional backbone.
- [RVL-CDIP with DINOv2-small](docs/benchmarks/rvl-cdip-dinov2-small.md): the same evaluation protocol on scanned document pages.
- [Contrastive adaptation across domains](docs/benchmarks/contrastive-adaptation.md): `supcon` and `ce_supcon` against `frozen` and `end_to_end` on Pets, Flowers-102, DTD, and RVL-CDIP under the lift rule, for accuracy, unseen classes, novelty, and rare positives.
- Benchmark datasets: `benchmarks/few_shot.py` also downloads DTD textures, Oxford-IIIT Pets, Oxford Flowers-102, and PatchCamelyon histopathology ([datasets](docs/guides/benchmarking.md#datasets)).
- [`CHANGELOG.md`](CHANGELOG.md).

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,torchvision]"

pytest -q            # offline, CPU-only; excludes tests marked `network` or `slow`
ruff check .
ruff format --check .
mypy
python -m build
```

Tests that download pretrained checkpoints are opt-in: `pytest -m network`. Each provider's adapter tests are skipped when its extra (`torchvision`, `timm`, `hf`) is missing, and the dataset adapter tests when `datasets` or `pandas` is missing; `pip install -e ".[dev,all]"` runs them all, including the preprocessing comparisons against `timm` and Transformers.

Optional extras install backbone providers: `torchvision`, `timm`, and `hf`. The `datasets` and `pandas` extras enable `ImageDataset.from_hf_dataset` and `ImageDataset.from_pandas`; without them, `ImageDataset` accepts a sequence, or a map-style torch `Dataset`, of `{"image", "label", ...}` mappings. `all` installs every extra. The core install depends only on `torch`, `numpy`, `scikit-learn`, and `pillow` ([ADR 0006](docs/adrs/0006-core-dependencies-and-testing-utilities.md)).

Branching, commit, and documentation conventions are in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Non-goals for v0.1

- OCR
- document-layout transformers
- VLM/LLM reasoning
- object detection / segmentation
- automatic check-field detection
- production vector databases
- fraud decision-engine orchestration
- large-scale distributed training

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
