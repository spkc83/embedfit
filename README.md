# EmbedFit

> **Status:** pre-alpha — see [What the package provides](#what-the-package-provides)  
> **Primary modality for v0.1:** images  
> **Reference use case:** few-shot check-fraud / deposit-risk classification and similarity retrieval  
> **Package name:** `embedfit` (provisional but recommended)

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

## Start here

A coding agent should read these files in order:

1. [`AGENTS.md`](AGENTS.md)
2. [`CONTEXT.md`](CONTEXT.md)
3. [`SPEC.md`](SPEC.md)
4. [`ARCHITECTURE.md`](ARCHITECTURE.md)
5. [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)
6. [`ACCEPTANCE_CRITERIA.md`](ACCEPTANCE_CRITERIA.md)
7. [`TEST_STRATEGY.md`](TEST_STRATEGY.md)
8. [`SECURITY_PRIVACY.md`](SECURITY_PRIVACY.md)

Then execute work from [`TASKS.md`](TASKS.md).

## What the package provides

EmbedFit is built milestone by milestone against [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md); [`TASKS.md`](TASKS.md) tracks each item. The capabilities under [Core v0.1 capabilities](#core-v01-capabilities) are the target, not the current feature set.

```python
from embedfit import EmbedFitModel, ImageDataset, Trainer, TrainingArguments

model = EmbedFitModel.from_pretrained(          # downloads ImageNet weights
    "torchvision/resnet18",
    labels=["legitimate", "risk"],              # class order is as written, never sorted
    positive_label="risk",
)
embeddings = model.encode(["scan_001.png", "scan_002.png"])     # numpy array, shape [2, 512]

train = ImageDataset.from_lists(train_images, train_labels)
Trainer(model, TrainingArguments(training_mode="frozen"), train).train()

model.predict(test_images)         # labels
model.predict_proba(test_images)   # [N, 2]; columns follow model.labels
model.risk_score(test_images)      # probability of positive_label
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

`encode` accepts PIL images, numpy arrays, torch tensors, and file paths, and returns one L2-normalised row per image. Resolution, interpolation mode, and normalisation constants come from the selected checkpoint, never from a global default; pass `image_size=` to run a convolutional backbone at higher resolution.

Importable today:

| Component | Location | Purpose |
| --- | --- | --- |
| `EmbedFitModel` | `embedfit` | Backbone plus head: `from_pretrained`, `encode`, `fit_head`, `predict`, `predict_proba`, `risk_score`, `freeze_backbone` / `unfreeze_backbone`, differentiable `forward` |
| `Trainer`, `TrainingArguments` | `embedfit` | Two-phase training (representation, then head) with `train`, `predict`, and `evaluate`. `frozen` mode is available; arguments are validated on creation |
| `ImageDataset`; `ImageSample` | `embedfit`; `embedfit.data` | Sample records (`image`, `label`, optional `sample_id`, `group_id`, `timestamp`, `amount`, `metadata`); file paths stay undecoded until encoding |
| `classification_metrics`, `confusion_matrix`, `roc_auc`, `pr_auc` | `embedfit.evaluation` | Standard metrics; macro averages cover every declared class |
| `recall_at_fpr`, `precision_at_k`, `recall_at_k`, `precision_at_review_rate`, `recall_at_review_rate`, `dollar_recall`, `dollar_recall_at_review_rate` | `embedfit.evaluation` | Rare-event ranking metrics with an explicit positive label and order-independent tie handling |
| `brier_score`, `calibration_bins` (returns `CalibrationBins`), `expected_calibration_error` | `embedfit.evaluation` | Calibration of positive-class probabilities |
| `select_threshold` (returns `ThresholdResult`) | `embedfit.evaluation` | Threshold for max F1 or a target precision, recall, FPR, or review rate; for validation data only |
| `LabelMapping` | `embedfit.data` | Ordered label ↔ class-index mapping that distinguishes `0` from `False` |
| `ClassificationHead`, `LogisticRegressionHead`, `create_head`, `register_head` | `embedfit.heads` | Classifiers over `[N, D]` embeddings and class indices, constructible by name |
| `ImageBackbone` | `embedfit.backbones` | Contract for provider adapters, which own preprocessing, feature extraction, and their own serialization |
| `TorchvisionBackbone` | `embedfit.backbones` | torchvision classification models with the linear classifier removed (`torchvision/<name>`); models without one, such as SqueezeNet, are rejected |
| `create_backbone`, `register_backbone_provider` | `embedfit.backbones` | Resolve `<provider>/<name>` identifiers; the provider is never guessed |
| `to_rgb_tensor` | `embedfit.data` | Converts any supported image form to a float RGB tensor in [0, 1] |
| `set_seed` | `embedfit.utils` | Seeds Python, NumPy, and torch (CPU + CUDA); optional deterministic mode |
| `resolve_device` | `embedfit.utils` | `"auto"` resolves CUDA → MPS → CPU; an explicit device never falls back silently |
| `EmbedFitError`, `ConfigurationError`, `ImageInputError` | `embedfit.exceptions` | Error hierarchy; messages describe structure, never image or metadata content |
| `DummyBackbone` | `embedfit.backbones` | Tiny trainable CNN with no checkpoint, for offline tests and examples |
| `make_bar_images` | `embedfit.testing` | Deterministic synthetic image classes with learnable structure |

Representation training (`supcon`, `pair`, `end_to_end`), data-splitting helpers, retrieval, model save/load, and the `timm` and Hugging Face providers from [`SPEC.md`](SPEC.md) are not implemented yet; `Trainer` rejects an unavailable `training_mode` when it is created.

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

Tests that download pretrained checkpoints are opt-in: `pytest -m network`. Without the `torchvision` extra, the torchvision adapter tests are skipped.

Optional extras install backbone providers and dataset adapters: `torchvision`, `timm`, `hf`, `datasets`, `pandas`, or `all`. The core install depends only on `torch`, `numpy`, `scikit-learn`, and `pillow` ([ADR 0006](docs/adrs/0006-core-dependencies-and-testing-utilities.md)).

Branching, commit, and documentation conventions are in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Core v0.1 capabilities

- torchvision, timm, and Hugging Face vision backbones
- frozen-feature + logistic-regression baseline
- supervised contrastive adaptation
- pairwise contrastive adaptation
- optional triplet learning
- few-shot benchmark harness
- image embedding extraction
- nearest-neighbor retrieval
- fraud/risk-oriented metrics
- leakage-safe grouped and temporal evaluation helpers
- save/load model bundles
- CPU/offline test suite

## Non-goals for v0.1

- OCR
- document-layout transformers
- VLM/LLM reasoning
- object detection / segmentation
- automatic check-field detection
- production vector databases
- fraud decision-engine orchestration
- large-scale distributed training

## Naming

See [`PROJECT_NAMING.md`](PROJECT_NAMING.md). `EmbedFit` is recommended over `ImageSetFit` because it describes the abstraction rather than the first modality and avoids looking like an official SetFit package.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
