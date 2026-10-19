# EmbedFit v0.1 Product Specification

## 1. Goal

Implement a reusable Python framework for few-shot supervised adaptation of pretrained image embeddings.

The default workflow should be:

```text
image dataset
→ pretrained backbone
→ optional supervised representation adaptation
→ embedding extraction
→ lightweight classifier
→ predictions / probabilities / embeddings / retrieval
```

## 2. Supported task types

Required for v0.1:

- binary single-label classification
- multiclass single-label classification

Deferred beyond v0.1 (optional for v0.2):

- multilabel classification

v0.1 handles single-label tasks only.

## 3. Required training modes

### `frozen`

Backbone receives no gradient updates.

```text
pretrained encoder
→ embeddings
→ logistic regression
```

### `supcon`

Supervised contrastive adaptation.

```text
balanced labeled batches
→ encoder
→ normalized embeddings
→ supervised contrastive loss
→ freeze encoder
→ logistic regression
```

### `pair`

Explicit positive/negative pairs.

```text
image A + image B
→ embeddings
→ pair loss
→ freeze encoder
→ lightweight classifier
```

### `triplet`

Deferred from v0.1; required by v0.2. `TrainingArguments` accepts `training_mode="triplet"` and `triplet_margin`, and `Trainer` rejects the mode as unavailable when it is created.

### `end_to_end`

A baseline using a differentiable linear classification head and cross-entropy. The head starts from a linear probe fitted on the frozen pretrained embeddings, and head and backbone are then fine-tuned together (ADR 0012).

This is for comparison, not the primary workflow.

## 4. Backbones

Required providers:

- torchvision
- timm
- Hugging Face Transformers vision encoders

Examples:

```python
EmbedFitModel.from_pretrained("torchvision/resnet50")
EmbedFitModel.from_pretrained("timm/convnext_tiny.fb_in22k_ft_in1k")
EmbedFitModel.from_pretrained("hf/facebook/dinov2-base")
```

## 5. Public API

Top-level exports:

```python
from embedfit import (
    EmbedFitModel,
    Trainer,
    TrainingArguments,
    ImageDataset,
    RetrievalIndex,
    FewShotEvaluator,
)
```

### `EmbedFitModel`

Required methods:

```python
from_pretrained(...)
encode(...)
fit_head(...)
predict(...)
predict_proba(...)
freeze_backbone()
unfreeze_backbone()
save_pretrained(...)
to(...)
train()
eval()
```

### `Trainer`

Required methods:

```python
train()
evaluate()
predict()
save_model()
```

### `TrainingArguments`

Dataclass with early validation.

Core fields:

```python
training_mode: str = "supcon"
num_epochs: int = 5
batch_size: int = 32
eval_batch_size: int = 64
learning_rate: float = 2e-5
weight_decay: float = 0.01
warmup_ratio: float = 0.0
temperature: float = 0.07
triplet_margin: float = 0.2   # triplet mode (§3), deferred to v0.2
gradient_accumulation_steps: int = 1
mixed_precision: str | None = None
max_grad_norm: float | None = 1.0
seed: int = 42
num_workers: int = 4
device: str | None = None
freeze_backbone_before_head: bool = True
head: str = "logistic_regression"
output_dir: str = "./embedfit-output"
deterministic: bool = False   # request deterministic torch algorithms (§23)
positives_per_example: int = 4   # pair mode (§11)
negatives_per_example: int = 4   # pair mode (§11)
hard_negative_fraction: float = 0.0       # pair mode: share of mined negatives (§11)
hard_negative_refresh_epochs: int = 1     # how often training embeddings are re-encoded
freeze_batch_norm: bool = True   # batch-norm running statistics fixed during training (ADR 0013)
sampler: str | None = None   # supcon / end_to_end batch sampler; None = mode default (§26)
augmentation: str = "none"   # task augmentation of training draws (§13, ADR 0014)
unfreeze_last_blocks: int | None = None   # train only the backbone's last N blocks (ADR 0015)
checkpoint_dir: str | None = None         # best and last epoch written as bundles (ADR 0016)
```

## 6. Input data

Minimum item:

```python
{"image": image, "label": label}
```

Optional:

```python
{
    "image": image,
    "label": label,
    "sample_id": "...",
    "group_id": "...",
    "timestamp": "...",
    "amount": 123.45,
    "metadata": {...},
}
```

Accepted image forms:

- PIL image
- torch tensor
- numpy array
- file path
- encoded image file bytes (PNG, JPEG, ...)

Dataset adapters should support:

- Python sequence
- `torch.utils.data.Dataset`
- Hugging Face Dataset via optional adapter (`ImageDataset.from_hf_dataset`, `datasets` extra)
- pandas DataFrame via explicit adapter (`ImageDataset.from_pandas`, `pandas` extra)

Both adapters map columns to sample fields (`columns={"image": ..., "label": ..., "sample_id": ..., "group_id": ..., "timestamp": ..., "amount": ...}`) and copy the columns listed in `metadata_columns` into `metadata`.

Corrupt images: `on_corrupt_image="raise"` (default) raises when a missing or undecodable file is encoded; `"skip"` drops such records when the dataset is built and reports how many in `ImageDataset.num_skipped`.

## 7. Backbone abstraction

Define `ImageBackbone`.

Required interface:

```python
@property
def identifier(self) -> str: ...      # "<provider>/<name>"

@property
def embedding_dim(self) -> int: ...

def preprocess(self, image): ...
def forward_features(self, pixel_values): ...
def freeze(self): ...
def unfreeze(self, last_blocks=None): ...
def parameter_blocks(self): ...       # parameter names per block, input side first; default none
def save_pretrained(self, path): ...
```

`parameter_blocks` defines the blocks that `unfreeze(last_blocks=N)` counts for partial unfreezing (ADR 0015); a backbone that defines none supports only whole-backbone training.

`preprocess` takes one image in any accepted input form and returns a float tensor `[3, H, W]` with a fixed `H` and `W` per backbone instance, so outputs can be stacked. Input-format handling is shared (`embedfit.data.to_rgb_tensor`); resizing and normalisation are the adapter's.

A frozen backbone stays in eval mode even when a parent module enters training mode, so batch-norm statistics and dropout cannot change while frozen.

Provider adapters must own provider-specific preprocessing.

Do not globally hard-code ImageNet mean/std.

## 8. Pooling

Vision transformer adapters must support:

- provider-native pooled output
- CLS token
- mean token pooling

Default to provider canonical output when available.

Pooling choice must be serialized.

## 9. Heads

Required:

- scikit-learn logistic regression
- PyTorch linear head

Deferred beyond v0.1 (optional for v0.2):

- nearest-centroid/prototype head

`LogisticRegressionHead` is the default.

It must persist class ordering and fitted estimator state.

## 10. Losses

Required:

- supervised contrastive
- cosine pair loss

Deferred to v0.2, with the `triplet` mode:

- triplet margin

Implement a small loss registry.

## 11. Sampling

Required:

- class-balanced batch sampler
- bounded pair sampler
- random pair sampling
- balanced positive/negative pair sampling

Hard-negative mining is implemented for `pair` mode: `hard_negative_fraction` sets the share of each sample's negative partners drawn from its most similar different-class training samples, and `hard_negative_refresh_epochs` sets how often the training embeddings are re-encoded. Only training data is mined.

Do not generate all O(N²) pairs by default.

## 12. Relationship-aware training

Provide a generic optional concept:

```python
RelationshipRule(
    field="fraud_ring_id",
    relation="same",
    pair_type="positive",
    weight=2.0,
)
```

The library must not know what a fraud ring is. It should simply support arbitrary relationship fields.

## 13. Augmentations

Separate:

1. task augmentation;
2. provider-required preprocessing.

Profiles:

- `none`
- `natural`
- `document`

Document profile should avoid semantic-destroying defaults such as horizontal flips and extreme crops.

`TrainingArguments.augmentation` selects the profile (default `none`) or a registered custom pipeline (§26). Augmentation acts on the canonical image tensor (float RGB `[3, H, W]` in [0, 1] at native resolution) before the backbone's own preprocessing, and only on the images that representation training (`supcon`, `pair`, `end_to_end`) draws from the training set. Validation batches, hard-negative mining, the `end_to_end` linear probe, head fitting, encoding, and prediction use the original images. A profile other than `none` is rejected in `frozen` mode. Each augmented image is seeded from the run's seed, the epoch, and its draw index in the epoch, so results do not depend on `num_workers` (ADR 0014).

- `natural`: a random crop of 50–100% of the area with an aspect ratio in [3/4, 4/3], a horizontal flip with probability 0.5, and with probability 0.8 brightness, contrast, and saturation jitter of ±20%.
- `document`: a crop of 85–100% of the area that keeps the page's aspect ratio, a rotation of up to ±2°, and brightness and contrast jitter of ±10%. No flips, no saturation or hue changes, no erasing.

## 14. High-resolution images

Backbones should permit configurable input resolution when the underlying model supports it.

Do not assume 224×224 globally.

## 15. Metrics

Standard:

- accuracy
- balanced accuracy
- precision
- recall
- F1
- macro F1
- weighted F1
- ROC-AUC where valid
- PR-AUC
- confusion matrix

Risk-oriented:

- recall at fixed FPR
- precision at review rate
- recall at review rate
- precision@K
- recall@K
- dollar recall
- dollar recall at review rate

Calibration:

- Brier score
- calibration-bin data
- optional ECE

## 16. Threshold selection

Validation-only utility:

```python
select_threshold(
    y_true,
    scores,
    positive_label="fraud",
    objective="target_fpr",
    target=0.01,
)
```

Objectives:

- max F1
- target precision
- target recall
- target FPR
- review rate

Do not tune on test data.

## 17. Splitting / leakage

Provide helpers for:

- stratified splitting
- grouped splitting
- temporal splitting
- cross-split ID overlap checks

Document temporal out-of-time testing as important for fraud-like domains.

## 18. Retrieval

Required v0.1 backend: brute-force cosine search.

API:

```python
index = RetrievalIndex(metric="cosine")
index.fit(embeddings, ids=ids, metadata=metadata)
results = index.search(query_embedding, k=10)
```

Result fields:

- id
- score
- rank
- metadata

Future backends must be swappable without changing caller code.

## 19. Few-shot benchmark harness

Support shots:

- 4
- 8
- 16
- 32
- 64
- 128

Support repeated seeds.

Output tidy records:

```text
method
backbone
shots
seed
metric
value
```

Required benchmark methods:

- frozen + LR
- end-to-end CE
- SupCon + LR

Pair + LR is desirable for v0.1.

## 20. Serialization

Saved directory:

```text
model/
├── embedfit_config.json
├── backbone/
├── head/
├── labels.json
└── training_metadata.json
```

Config must include:

- framework/version
- backbone provider
- backbone identifier
- pooling mode
- preprocessing config/fingerprint
- embedding dimension
- normalization behavior
- head type
- labels

Never serialize training samples.

## 21. Device support

Required:

- CPU
- CUDA

Best-effort:

- Apple MPS

`device="auto"` should resolve CUDA → MPS → CPU.

## 22. Mixed precision

Support:

- none
- fp16
- bf16

Only when platform supports it.

## 23. Reproducibility

Seed:

- Python
- NumPy
- torch
- CUDA

Optional deterministic mode.

## 24. Logging

Use Python logging.

Do not use uncontrolled `print` statements inside library code.

No sensitive image/metadata logging.

## 25. Security/privacy

Raw input images can contain sensitive financial information.

Defaults:

- never log image values;
- never dump examples to checkpoints;
- never put metadata into models implicitly;
- never include sensitive metadata in exception strings.

## 26. Extensibility

Allow custom:

- backbone
- head
- loss
- sampler
- metric
- augmentation pipeline (`embedfit.augmentation.register_augmentation`)

Simple registries are sufficient in v0.1.

Implemented: backbones (`register_backbone_provider`, by `<provider>/<name>` identifier), heads (`register_head`, `TrainingArguments.head`), losses (`register_loss`), batch samplers for `supcon` and `end_to_end` (`register_sampler`, `TrainingArguments.sampler`), and positive-class score metrics (`register_metric`, `Trainer.evaluate(metrics=...)`). Pair samplers and multiclass metrics are not pluggable in v0.1.

## 27. Version boundary

v0.1 is complete when:

- frozen + LR works;
- SupCon + LR works;
- pair training works or is explicitly deferred with documented issue;
- retrieval works;
- risk metrics work;
- save/load works;
- few-shot harness works;
- offline CPU tests pass;
- docs/examples exist.
