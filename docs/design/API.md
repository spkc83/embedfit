# Public API Design

## Model creation

```python
from embedfit import EmbedFitModel

model = EmbedFitModel.from_pretrained(
    "torchvision/resnet50",
    labels=["legitimate", "fraud"],
    positive_label="fraud",
)
```

## Encoding

```python
emb = model.encode(
    images,
    batch_size=64,
    normalize=True,
    convert_to_numpy=True,
)
```

### `encode` input and output semantics

- A `list` or `tuple` is a batch; any other value is a single image. A 4-D array or tensor is rejected rather than interpreted as a batch, so batch-versus-image is never inferred from shape.
- Each image may be a PIL image, a numpy array (`[H, W]` or `[H, W, C]`), a torch tensor (`[H, W]` or `[C, H, W]`), or a file path. Arrays and tensors must be `uint8` in [0, 255] or floating point in [0, 1]; out-of-range floats raise instead of being rescaled by guesswork. Single-channel input is replicated to RGB. 16-bit grayscale PIL images are scaled by 65535, and the EXIF orientation tag is applied.
- The result always has shape `[N, embedding_dim]` in input order (`N = 1` for a single image), as float32 numpy whatever the backbone's dtype, or a CPU tensor when `convert_to_numpy=False`.
- `normalize=None` applies the model's `normalize_embeddings` policy; `True`/`False` overrides it for that call.
- `encode` runs without gradients in eval mode and restores the previous mode. Training code uses `model(pixel_values)`, which is differentiable and applies the same normalisation policy.
- A malformed image raises `ImageInputError` naming its index in the input, never its pixel values or file path.

## Backbone options

Keyword arguments to `from_pretrained` that the model does not consume are forwarded to the provider adapter:

```python
EmbedFitModel.from_pretrained("torchvision/resnet50", weights="IMAGENET1K_V2", image_size=448)
EmbedFitModel.from_pretrained("torchvision/resnet18", weights=None)  # random init, no download
```

`device=None` keeps the model on CPU, `"auto"` selects CUDA → MPS → CPU, and an explicit device that is unavailable raises.

## Frozen baseline

```python
model.fit_head(train_images, train_labels)
pred = model.predict(images)
proba = model.predict_proba(images)
```

### Head semantics

- The model must be created with `labels`. Class order is exactly the order given; it is never inferred from training data or sorted.
- `fit_head`, `predict`, `predict_proba`, and `risk_score` take either images (positionally) or precomputed `embeddings=`, never both. Precomputed embeddings must come from the same model's `encode` with its default normalisation; float64 input is cast to float32.
- Training labels must all be known to the model (matching in value *and* type, so `True` is not `1`), and every model label must occur at least once. A missing class is an error rather than a silently absent probability column.
- `predict_proba` returns `[N, num_classes]`; column `i` is the probability of `model.labels[i]`.
- `predict` returns the external label values as a numpy array.
- `risk_score` returns the `positive_label` column and raises if the model was created without `positive_label`.
- Calling a prediction method before the head is fitted raises `HeadNotFittedError`.

Heads are interchangeable: `EmbedFitModel(..., head=...)` accepts a registered name or a configured `ClassificationHead` instance such as `LogisticRegressionHead(class_weight="balanced")`, and `register_head` adds new names. An empty input yields zero rows.

## Trainer

```python
args = TrainingArguments(
    training_mode="supcon",
    num_epochs=5,
    batch_size=32,
    learning_rate=2e-5,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

output = trainer.train()
metrics = trainer.evaluate()
```

`Trainer` validates its configuration when it is created: an unavailable `training_mode`, a model without labels, an empty training set, or training labels unknown to the model all raise before any work starts. `train()` seeds every random number generator from `args.seed`, runs the representation phase (skipped in `frozen` mode), freezes the backbone unless `freeze_backbone_before_head=False`, switches the model to eval mode, and fits the head. A head *instance* given to the model is kept and fitted as configured, and `args.head` must then stay at its default; otherwise `args.head` names the head, replacing the model's only when the names differ. `freeze_backbone_before_head=False` is rejected in `frozen` mode. `evaluate(dataset)` returns the label metrics of `classification_metrics`, plus `roc_auc`, `pr_auc`, and `brier_score` when the model has a `positive_label`; a metric that is undefined for the data is reported as NaN with a logged warning. `predict(dataset)` returns predictions, probabilities, the label of each probability column, and the dataset's true labels.

### Representation training (`supcon`)

- Batches come from `train_dataset` only, through `BalancedClassBatchSampler`. `batch_size` is split into as many classes as fit with at least two samples each (`classes_per_batch = min(num_classes, batch_size // 2)`, `samples_per_class = batch_size // classes_per_batch`), so the realised batch can be slightly smaller than `batch_size`.
- The loss is `SupConLoss(temperature)` on the model's own embeddings. There is no projection head: the backbone output that is trained is the one the head and retrieval use. The loss always compares L2-normalised vectors, so with the default `normalize_embeddings=True` the head sees exactly the geometry that was trained. With `normalize_embeddings=False` the head and retrieval receive unnormalised vectors whose length the objective never constrained.
- Optimisation is AdamW (`learning_rate`, `weight_decay` on all backbone parameters) with linear warm-up over `warmup_ratio` of the steps followed by linear decay to zero, gradient accumulation (each step averages over the batches it actually accumulated, including a short final group), optional gradient-norm clipping, and optional autocast (`bf16` on CPU or supporting GPUs, `fp16` on CUDA only).
- `train()` unfreezes the whole backbone before representation training, discarding any partial freezing; `freeze_backbone_before_head` decides its state afterwards.
- Images are preprocessed on the fly every epoch. `num_workers > 0` copies the backbone into each worker process, which only pays off on larger datasets.
- With an `eval_dataset`, its contrastive loss is measured after every epoch on a fixed set of balanced batches (a class-balanced estimate, not the mean over every validation sample), and the backbone weights of the best epoch (earliest on ties) are restored in memory at the end. Without one, the last epoch is kept. `eval_dataset` must be validation data, never test data; because it selects the checkpoint, `evaluate()` on it is optimistically biased, so report final numbers on a separate test set.
- `TrainOutput` reports `representation_steps`, per-epoch `train_losses` and `eval_losses`, and `best_epoch`.
- Setups that cannot train fail when the `Trainer` is created: a model label without training samples, `batch_size` below 4, an empty validation set, fewer than two classes or a class with fewer than two samples in the training or validation data, or a mixed-precision mode the device does not support. A non-finite loss during training raises `TrainingDivergedError`.
- A run is reproducible from `args.seed`: it seeds every random number generator, the batch sampler derives each epoch's batches from the seed and the epoch number, and `deterministic=True` additionally requests deterministic torch kernels.

### Representation training (`pair`)

`pair` mode shares the optimiser, schedule, accumulation, mixed precision, checkpoint selection, and failure rules of `supcon`; only the batches and the loss differ.

- Each epoch, `PairSampler` draws for every training sample up to `positives_per_example` partners from its positive pool and up to `negatives_per_example` from its negative pool, without replacement. At most `N × (positives + negatives)` pairs exist per epoch, no sample is paired with itself, and an unordered pair occurs at most once per epoch. Pairs are a function of `args.seed` and the epoch.
- Without rules, a positive pair always shares a class and a negative pair never does.
- `Trainer(..., relationship_rules=[RelationshipRule(field, relation, pair_type, weight)])` overrides that where a rule matches. `field` names a sample field or a `metadata` key, `relation` is `"same"` or `"different"`, and matching pairs get the rule's `pair_type` and `weight`. The first matching rule decides; samples without a value for the field never match. Rules are rejected in every other mode.
- A batch holds `batch_size` pairs. Both images of every pair go through the backbone in one forward pass, so the backbone sees `2 × batch_size` images and `pair` mode needs about twice the memory of `supcon` at the same `batch_size`; and `CosinePairLoss` regresses their cosine similarity onto 1 for positives and 0 for negatives (SetFit's objective), weighted by the pair weights.
- The learning-rate schedule spans the exact number of optimizer steps. Epochs differ slightly in length because of pair de-duplication, so every epoch's pairs are drawn once up front and reused during training.
- The loss's `negative_target` is fixed at 0 in `Trainer`; other values are available by using `CosinePairLoss` directly.
- A training or validation set from which no positive pair or no negative pair can be formed is rejected when the `Trainer` is created.

## Retrieval

```python
index = RetrievalIndex(metric="cosine")
index.fit(embeddings, ids=sample_ids)
hits = index.search(query_embedding, k=20)
```

## Few-shot benchmark

```python
evaluator = FewShotEvaluator(
    shots=[4, 8, 16, 32, 64],
    seeds=[13, 42, 101],
)

results = evaluator.run(
    dataset=dataset,
    methods=["frozen", "end_to_end", "supcon"],
    model_factory=model_factory,
)
```

## API stability

Before v1.0, mark experimental APIs explicitly.

Avoid exporting low-level helpers from the package root.
