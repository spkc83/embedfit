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

`fit_head` may accept either images or precomputed embeddings via explicit argument/overload. Avoid ambiguous auto-detection when possible.

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
