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
