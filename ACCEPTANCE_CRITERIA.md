# Acceptance Criteria

These are executable product requirements.

## A. Core model

The following must work:

```python
model = EmbedFitModel.from_pretrained(
    "torchvision/resnet18",
    labels=["legitimate", "risk"],
)

embeddings = model.encode(images)
assert embeddings.ndim == 2
assert embeddings.shape[0] == len(images)
```

## B. Frozen baseline

```python
args = TrainingArguments(training_mode="frozen")
trainer = Trainer(model=model, args=args, train_dataset=train_ds)
trainer.train()
```

Requirements:

- no backbone parameter receives gradients;
- a head is fitted;
- predictions work.

## C. Prediction semantics

For binary classification:

- positive label must be explicit;
- `predict_proba` class ordering is documented;
- risk-score helper must use explicit positive class.

## D. SupCon

With `training_mode="supcon"`:

- encoder weights change after representation training;
- loss is finite;
- encoder can be frozen afterward;
- head fitting succeeds;
- inference does not require the training sampler.

## E. Pair training

Generated ordinary class-based pairs must satisfy:

- positive → same class
- negative → different class

Relationship rules may override only when explicitly configured.

## F. Serialization

```python
model.save_pretrained(path)
loaded = EmbedFitModel.from_pretrained(path)
```

Must preserve:

- embeddings within numerical tolerance
- predictions
- probability class order
- labels
- pooling policy
- normalization policy
- backbone identifier

## G. Retrieval

After indexing embeddings:

- query returns sorted similarity scores;
- self-match is rank 1 when not excluded;
- `exclude_self=True` removes it;
- returned metadata corresponds to ID.

## H. Risk metrics

Metrics must handle:

- no positive examples with explicit defined behavior/error
- tied scores
- highly imbalanced data
- invalid amount arrays
- review rate boundaries

## I. Few-shot evaluator

For:

```text
shots = [4, 8]
seeds = [1, 2]
methods = [frozen, supcon]
```

return at least:

```text
method
shots
seed
metric
value
```

## J. Offline CI

Default `pytest` must require:

- no internet
- no GPU
- no pretrained downloads

## K. Error quality

Invalid model prefix:

```text
foo/resnet50
```

must produce an actionable provider error.

Invalid configurations, such as an unavailable training mode (`triplet`) or options that contradict the mode, must fail when `TrainingArguments` or `Trainer` is created, not halfway through training.

## L. Privacy defaults

Tests should verify that standard log messages do not include:

- file image bytes
- image arrays
- arbitrary metadata dict content
