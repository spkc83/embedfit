# Architecture

## Architectural intent

EmbedFit is centered on an embedding model, not a classifier.

```text
                    ┌─────────────────────────────┐
                    │      pretrained backbone     │
                    └──────────────┬──────────────┘
                                   │
                         representation tuning
                                   │
                                   ▼
                         task-adapted embedding
                                   │
             ┌─────────────────────┼────────────────────┐
             │                     │                    │
             ▼                     ▼                    ▼
      classification          retrieval           diagnostics
     lightweight head       nearest neighbors      drift/novelty
```

## Package dependency direction

```text
utils / data
    ↓
backbones / heads / losses / sampling
    ↓
model
    ↓
trainer
    ↓
high-level workflows
```

`evaluation` and `retrieval` should operate on arrays/embeddings without requiring Trainer.

## Main objects

### `EmbedFitModel`

Owns:

- backbone
- embedding normalization policy
- classification head
- label mapping
- serialization metadata

Does not own:

- dataset splitting
- experiment tracking
- benchmark orchestration

### `Trainer`

Owns:

- training loop
- optimizer/scheduler
- representation phase
- head fitting phase
- evaluation callbacks
- checkpoint selection

The two phases must be explicit in code and logs.

### `ImageBackbone`

Provider adapter.

Responsibilities:

- instantiate checkpoint
- preprocess input
- produce one vector per image
- freeze/unfreeze parameters
- serialize provider metadata

### `ClassificationHead`

Works on `[N, D]` feature matrices.

The default head should not depend on images or backbone internals.

### `RetrievalIndex`

Works on embeddings only.

## Training lifecycle

### Frozen path

```text
prepare dataset
→ encode training images
→ fit logistic-regression head
→ evaluate
```

### SupCon path

```text
prepare class-balanced loader
→ train encoder under supervised contrastive objective
→ restore/select best representation checkpoint
→ freeze encoder
→ encode training images
→ fit logistic-regression head
→ evaluate
```

### Pair path

```text
generate bounded pairs
→ train encoder using pair objective
→ freeze
→ fit lightweight head
→ evaluate
```

## Embedding contract

Backbone output after pooling:

```text
shape: [batch, embedding_dim]
dtype: floating point
```

`EmbedFitModel.encode()` may L2-normalize.

No module may silently change embedding dimensionality unless that change is an explicitly configured projection layer.

## Projection head policy

v0.1 should avoid a mandatory projection head.

If SupCon experiments need a projection head, implement it as an optional component with explicit choice about whether:

- the projected representation is used downstream; or
- only the backbone representation is retained.

Do not accidentally train on one representation and classify on another without documenting it.

## Backbone providers

Provider prefix syntax:

```text
torchvision/<name>
timm/<name>
hf/<repo-id>
```

Do not guess across providers when ambiguous.

## Dataset flow

`ImageDataset` normalizes records, not pixels.

Pixel transforms belong to the active backbone/data-collator pipeline.

Reason: preprocessing requirements belong to the checkpoint.

## Evaluation separation

Metrics receive:

- labels
- scores/probabilities
- optional amounts/groups

They must not call the model.

This keeps evaluation testable and reusable.

## Serialization boundary

Saved models should be reconstructable without the training dataset.

A save bundle contains model/config/head state only.

## Future ROI architecture

Potential later extension:

```text
full image ──────────────┐
payee crop ──────────────┤
amount crop ─────────────┼→ region encoders → fusion → embedding
signature crop ──────────┤
MICR crop ───────────────┘
```

v0.1 should only define enough abstraction that future multi-region inputs do not require rewriting every head/evaluator.
