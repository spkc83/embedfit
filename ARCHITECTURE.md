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
backbones / heads / losses / sampling / augmentation
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
- per-epoch validation loss (when `eval_dataset` is given)
- checkpoint selection, and optional representation checkpoints on disk (ADR 0016)

The two phases must be explicit in code and logs.

### `ImageBackbone`

Provider adapter.

Responsibilities:

- instantiate checkpoint
- preprocess input
- produce one vector per image
- freeze/unfreeze parameters, and define the blocks that partial unfreezing counts (ADR 0015)
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
→ restore best epoch (when eval_dataset is given)
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

### End-to-end path

The comparison baseline (ADRs 0011, 0012):

```text
fit LinearHead probe on embeddings of the untouched backbone
→ train backbone and linear classifier jointly under cross-entropy
→ restore best epoch (when eval_dataset is given)
→ freeze encoder
→ refit classifier as LinearHead to convergence
→ evaluate
```

In every path that trains the encoder, batch-norm layers stay in eval mode by default (`freeze_batch_norm=True`, ADR 0013). `unfreeze_last_blocks` restricts training to the encoder's last blocks; the frozen part, including its batch-norm statistics, does not change (ADR 0015). `trainable_parameters` restricts it further to the parameters of normalisation layers or to bias terms (ADR 0021).

## Embedding contract

Backbone output after pooling:

```text
shape: [batch, embedding_dim]
dtype: floating point
```

`EmbedFitModel.encode()` may L2-normalize.

No module may silently change embedding dimensionality unless that change is an explicitly configured projection layer.

## Projection head policy

By default there is no projection head: the backbone output that representation training adapts is the embedding that the head and retrieval use.

`supcon` training can compute its loss on a projection head (`projection_dim`, ADR 0018). The head exists only during training: downstream, the classification head, `encode`, and retrieval use the backbone's embedding, and the projection is not saved.

## Backbone providers

Provider prefix syntax:

```text
torchvision/<name>
timm/<name>
hf/<repo-id>
dummy/tiny
```

`dummy/tiny` is a small untrained CNN for offline tests.

Do not guess across providers when ambiguous.

## Dataset flow

`ImageDataset` normalizes records, not pixels.

Resizing and normalisation belong to the active backbone's `preprocess`. Task augmentation (`embedfit.augmentation`) is the only other pixel transform: it runs before `preprocess`, on representation-training draws only (ADR 0014).

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

A save bundle holds configuration, backbone weights, head state, label names, and aggregate training metadata only.

## Future ROI architecture

Potential later extension:

```text
full image ──────────────┐
field crop A ────────────┤
field crop B ────────────┼→ region encoders → fusion → embedding
signature crop ──────────┤
code-line crop ──────────┘
```

v0.1 should only define enough abstraction that future multi-region inputs do not require rewriting every head/evaluator.
