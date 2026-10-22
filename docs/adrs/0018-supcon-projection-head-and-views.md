# ADR 0018 — A SupCon Projection Head and Several Views per Image

Status: Accepted

## Decision

Two `supcon` options bring the training setup closer to Khosla et al. (2020). Both default to the previous behaviour and are rejected in every other mode.

### `projection_dim`

`TrainingArguments.projection_dim` (default `None`) puts a projection head on top of the backbone during `supcon` training:

```text
backbone embedding (D, normalised or not, as the model produces it)
→ Linear(D, D) → ReLU → Linear(D, projection_dim)
→ SupConLoss (which L2-normalises its input)
```

- `D` is the backbone's `embedding_dim`.
- The head is created when training starts, initialised from `args.seed` (independently of the global random state), on the backbone's device and dtype. It is trained with the backbone by the same AdamW optimiser and schedule, is part of the best-epoch state that is restored, and is used by the validation loss.
- It is discarded after training. It is not stored in the model, in a saved bundle, or in a checkpoint. The classification head, `encode`, and retrieval use the backbone embedding, exactly as without a projection head. Only the value of `projection_dim` is recorded with the training arguments.

### `contrastive_views`

`TrainingArguments.contrastive_views` (default 1) augments every image that a `supcon` batch draws that many times, independently. A value above 1 needs an `augmentation` other than `"none"`.

- The views of a draw are stacked to `[B, V, C, H, W]`, embedded in one forward pass, and given to `SupConLoss` as `[B, V, D]` (or `[B, V, projection_dim]`); views of one image are positives of each other.
- View `v` of the draw with index `d` in its epoch is augmented with the seed of draw `d × V + v` (ADR 0014). Every view has a seed of its own, a run stays reproducible and independent of `num_workers`, and with one view the seeds are exactly those of earlier versions.
- Validation batches use one view of the original images, as before.

## Context

A review of the v0.1 benchmarks found that `supcon` was applied directly to the embedding that the downstream head uses, with one view per image. Khosla et al. compute the loss on a projection of the encoder output and discard the projection; the encoder output, not the projection, is what transfers. With one view, an image's only positives are other images of its class, of which a few-shot batch holds one or two; two augmented views give every image at least one positive and double the pairs per batch.

## Alternatives considered

- **Keep the projection head and use the projected embedding downstream.** Khosla et al. report that the encoder output transfers better, and keeping the projection would change the model's embedding dimension, the bundle format, and retrieval.
- **A linear projection.** SimCLR and SupCon use a non-linear head; the hidden layer of width `D` follows the SupCon reference implementation's two-layer MLP.
- **Store the projection in the bundle.** Nothing after training uses it, and the bundle would carry weights no code path reads.
- **Several views for `pair` and `end_to_end`.** `pair` mode already augments each side independently, and cross-entropy has no notion of positives between views. Neither needs a new option for this change.
- **A view index in the augmentation seed (`[seed, stream, epoch, draw, view]`).** Equally reproducible, but it changes the seeds of one-view runs. Scaling the draw index keeps them.

## Consequences

- With `contrastive_views=V`, a `supcon` step embeds `V × B` images: memory and compute per step grow `V` times.
- The projection head adds `D × D + D × projection_dim + D + projection_dim` trained parameters during training only.
- The `supcon` validation loss with a projection head is the loss of the projected embeddings, so it is comparable only between runs with the same `projection_dim`.
