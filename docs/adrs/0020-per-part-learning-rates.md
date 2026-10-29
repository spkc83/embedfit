# ADR 0020 — Per-Part Learning Rates

Status: Accepted

## Decision

Two `TrainingArguments` options give parts of the trained network their own learning rate. Both default to `None`, which keeps one learning rate, `learning_rate`, for every trained parameter.

### `head_learning_rate`

The learning rate of the parameters that representation training creates besides the backbone: the classifier of `end_to_end` (the linear probe it starts from, ADR 0012) and the projection head (`projection_dim`, ADR 0018) of `supcon`, and of `end_to_end` with a `contrastive_weight` (ADR 0022). It must be > 0. It is rejected when training creates no such parameters: in `frozen`, `pair`, and `triplet` mode, and in `supcon` mode without `projection_dim`. `end_to_end` always trains its classifier, so it always accepts one.

### `layer_lr_decay`

Layer-wise learning-rate decay of the backbone, in (0, 1]. With `n` blocks, numbered from the input side as `ImageBackbone.parameter_blocks()` lists them (ADR 0015), and `d = layer_lr_decay`:

| Part of the backbone | Learning rate |
|---|---|
| block `i`, for `i` in `[0, n)` | `learning_rate × d ** (n − 1 − i)` |
| stem (parameters before the first block) | `learning_rate × d ** n` |
| tail (parameters after the last block, such as a final norm or a pooler) | `learning_rate` |

Parameters in no block are placed exactly as for partial unfreezing (ADR 0015): before the first block they form the stem, between two blocks they join the block after them, after the last block they form the tail. A backbone that defines no blocks cannot enumerate its layers, so `layer_lr_decay` raises `ConfigurationError` when the `Trainer` is created rather than guessing a layering. `frozen` mode rejects the option. A decay of 1 gives every backbone parameter `learning_rate`.

### How the groups are built

`train_representation` builds the AdamW parameter groups from the trainable parameters only, so parameters frozen by `unfreeze_last_blocks` are in no group; the decay still counts every block, so the last trained block keeps `learning_rate` and the blocks below it keep their place in the decay. Parameters are grouped by their learning rate, one group per distinct rate, in the order of their first parameter, backbone first. With both options unset, or with values that give every parameter `learning_rate` (a decay of 1 and a head rate equal to `learning_rate`), there is one group of every trainable parameter in the order of `parameters()`, which is the optimizer of earlier versions, step for step.

All groups share everything else:

- `weight_decay` applies to every group alike.
- The warm-up and linear decay of the schedule scale each group's own learning rate by the same factor at every step (`LambdaLR` multiplies each group's initial rate).
- `max_grad_norm` clips the norm of all trained parameters together, as before.

Both values are part of the training arguments recorded in a saved model bundle and in every representation checkpoint (ADR 0016).

## Context

The benchmarks found the best backbone learning rate of DINOv2-small around 1e-6 to 3e-6. With one optimizer group, the randomly initialised `supcon` projection head and the `end_to_end` classifier train at that rate too, so in a budget of a few hundred steps they barely move from their starting point: the projection head stays close to a random projection, and the classifier stays close to the linear probe. A newly created module needs a learning rate several orders of magnitude above that of a pretrained backbone.

Layer-wise learning-rate decay (as used for BEiT and ELECTRA fine-tuning) is the usual middle ground between training the whole backbone at one rate and partial unfreezing: the lower layers, which hold the most general features, move least.

## Alternatives considered

- **A head rate as a multiple of `learning_rate`.** It ties the head to the backbone rate, so a sweep over the backbone rate changes the head rate too; an absolute rate keeps the two sweeps independent.
- **One parameter group per block.** Gives the same updates with more groups; grouping by rate keeps the defaults at exactly one group.
- **Counting layers by module depth or parameter-name patterns.** Not portable between providers (ADR 0015), and a guess for an architecture without blocks would silently mean different things on different backbones.
- **Decay the stem like block 0.** The stem sits below the first block, so it gets one more factor of `d`, as the embedding layer does in BEiT's layer-wise decay.
- **A separate weight decay for the head or for biases and norms.** Not needed for the problem above, and a change to every run's optimisation; it can be added as its own option.

## Consequences

- `learning_rate` is the rate of the last backbone block and the tail; with `layer_lr_decay` set, the effective rate of the rest of the backbone is lower, so a learning rate tuned without decay is not directly comparable to one tuned with it.
- The number of blocks differs between architectures and providers (ADR 0015), so the same decay lowers the stem's rate by different amounts on different backbones (`d ** 4` for a ResNet, `d ** 12` for ViT-B).
- A custom backbone needs `parameter_blocks()` for layer-wise decay, as for partial unfreezing.
