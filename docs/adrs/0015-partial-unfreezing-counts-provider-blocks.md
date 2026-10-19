# ADR 0015 — Partial Unfreezing Counts Provider-Defined Blocks

Status: Accepted

## Decision

`TrainingArguments.unfreeze_last_blocks` (default `None`, the whole backbone) trains only the last N blocks of the backbone in every mode that trains it (`supcon`, `pair`, `end_to_end`); `frozen` rejects it. A block is a stage or layer as the backbone adapter defines it through `ImageBackbone.parameter_blocks()`, which lists the parameter names of each block, input side first. `ImageBackbone.unfreeze(last_blocks=N)` and `EmbedFitModel.unfreeze_backbone(last_blocks=N)` apply it.

- Parameters in no block are placed by registration order: before the first block they form the stem, which never trains under partial unfreezing (not even with N equal to the block count); between two blocks they join the block after them; after the last block they form the tail (a final norm, for example), which trains with the last block.
- N outside `[1, num_blocks]`, or a backbone that defines no blocks, raises `ConfigurationError` when the `Trainer` is created. N is never clamped.
- While only part of the backbone trains, `ImageBackbone.train()` keeps every batch-norm layer without a trainable parameter in eval mode, whatever `freeze_batch_norm` says. Because the rule sits in `train()`, it also holds when a parent module enters train mode and when `encode` restores the previous mode (hard-negative mining). `freeze_batch_norm` keeps deciding for batch-norm layers in the trained blocks (ADR 0013).
- `train_representation` called directly applies `unfreeze_last_blocks` too, so recorded arguments always describe what trained.

Blocks per provider:

| Provider | Blocks |
|---|---|
| `dummy` | each layer of `features` with parameters, then `projection` (3 for `dummy/tiny`) |
| `torchvision` | ResNet family: `layer1`–`layer4`; ViT: encoder layers; RegNet: trunk stages; ConvNeXt and Swin: the stages of `features`, each with the downsampling layer before it; EfficientNet: the stages between stem and final convolution. Other architectures: none |
| `timm` | timm's own coarse grouping (`group_matcher(coarse=True)`), matched with `re.match` as timm does; the first number a `blocks` pattern captures is the block |
| `hf` | the entries of the outermost `nn.ModuleList` with at least two entries, the one with the most parameters if several are equally shallow (`layers` or `encoder.layer` for ViT and DINOv2; each MBConv block for EfficientNet, 16 for B0). Entry i of every sibling list of the same length joins block i, so PVT's per-stage patch embeddings (`encoder.patch_embeddings`) stay with their stages (`encoder.block`) |

Tested counts: torchvision ResNet-18 4, ViT-B/32 12, ConvNeXt-T 4, Swin-T 4, EfficientNet-B0 7, RegNetY-400MF 4; timm ResNet-18 4, ViT-Tiny 12, ConvNeXt-Atto 4, EfficientNet-B0 7, Swin-Tiny 4; Transformers ViT and DINOv2 one block per layer, PVT 4 (patch embedding and stage zipped).

## Context

The end-to-end baseline fine-tuned the whole backbone (`docs/research/BASELINES.md`). Training only the top of a pretrained network is the usual alternative for few-shot data: fewer parameters move, and the lower layers' general features stay as they were. The same option is equally meaningful for the contrastive modes, and comparisons are only fair if every method can train the same parameters.

A frozen parameter does not stop a batch-norm layer in train mode from moving its running statistics. Without the eval-mode rule, a "frozen" stem with `freeze_batch_norm=False` would compute different features after training.

## Alternatives considered

- **Parameter-name patterns** (`["layer4.*"]`). Flexible, but not portable: the same block is `layer4` in torchvision, `blocks.11` in timm, and `encoder.layer.11` in Transformers, so a benchmark configuration could not be shared between backbones, and a pattern that matches nothing would need its own error.
- **A fraction of blocks or parameters.** Still needs the block structure, or cuts blocks apart (ResNet-18's `layer4` holds most of its weights), and its rounding becomes part of the API.
- **Honour whatever `requires_grad` the caller set before `train()`.** Not recorded in `TrainingArguments`, so a run could not be reproduced from its saved arguments.
- **Hold whole frozen blocks in eval mode**, dropout included. Correct for batch norm as well, but needs module-level bookkeeping, and leaving dropout active in frozen layers is the common practice when fine-tuning only the top of a network.

## Consequences

- Block counts differ between architecture families (mostly stages for convolutional networks, layers for transformers) and sometimes between providers for the same architecture (EfficientNet-B0: 7 stages in torchvision and timm, 16 MBConv blocks in Transformers), so N is comparable only for the same backbone. `num_blocks` reports it.
- A custom backbone provider supports partial unfreezing by overriding `parameter_blocks`; without it, only `unfreeze_last_blocks=None` is available.
- Transformers models are discovered structurally; hierarchical models such as Swin keep each stage's downsampling layer with that stage (at its end), unlike timm and torchvision. Only ViT, DINOv2, and PVT are tested. With the zipped lists, PVT's first block holds its first patch embedding, which is the stem in other providers' terms, so N equal to the block count trains it too.
- With partial unfreezing on, a batch-norm layer built without affine parameters is held in eval mode even inside a trained block. Pretrained backbones of the three providers use affine batch norm.
- The forward pass still runs through the frozen layers every step; only the backward pass and the optimiser state shrink.
