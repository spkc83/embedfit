# ADR 0021 — Training Only the Normalisation-Layer or Bias Parameters of the Backbone

Status: Accepted

## Decision

`TrainingArguments.trainable_parameters` (default `"all"`) selects which backbone parameters train, in every mode that trains the backbone (`supcon`, `pair`, `end_to_end`); `frozen` rejects any value other than `"all"`.

| Value | Backbone parameters that train |
|---|---|
| `"all"` | every one: the behaviour without the option |
| `"layernorm"` | the parameters that normalisation layers hold themselves: layer, group, RMS, batch, and instance norm, the scale and shift of batch norm included |
| `"bias"` | every parameter whose own name is `bias` or ends in `_bias` (BitFit): the biases of convolutions, linear layers, and norms, and `in_proj_bias` of `nn.MultiheadAttention` |

`"bias"` deliberately leaves out `nn.MultiheadAttention`'s `bias_k` and `bias_v`, relative-position bias tables (`relative_position_bias_table`), and LeViT's `attention_biases`: they are learned tokens or lookup tables, not additive bias terms of a layer, and their names do not match.

- **Normalisation layers are recognised by module type**, through `ImageBackbone.is_normalization(module)`. The default is an `isinstance` test against torch's `_NormBase` (batch and instance norm), `nn.LayerNorm`, `nn.GroupNorm`, and `nn.RMSNorm`, which covers their subclasses: timm's `LayerNorm2d`, `GroupNorm1`, and `BatchNormAct2d`, Transformers' ConvNeXt layer norm. The timm adapter adds timm's RMS, simple, EvoNorm, and filter-response norms, which subclass no torch norm, looked up by class name in `timm.layers` because the set changes between timm versions. The Transformers adapter adds classes whose name ends in `LayerNorm` or `RMSNorm` (`VitDetLayerNorm`, `Aimv2RMSNorm`): Transformers has no common base class for its own norms. A parameter belongs to the layer that holds it directly (`parameters(recurse=False)`), so a container named after a norm, such as timm's `ConvNormAct`, adds nothing beyond its norm.
- **Batch norm.** Its scale and shift train under `"layernorm"`, and its shift under `"bias"`, like any other selected parameter. Its running statistics follow `freeze_batch_norm` as before (ADR 0013): fixed by default. With `freeze_batch_norm=False`, only the batch-norm layers that hold a selected parameter adapt their statistics, under `"bias"` as under `"layernorm"`, and with `unfreeze_last_blocks` only those in the selected blocks. A batch-norm layer without a trainable parameter under the selection (one built with `affine=False`, for example) stays in eval mode whatever `freeze_batch_norm` says, by the rule that partial unfreezing introduced (ADR 0015).
- **With `unfreeze_last_blocks=N`** the selection applies within the part that partial unfreezing trains: only the normalisation-layer or bias parameters of the last N blocks and the tail train.
- **Parameters outside the backbone always train**: the `end_to_end` classifier and the projection head (`supcon`, or `end_to_end` with a `contrastive_weight`, ADR 0022).
- A selection that leaves no backbone parameter trainable (`"layernorm"` on a backbone without normalisation layers, or within last blocks that hold none) raises `ConfigurationError` when the `Trainer` is created.
- `ImageBackbone.unfreeze(last_blocks, trainable_parameters)` and `EmbedFitModel.unfreeze_backbone(last_blocks, trainable_parameters)` apply the selection; `ImageBackbone.trainable_parameter_names` lists it without changing the backbone. `train_representation` called directly applies it too, as it does `unfreeze_last_blocks`.
- Saved bundles and checkpoints hold the full weights in the same format; the value is recorded with the other training arguments in `training_metadata.json`.
- `benchmarks/few_shot.py --trainable-parameters` (and `benchmarks/embedding_quality.py`) passes it to every trained method. It is part of a results file's configuration; a file from before the option resumes as `"all"`.

## Context

Tuning only the normalisation layers of a pretrained ViT (LN-Tune; Basu et al., "Strong Baselines for Parameter Efficient Few-Shot Fine-tuning", AAAI 2024) (https://arxiv.org/abs/2304.01917) is a strong few-shot baseline that changes only a small fraction of the weights. Tuning only the biases (BitFit; Ben Zaken et al., 2022, https://arxiv.org/abs/2106.10199) is the other common sparse baseline. Both are more constrained than partial unfreezing: every layer adapts, but each only a little.

## Alternatives considered

- **Parameter-name patterns for norms** (`norm`, `ln_`, `layernorm`). Names differ between providers (`norm1`, `layernorm_before`, `bn1`, `downsample.1`), and a pattern that matches nothing, or something that is not a norm, goes unnoticed. Module types do not depend on the provider's naming; name matching is kept for Transformers' own norm classes, which have nothing else in common.
- **Excluding batch norm from `"layernorm"`.** Convolutional backbones would then have nothing to train, or only a final norm. Their scale and shift are the normalisation parameters of those backbones and already train under ADR 0013, with the statistics fixed.
- **A union with `unfreeze_last_blocks`** (all of the last blocks plus the norms everywhere). A second, different way to combine the two options; the intersection keeps each option meaning one thing.
- **A list of selections** (`["layernorm", "bias"]`) or arbitrary patterns. More flexible, but no tested use; one string keeps the recorded arguments and the benchmark configuration simple.

## Consequences

- Weight decay (`weight_decay`, default 0.01) pulls every selected parameter toward zero, which for the scale of a layer or batch norm means toward switching the layer's output off rather than toward its pretrained value. Use `weight_decay=0` with `"layernorm"` and `"bias"`.
- Untrained parameters are bit-identical before and after training (they are not in the optimiser, so weight decay does not touch them), and WiSE-FT interpolation in the benchmarks leaves them unchanged.
- The forward and backward passes still run through the whole network; only the optimiser state shrinks.
- `"layernorm"` on a batch-norm backbone with `freeze_batch_norm=False` also adapts the running statistics of every batch-norm layer, since every one of them has trainable parameters.
- A custom backbone provider whose own normalisation layers subclass no torch norm overrides `is_normalization`; otherwise `"layernorm"` does not see them.
