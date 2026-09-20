# ADR 0010 — Pooling Is an Explicit, Serialized Backbone Option; Adapters Reproduce Provider Preprocessing on Tensors

Status: Accepted

## Decision

1. `TimmBackbone` and `HFBackbone` take `pooling` with the values `"provider_default"`, `"cls"`, and `"mean"`. The default is the provider's canonical output. `"mean"` averages patch tokens only, excluding class and register tokens. The value is exposed as `ImageBackbone.pooling`, written to the saved model, and checked against the loaded backbone.
2. For Transformers models, `"provider_default"` means `pooler_output` when the model returns one and the class token otherwise.
3. Token pooling requested from a model that produces feature maps is rejected when the backbone is created, by a single probe forward pass that also fixes `embedding_dim`.
4. Adapters read resize, crop, interpolation, and normalisation settings from the checkpoint (torchvision weights preset, `timm` data configuration, Transformers image processor) and apply them with shared tensor code that uses the same size and crop arithmetic as the providers.
5. Each adapter reproduces an enumerated set of provider settings and raises for any other (`crop_mode="border"` in `timm`; `do_resize=False`, non-1/255 rescaling, non-square sizes, and other resampling filters in Transformers). Every preprocessing value, including the resize size, is written to the saved backbone.

## Rationale

Different pooling choices give different embedding spaces, and which one works best after few-shot adaptation is an empirical question (`docs/research/EXPERIMENT_MATRIX.md`). An explicit option makes it a recorded experimental variable instead of an accident of the adapter, and serializing it prevents a saved model from being reloaded with another policy than the head was fitted on.

Transformers models disagree on whether they have a pooler, and a pooler that was never trained for the task can be worse than the class token; naming the fallback removes the ambiguity (`AGENTS.md`, rule 6).

Providers preprocess PIL images, while EmbedFit's canonical input is a float tensor (ADR 0007), so the adapters cannot simply call the provider's transform. Reimplementing it is only safe if it lands on the same pixels. Three separate rounding choices each shifted the centre crop by one pixel and changed hard-edged test images by more than half the intensity range: rounding the resized long side instead of truncating it, rounding `image_size / crop_pct` where `timm` floors it (about a third of `timm`'s pretrained configurations have a `crop_pct` for which the two differ), and ignoring `crop_mode="squash"`. The shared code therefore mirrors the providers' arithmetic, and the tests compare against `timm.data.create_transform` and Transformers image processors across image shapes, `crop_pct` values, and crop modes.

A setting that an adapter does not understand must not be dropped silently (`AGENTS.md`, rule 6): a processor that skips rescaling feeds the model values 255 times larger than the adapter would. Raising, with the explicit options as the way out, keeps wrong embeddings from being produced quietly.

Saving the derived resize size means a later provider release that changes a model's default `crop_pct` cannot change what a saved model does.

## Consequences

Creating a `timm` or Transformers backbone costs one forward pass on a single blank image; loading a saved one loads the weights first and probes once.

`"mean"` and `"cls"` pooling need to know how many leading tokens are not patches. `timm` architectures that do not declare `num_prefix_tokens` (for example XCiT, CaiT, VOLO) support only `"provider_default"`.

An image processor whose `size` has neither `shortest_edge` nor `height`/`width` is not translated automatically; `image_size` must then be passed.

`timm` and `transformers` stay optional extras (ADR 0006). Their offline tests build models from names and configs with random weights; tests that download checkpoints are marked `network`.
