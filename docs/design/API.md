# Public API Design

## Model creation

```python
from embedfit import EmbedFitModel

model = EmbedFitModel.from_pretrained(
    "torchvision/resnet50",
    labels=["routine", "risk"],
    positive_label="risk",
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
- Each image may be a PIL image, a numpy array (`[H, W]` or `[H, W, C]`), a torch tensor (`[H, W]` or `[C, H, W]`), a file path, or encoded image file bytes (`bytes`, decoded like a file). Arrays and tensors must be `uint8` in [0, 255] or floating point in [0, 1]; out-of-range floats raise instead of being rescaled by guesswork. Single-channel input is replicated to RGB. 16-bit grayscale PIL images are scaled by 65535, and the EXIF orientation tag is applied.
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

Three checkpoint providers are built in, each installed by the extra of the same name. A fourth, `dummy/tiny`, is a small untrained CNN for offline tests and needs no extra.

| Identifier | Adapter | Options |
| --- | --- | --- |
| `torchvision/<name>` | `TorchvisionBackbone` | `weights`, `image_size`, `resize_size`, `interpolation`, `mean`, `std` |
| `timm/<name>[.<tag>]` | `TimmBackbone` | `pretrained`, `pooling`, `image_size`, `resize_size`, `interpolation`, `mean`, `std` |
| `hf/<repo-id>` | `HFBackbone` | `pooling`, `config`, `image_size`, `resize_size`, `interpolation`, `mean`, `std` |

Every adapter takes its resize, crop, interpolation, and normalisation settings from the checkpoint (the torchvision weights preset, the `timm` data configuration, or the Transformers image processor) and reproduces that pipeline on float tensors with the provider's own size and crop arithmetic. Options override single settings. What each adapter reproduces is explicit, and anything else raises instead of being ignored:

- `timm`: `input_size`, `crop_pct` (floored, as `timm` does), `crop_mode` `"center"` and `"squash"`, `interpolation`, `mean`, `std`. `crop_mode="border"` raises.
- Transformers: `size` as `shortest_edge` or a square `height`/`width`, a square `crop_size` with `do_center_crop`, ConvNeXt-style `crop_pct`, nearest / bilinear / bicubic `resample`, `do_normalize` with `image_mean` / `image_std`, and rescaling by 1/255. `do_resize=False`, another rescale factor, non-square sizes, and other resampling filters raise, with a message that says to pass the preprocessing options explicitly.
- For both, passing only `image_size` keeps the checkpoint's resize-to-crop ratio, so naming the checkpoint's own size changes nothing.

The tests compare each adapter with `timm.data.create_transform` and with Transformers image processors on hard-edged images, where a one-pixel misalignment would show. Remaining differences are interpolation noise from resizing floats instead of 8-bit pixels. A provider whose package is missing raises an error naming the extra to install; nothing falls back to another provider.

`pooling` selects how a transformer encoder's tokens become one vector: `"provider_default"` (the model's own pooled output; for Transformers models the `pooler_output` when there is one, otherwise the class token), `"cls"` (the class token), or `"mean"` (the mean of the patch tokens, excluding the class token and any distillation or register tokens). The number of leading non-patch tokens comes from the model (`num_prefix_tokens` in `timm`; the class token plus `config.num_register_tokens` in Transformers); a `timm` architecture that does not declare it cannot use token pooling. Token pooling is rejected when the backbone is created if the model produces feature maps instead of tokens. The policy is part of the saved model and is verified when it is loaded.

`TimmBackbone(..., pretrained=False)` and `HFBackbone(name, config=...)` build randomly initialised models without any download, which is how the offline tests exercise them.

`device=None` keeps the model on CPU, `"auto"` selects CUDA → MPS → CPU, and an explicit device that is unavailable raises.

## Frozen baseline

```python
model.fit_head(train_images, train_labels)
pred = model.predict(images)
proba = model.predict_proba(images)
```

### Head semantics

- The model must be created with `labels`. Class order is exactly the order given; it is never inferred from training data or sorted.
- `fit_head`, `predict`, `predict_proba`, and `risk_score` take either images (positionally) or precomputed `embeddings=`, never both. Precomputed embeddings must come from the same model's `encode` with its default normalisation; float64 input is cast to float32.
- Training labels must all be known to the model (matching in value *and* type, so `True` is not `1`), and every model label must occur at least once. A missing class is an error rather than a silently absent probability column.
- `predict_proba` returns `[N, num_classes]`; column `i` is the probability of `model.labels[i]`.
- `predict` returns the external label values as a numpy array.
- `risk_score` returns the `positive_label` column and raises if the model was created without `positive_label`.
- Calling a prediction method before the head is fitted raises `HeadNotFittedError`.

Heads are interchangeable: `EmbedFitModel(..., head=...)` accepts a registered name or a configured `ClassificationHead` instance such as `LogisticRegressionHead(class_weight="balanced")`, and `register_head` adds new names. An empty input yields zero rows.

## Datasets

```python
train_ds = ImageDataset.from_lists(images, labels)
train_ds = ImageDataset.from_pandas(
    frame,
    columns={"image": "path", "label": "class", "group_id": "source"},
    metadata_columns=["source"],
    on_corrupt_image="skip",
)
train_ds = ImageDataset.from_hf_dataset(hf_dataset["train"], on_corrupt_image="skip")
train_ds.num_skipped
```

- `ImageDataset(records)` takes a sequence, or a map-style torch `Dataset`, of `ImageSample` objects or mappings with the same keys. `from_pandas` (the `pandas` extra) and `from_hf_dataset` (the `datasets` extra) read a table; a missing extra raises `ConfigurationError` naming the extra to install.
- `columns` maps the fields `image`, `label`, `sample_id`, `group_id`, `timestamp`, and `amount` to column names. `image` and `label` default to same-named columns; the other fields are read only when mapped. `metadata_columns` are copied into `metadata`. Missing values in optional fields become `None`; a missing label is an error.
- A Hugging Face `datasets.Image` column is kept as encoded bytes, so all encoded images are held in memory; `ClassLabel` columns stay integers (`features[column].int2str` gives names). Details are in [`DATA_MODEL.md`](DATA_MODEL.md#adapters).
- `on_corrupt_image="raise"` (default) reads no image when the dataset is built; an unreadable file raises `ImageInputError` when encoded. `"skip"` (on the constructor, `from_lists`, and both adapters) decodes every file-path and bytes image once when the dataset is built, drops those that are missing or cannot be decoded, and records the number in `num_skipped`.

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

Every `TrainingArguments` field, with its default, is listed in `SPEC.md` §5. `eval_batch_size` sets the images per forward pass whenever images are encoded without gradients: head fitting, `predict`, `evaluate`, hard-negative re-encoding, and the `end_to_end` linear probe. `triplet_margin` belongs to the deferred `triplet` mode and is only validated.

`Trainer` validates its configuration when it is created: an unavailable `training_mode`, a model without labels, an empty training set, or training labels unknown to the model all raise before any work starts. `train()` seeds every random number generator from `args.seed`, runs the representation phase (skipped in `frozen` mode), freezes the backbone unless `freeze_backbone_before_head=False`, switches the model to eval mode, and fits the head. A head *instance* given to the model is kept and fitted as configured, and `args.head` must then stay at its default; otherwise `args.head` names the head, replacing the model's only when the names differ. `freeze_backbone_before_head=False` is rejected in `frozen` mode. During representation training, batch-norm layers stay in eval mode, so their running statistics are those of the pretrained backbone while their scale and shift train; `freeze_batch_norm=False` switches to standard train-mode batch norm (ADR 0013). `evaluate(dataset)` returns the label metrics of `classification_metrics`, plus `roc_auc`, `pr_auc`, and `brier_score` when the model has a `positive_label`; a metric that is undefined for the data is reported as NaN with a logged warning. `evaluate(dataset, metrics=[...])` adds metrics registered with `embedfit.evaluation.register_metric` under their names; they need a `positive_label`, and an unknown, repeated, or default name raises before anything is predicted (see [EVALUATION.md](EVALUATION.md#metric-registry)). `predict(dataset)` returns predictions, probabilities, the label of each probability column, and the dataset's true labels.

### Representation training (`supcon`)

- Batches come from `train_dataset` only, through `BalancedClassBatchSampler` (the default registered sampler `"balanced"`), or through the registered sampler named by `args.sampler` (see [Batch-sampler registry](LOSSES_AND_SAMPLING.md#batch-sampler-registry)). With the default sampler, `batch_size` is split into as many classes as fit with at least two samples each (`classes_per_batch = min(num_classes, batch_size // 2)`, `samples_per_class = batch_size // classes_per_batch`), so the realised batch can be slightly smaller than `batch_size`.
- The loss is `SupConLoss(temperature)` on the model's own embeddings by default: the backbone output that is trained is the one the head and retrieval use. With `projection_dim=P` it is computed instead on a projection head `Linear(D, D) → ReLU → Linear(D, P)` over the embeddings (`D` is `embedding_dim`), initialised from `seed`, trained with the backbone, used by the validation loss, part of the restored best-epoch state, and discarded after training: it is not in the model, a saved bundle, or a checkpoint, and the head and retrieval still use the backbone embedding (ADR 0018).
- With `contrastive_views=V` (default 1; above 1 needs an `augmentation`), every drawn image is augmented `V` times, each view with its own seed (view `v` of draw `d` is seeded as draw `d × V + v`), and the loss receives `[B, V, D]` embeddings, so views of one image are positives of each other. A step then embeds `V × B` images. Validation batches use one view of the original images. The loss always compares L2-normalised vectors, so with the default `normalize_embeddings=True` the head sees exactly the geometry that was trained. With `normalize_embeddings=False` the head and retrieval receive unnormalised vectors whose length the objective never constrained.
- Optimisation is AdamW (`learning_rate`, `weight_decay` on every trainable parameter: the backbone parameters that are not frozen, plus the classifier in `end_to_end` mode and the projection head with `projection_dim`) with per-part learning rates (ADR 0020): `head_learning_rate` sets the rate of the `end_to_end` classifier and the projection head (rejected when training creates neither), and `layer_lr_decay=d` in (0, 1] trains backbone block `i` of `n` at `learning_rate × d ** (n − 1 − i)`, the stem below the first block at `learning_rate × d ** n`, and the tail after the last block (a final norm, a pooler) at `learning_rate`, with blocks counted as for `unfreeze_last_blocks`; a backbone that defines no blocks fails when the `Trainer` is created. Frozen parameters are in no group, and parameters with the same rate share one, so with both options unset there is a single group at `learning_rate`. Every group gets the same `weight_decay` and the same schedule relative to its own rate, and clipping applies to all trained parameters together. The optimiser has linear warm-up over `warmup_ratio` of the steps followed by linear decay to zero, gradient accumulation over `gradient_accumulation_steps` batches (each step averages over the batches it actually accumulated, including a short final group), gradient-norm clipping at `max_grad_norm` (`None` disables it), and optional autocast with `mixed_precision` (`bf16` on CPU or supporting GPUs, `fp16` on CUDA only).
- `train()` unfreezes the whole backbone before representation training, discarding any earlier freezing; `freeze_backbone_before_head` decides its state afterwards. With `unfreeze_last_blocks=N`, only the last N blocks of the backbone and the parameters after them (a final norm, for example) train, in every mode that trains the backbone; the stem and earlier blocks stay frozen, and so do the running statistics of their batch-norm layers, whatever `freeze_batch_norm` says (ADR 0015). Blocks are defined by the backbone (`ImageBackbone.parameter_blocks()`, counted by `num_blocks`; `block_positions()` gives each parameter's block, -1 for the stem and `num_blocks` for the tail): the four stages of a ResNet, ConvNeXt, or Swin, the transformer layers of a ViT or DINOv2. N outside `[1, num_blocks]`, or a backbone that defines no blocks, fails when the `Trainer` is created.
- `trainable_parameters` (default `"all"`) trains only part of the backbone's parameters, in every mode that trains it (ADR 0021): `"layernorm"` the parameters of its normalisation layers (layer, group, RMS, batch, and instance norm, recognised by module type through `ImageBackbone.is_normalization`; the timm and Transformers adapters add their providers' own norm classes), `"bias"` its bias terms (every parameter named `bias` or ending in `_bias`). With `unfreeze_last_blocks` it selects among the parameters of those blocks. Batch-norm running statistics still follow `freeze_batch_norm`, and a batch-norm layer left with no trainable parameter stays in eval mode. The `end_to_end` classifier and the projection head always train. A selection without any backbone parameter fails when the `Trainer` is created, and `frozen` mode rejects any value but `"all"`. `EmbedFitModel.unfreeze_backbone(last_blocks, trainable_parameters)` applies the same selection, and `ImageBackbone.trainable_parameter_names(last_blocks, trainable_parameters)` lists it.
- Images are preprocessed on the fly every epoch. `num_workers > 0` copies the backbone into each worker process, which only pays off on larger datasets.
- With `augmentation` set, the training draws are augmented and the validation batches are not (see [Augmentation](#augmentation)).
- With an `eval_dataset`, its contrastive loss is measured after every epoch on a fixed set of batches from the same sampler (with the default sampler a class-balanced estimate, not the mean over every validation sample), and the backbone weights of the best epoch (earliest on ties) are restored in memory at the end. Without one, the last epoch is kept. `eval_dataset` must be validation data, never test data; because it selects the checkpoint, `evaluate()` on it is optimistically biased, so report final numbers on a separate test set.
- With `checkpoint_dir`, every epoch is also written to disk as a model bundle with an unfitted head, and the best and the last are kept (`epoch-NNNN/`, listed in `checkpoints.json` with each epoch's losses; ADR 0016). `EmbedFitModel.from_pretrained` loads one, and `fit_head` on the training data gives the model that `train()` produces from that epoch. The directory must be missing or empty and must not overlap `output_dir`; `FewShotEvaluator` rejects it. Training cannot be resumed from a checkpoint.
- The budget is `num_epochs` epochs, or with `max_steps` set, that many optimizer steps in every trained mode (ADR 0017). `max_steps` replaces `num_epochs`, which is then ignored: training runs as many epochs as it needs and stops directly after step `max_steps`, part-way through an epoch if need be. A step is one optimizer update (one per `gradient_accumulation_steps` batches; an `fp16` step skipped on overflow does not count), and the warm-up and decay are sized on `max_steps`. A final partial epoch counts as an epoch for the training and validation losses, best-epoch selection, and checkpoints.
- `TrainOutput` reports `representation_steps`, per-epoch `train_losses` and `eval_losses`, and `best_epoch`.
- Setups that cannot train fail when the `Trainer` is created: a model label without training samples, an empty validation set, or a mixed-precision mode the device does not support; with the default `balanced` sampler, `batch_size` below 4, or fewer than two classes or a class with fewer than two samples in the training or validation data; with any other sampler, the checks of the [batch-sampler registry](LOSSES_AND_SAMPLING.md#batch-sampler-registry), including a batch in any epoch that training runs without two samples of one class (with `max_steps`, the epochs needed to take that many steps; with `contrastive_views` above 1 this applies to validation batches only, since the views of an image are positives of each other), and with `max_steps` a sampler that gives an epoch no optimizer step. A non-finite loss during training raises `TrainingDivergedError`.
- A run is reproducible from `args.seed`: it seeds every random number generator, the batch sampler derives each epoch's batches from the seed and the epoch number, and `deterministic=True` additionally requests deterministic torch kernels.

### Representation training (`pair`)

`pair` mode shares the optimiser, schedule, accumulation, mixed precision, checkpoint selection, and failure rules of `supcon`; only the batches and the loss differ.

- Each epoch, `PairSampler` draws for every training sample up to `positives_per_example` partners from its positive pool and up to `negatives_per_example` from its negative pool, without replacement. At most `N × (positives + negatives)` pairs exist per epoch, no sample is paired with itself, and an unordered pair occurs at most once per epoch. Pairs are a function of `args.seed` and the epoch.
- Without rules, a positive pair always shares a class and a negative pair never does.
- `Trainer(..., relationship_rules=[RelationshipRule(field, relation, pair_type, weight)])` overrides that where a rule matches. `field` names a sample field or a `metadata` key, `relation` is `"same"` or `"different"`, and matching pairs get the rule's `pair_type` and `weight`. The first matching rule decides; samples without a value for the field never match. A rule whose field has no value on any sample of the training or validation set is rejected when the `Trainer` is created. Rules are rejected in every other mode.
- A batch holds `batch_size` pairs. Both images of every pair go through the backbone in one forward pass, so the backbone sees `2 × batch_size` images and `pair` mode needs about twice the memory of `supcon` at the same `batch_size`; and `CosinePairLoss` regresses their cosine similarity onto 1 for positives and 0 for negatives (SetFit's objective), weighted by the pair weights.
- Without mining, an epoch's pairs depend only on the seed and the epoch, so every epoch's pairs are drawn once up front and reused during training, and the learning-rate schedule spans the exact number of optimizer steps although epochs differ slightly in length because of pair de-duplication. With `hard_negative_fraction > 0` each epoch's pairs are drawn when the epoch starts, from the current embeddings, so the schedule is planned on the largest possible epoch and the learning rate ends slightly above zero.
- The loss's `negative_target` is fixed at 0 in `Trainer`; other values are available by using `CosinePairLoss` directly.
- A training or validation set from which no positive pair or no negative pair can be formed is rejected when the `Trainer` is created.
- `hard_negative_fraction` (default 0) fills that share of every sample's negative partners with its most similar different-class training samples under the current backbone, re-encoding the training set every `hard_negative_refresh_epochs` epochs without gradients. The share is rounded half up to a whole number of negatives, and a positive share that rounds to zero is rejected. Only training data is ever mined, validation pairs stay random, and a positive `hard_negative_fraction` is rejected outside `pair` mode when `TrainingArguments` is created.

### End-to-end baseline (`end_to_end`)

The comparison baseline of ADR 0002: a linear classifier on the model's embeddings is trained jointly with the backbone under cross-entropy, using the same loop as the other modes (optimiser, schedule, accumulation, mixed precision, best-epoch restore, which here covers backbone and classifier together).

- Batches are uniformly shuffled from `train_dataset` (the registered sampler `"random"`), reshuffled every epoch from `args.seed`. There is no class balancing by default, which is part of what the baseline measures; `sampler="balanced"` or a custom registered sampler replaces the batches. With a `contrastive_weight` the default is `"balanced"` instead (see below).
- The classifier is a plain linear layer on the embeddings the model produces. It starts from a linear probe: a `LinearHead` fitted to convergence on the training embeddings of the untouched backbone, seeded with `args.seed` (ADR 0012). It is then trained with the same optimiser settings as the backbone (including weight decay and gradient clipping). Every class of the model needs at least one training sample. Without a `contrastive_weight`, `temperature` is not used, so the baseline shares no hyperparameter with the contrastive losses (ADR 0012).
- After joint training the classifier becomes the model's head as a `LinearHead` (`head_name == "linear"`) and, like the head of every other mode, is fitted to convergence on the training embeddings of the final, frozen backbone (ADR 0011). All modes are therefore scored with a converged head, but not with the same one: the other modes fit `args.head`, by default a `LogisticRegressionHead`, which is much more strongly regularised. To compare backbones only, give every mode `head="linear"` (ADR 0019). The model must have been created with a head name, and `args.head` must be the default or `"linear"`.
- Validation loss is the mean cross-entropy over every validation sample, on the same batches every epoch.
- `contrastive_weight=w` (default 0, a finite number >= 0; rejected above 0 in other modes) trains the backbone and classifier jointly under `CE + w · SupConLoss(temperature)` on every batch (ADR 0022). The contrastive term is taken on the embeddings, or with `projection_dim` on the projection head of `supcon`, which is likewise trained, restored with the best epoch, and discarded. The default sampler becomes `"balanced"`, and another sampler is checked like a `supcon` sampler, so a batch without two samples of one class fails before training. `contrastive_views=V` (with an `augmentation`) embeds `V` views of every drawn image; cross-entropy is the mean over all of them and SupCon receives `[B, V, D]`. The validation loss is the same combined objective on one view of the original images, over the fixed epoch-0 batches of the sampler (with `"balanced"`, a class-balanced sample of the validation data). Everything else, including the linear-probe start and the converged head at the end, is as above; with `w = 0` training is bit-identical to the baseline.
- `LinearHead` is also a standalone head (`head="linear"`): softmax regression fitted with full-batch L-BFGS from a seeded initialisation (or from the weights it already holds), saved without pickle like the logistic-regression head.

## Augmentation

```python
args = TrainingArguments(training_mode="supcon", augmentation="document")
```

`augmentation` adds task augmentation to representation training (SPEC §13, ADR 0014). It is separate from the backbone's preprocessing: each image that training draws from `train_dataset` is converted to the canonical float RGB tensor in [0, 1] at its native resolution, augmented, and then resized and normalised by the backbone exactly as without augmentation.

- `"none"` (default): no augmentation; training is unchanged.
- `"natural"` (`NaturalAugmentation`): a random crop of 50–100% of the area with an aspect ratio in [3/4, 4/3], a horizontal flip with probability 0.5, and, with probability 0.8, brightness, contrast, and saturation jitter of ±20%.
- `"document"` (`DocumentAugmentation`): a crop of 85–100% of the area that keeps the page's aspect ratio (at least about 92% of each side remains), a rotation of up to ±2° with the uncovered corners filled by repeating the border, and brightness and contrast jitter of ±10%. It never flips, changes saturation or hue, or erases part of the page.

Only training draws are augmented, in `supcon`, `pair` (each side of a pair independently), and `end_to_end`. Validation batches, hard-negative re-encoding, the `end_to_end` linear probe, head fitting, `encode`, `predict`, and `evaluate` see the original images. Any value other than `"none"` is rejected in `frozen` mode. The name is recorded in the saved training metadata.

Each augmented image has its own seed, derived from `seed`, the epoch, and the image's draw index in the epoch: a run is reproducible, does not depend on `num_workers`, and a sample drawn twice in an epoch gets two different views, also when a custom `sampler` repeats a position within a batch. The pipeline runs inside `torch.random.fork_rng` with only the CPU generator seeded, so the global random state is the same afterwards.

Custom pipelines are registered by name:

```python
from embedfit.augmentation import DocumentAugmentation, register_augmentation

register_augmentation("scans", lambda: DocumentAugmentation(max_rotation=1.0, crop_scale=(0.9, 1.0)))
args = TrainingArguments(training_mode="supcon", augmentation="scans")
```

A pipeline (`embedfit.augmentation.Augmentation`) is any callable that takes a float32 `[3, H, W]` CPU tensor in [0, 1] (a private copy it may modify) and returns a floating-point `[3, H', W']` tensor in [0, 1]; the size may change. Its randomness must come from torch's global generator, as in `torchvision.transforms.v2` transforms, which therefore work unchanged; randomness from Python's `random` or NumPy is not seeded per image. `"none"` is reserved. The pipeline is run once on a synthetic 32×32 image when the `Trainer` is created, and every output is checked, so a malformed pipeline raises `ConfigurationError` before training. The probe is a smoke check on one small image: a pipeline that fails only on some sizes or contents raises the same error when it meets such an image during training. `available_augmentations()` lists the names and `create_augmentation(name, **options)` builds one.

Augmentation runs per image on the CPU at native resolution: in the data-loader workers with `num_workers > 0`, and in the training process with `num_workers=0`, where it adds to every step. The `document` rotation is the most expensive operation: measured on one CPU thread, `document` took about three times as long as the backbone's own resize for a 512×396 page and about seven times as long for a 1000×760 page; `natural` took about as long as the resize at 512×396. Use `num_workers > 0` for document images.

Memory also grows with the image size, because nothing is resized before the pipeline. A draw holds several full-resolution float32 copies of the image at once, each `12 × H × W` bytes (3 channels × 4 bytes): the canonical tensor, the pipeline's intermediate and output tensors, and the copy the backbone's preprocessing makes before it resizes. `document` also builds an `8 × H × W`-byte sampling grid for the rotation. For a 300 dpi A4 page (2480×3508 pixels) one such copy is about 104 MB. This cost applies per draw in every data-loader worker, and `pair` mode loads two images per item. Shrinking large pages before training bounds it; `benchmarks/few_shot.py` stores RVL-CDIP pages with a 512-pixel short side.

## Few-shot benchmark

```python
evaluator = FewShotEvaluator(
    shots=[4, 8, 16, 32, 64],
    seeds=[13, 42, 101],
)

results = evaluator.run(
    dataset,
    test_dataset=test_dataset,
    methods=["frozen", "end_to_end", "supcon"],
    model_factory=model_factory,
)
```

`FewShotEvaluator` measures each method at each training-set size over repeated seeds.

- For every `(shots, seed)`, `sample_few_shot(dataset, shots, seed)` draws the training sample once, and every method trains on that same sample, so methods differ only in how they learn.
- Every run gets a new model from `model_factory` and `TrainingArguments` for its method (from `training_arguments`, or `TrainingArguments(training_mode=method)`) with `seed` set to the run's seed.
- `test_dataset` is only ever passed to `Trainer.evaluate`. No validation set is used inside the benchmark, so representation training keeps its last epoch and nothing is selected on test data.
- The result is a list of tidy records with the keys `method`, `backbone`, `shots`, `seed`, `metric`, and `value`, one per metric reported by `Trainer.evaluate`. `run(..., metrics=[...])` records registered metrics as well. `aggregate(records)` returns the mean and sample standard deviation over seeds per method, backbone, shots, and metric, leaving NaN values out and counting them in `num_undefined`.
- Invalid configurations fail before any training: unknown or repeated methods, arguments whose `training_mode` differs from their method, a pool too small for the largest shot count, a factory that returns the same model twice, or `metrics` that are unknown or given for models without a `positive_label`.

Interpret results with `docs/research/BASELINES.md` in mind: a representation-tuning method is only shown to be sample-efficient if it beats `frozen` and `end_to_end` on identical few-shot samples across seeds.

## Saving and loading

```python
model.save_pretrained("models/risk-v1")
trainer.save_model()                                   # to args.output_dir, with training metadata
loaded = EmbedFitModel.from_pretrained("models/risk-v1", device="auto")
```

```text
models/risk-v1/
├── embedfit_config.json      framework and format version, backbone provider / identifier /
│                             pooling / embedding dimension / preprocessing fingerprint,
│                             normalisation policy, head type
├── labels.json               labels in class-index order, positive label
├── training_metadata.json    aggregate facts only (Trainer: arguments and TrainOutput)
├── backbone/                 config.json + weights.pt (hf: config.json + model/ as safetensors),
│                             written by the backbone adapter
└── head/                     config.json, plus parameters.npz once fitted; written by the head
```

- A loaded model reproduces the saved model's embeddings, probabilities, predictions, label order, positive label, pooling, normalisation policy, and backbone dtype. Its backbone is rebuilt from the saved weights and preprocessing settings without downloading anything. It comes back in eval mode and unfrozen, whatever its state when saved.
- Saving is atomic: the bundle is assembled in a temporary sibling directory and moved into place when complete, so a failed save leaves the target as it was. The target may be missing, empty, or an existing bundle (replaced as a whole); a file or a non-empty directory that is not a bundle is refused.
- `from_pretrained` treats its argument as a saved model when the directory contains `embedfit_config.json`, and that takes precedence even if the path also reads as a `<provider>/<name>` identifier; any other directory is never mistaken for a model, and an existing path that is not a bundle gets a "not a saved model directory" error. With a saved model only `device` may be passed: labels, positive label, normalisation, head, and backbone options come from the bundle, and passing any of them raises, even with a value equal to the default.
- Bundles never contain training samples, identifiers, or metadata, and nothing is pickled: JSON for configuration, `.npz` arrays (loaded with `allow_pickle=False`) for the parameters of the logistic-regression and linear heads, a tensor-only state dict (loaded with `weights_only=True`) for torchvision, `timm`, and dummy backbones, and Transformers' safetensors format (loaded with `use_safetensors=True`) for `hf` backbones. Loading a bundle therefore does not execute code from it.
- The content of the backbone's `config.json` is fingerprinted (SHA-256 of its canonical JSON, so formatting does not matter) at save time. Loading rejects a bundle whose backbone configuration was edited afterwards; whose recorded backbone identifier, pooling, or embedding dimension, or head type, disagrees with what was loaded; whose label count disagrees with the recorded count or with the fitted head's classes; or whose `format_version` is not one the library reads. The fingerprint guards against accidental edits to preprocessing settings; it is not a signature, and weight files are not hashed.
- Corrupt files raise `ConfigurationError`: a weights file that is not a tensor-only state dict matching the architecture, or head parameters that are pickled or mis-shaped.
- A head given to the model as an instance is still treated as an instance after loading, so `Trainer` keeps it rather than replacing it by name.
- `Trainer.save_model` records the training arguments except `output_dir` and `checkpoint_dir`, since a local path can reveal a user name.
- Labels must be `str`, `int`, `float`, or `bool` to be saved, because other types do not survive JSON; this is checked before anything is written.
- A custom head is saved when its class sets `registry_name` to its registered name and implements `save` and `load`; a custom backbone provider is loadable when it is registered with a `loader`.

## Retrieval

```python
index = RetrievalIndex(metric="cosine")
index.fit(embeddings, ids=sample_ids)
hits = index.search(query_embedding, k=20)
```

## API stability

Every public API is pre-1.0 and may change between minor versions.

The package root exports only `EmbedFitModel`, `Trainer`, `TrainingArguments`, `ImageDataset`, `FewShotEvaluator`, `RetrievalIndex`, `EmbedFitError`, and `__version__`; everything else is imported from its subpackage.
