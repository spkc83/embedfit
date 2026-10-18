# ADR 0014 — Task Augmentation Runs Before Provider Preprocessing, on Training Draws Only

Status: Accepted

## Decision

Task augmentation (SPEC §13) is a function from one canonical image tensor to another: float32 `[3, H, W]` with values in [0, 1] at native resolution, as `embedfit.data.to_rgb_tensor` produces it (ADR 0007). During representation training each drawn image becomes `backbone.preprocess(augment(to_rgb_tensor(image)))`. The backbone's preprocessing (resizing, cropping, normalisation; ADR 0003, ADR 0010) is unchanged and never knows about augmentation, and augmentation never knows the provider's sizes or normalisation constants.

`TrainingArguments.augmentation` names the pipeline: `"none"` (the default), `"natural"`, `"document"`, or a name registered with `embedfit.augmentation.register_augmentation`. Only images that representation training draws from the training set (`supcon`, `pair`, `end_to_end`) are augmented. Validation batches, hard-negative re-encoding, the `end_to_end` linear probe (ADR 0012), head fitting (ADR 0011), `encode`, `predict`, and `evaluate` use the original images. A name other than `"none"` is rejected in `frozen` mode, which trains no representation.

Every augmented image gets its own seed, derived from `TrainingArguments.seed`, the epoch, and the image's draw index in the epoch. The pipeline runs inside `torch.random.fork_rng(devices=[])` with only the global CPU generator seeded (`torch.default_generator.manual_seed`), so a pipeline that draws from torch's global generator, as `torchvision.transforms.v2` transforms do, is reproducible without changes, and the global random state (dropout masks, other code) is the same afterwards.

## Context

SPEC §13 requires task augmentation to be separate from provider-required preprocessing, with a document profile that avoids flips and extreme crops. Provider preprocessing ends in a fixed-size, normalised tensor whose constants belong to the checkpoint. The library's core depends only on numpy, Pillow, scikit-learn, and torch; torchvision is an optional extra.

`BalancedClassBatchSampler` repeats the samples of rare classes within an epoch, and pair mode can draw the same image on both sides of different pairs.

## Alternatives considered

- **Augment after `preprocess`, on the normalised fixed-size tensor.** Cheaper and batchable on the GPU, but colour changes in normalised space need the provider's mean and standard deviation, and crops would be enlarged from the already reduced image, losing the detail that high-resolution documents need (SPEC §14).
- **Augment PIL images with torchvision transforms.** The richest set of operations, but it would make torchvision a core dependency, and tensors, arrays, and 16-bit images would need lossy conversions.
- **Seed by sample position instead of by draw.** Simpler, but a sample drawn twice in one epoch, which happens for exactly the rare classes that need augmentation most, would get the same view twice.
- **torch's default per-worker seeding.** Reproducible only for a fixed `num_workers`; today, changing `num_workers` does not change results, and that stays true.
- **Augment the validation batches or head fitting too.** Validation loss chooses the kept epoch, which must not follow augmentation noise; augmented head fitting would change `fit_head` and is not specified.

## Consequences

- The default `"none"` leaves training unchanged.
- A pipeline is probed on a synthetic 32×32 image when the `Trainer` is created, and every output is checked (tensor, floating point, `[3, H, W]`, finite, in [0, 1]), so a malformed custom pipeline fails with a `ConfigurationError` that describes structure only.
- Operations run per image at native resolution on the CPU, in the data-loader workers when `num_workers > 0` and in the training process otherwise. Rotation dominates: on one CPU thread, `document` took about three times as long as the backbone's own resize on a 512×396 page and about seven times as long on a 1000×760 page. Memory grows with the image size: a draw holds several full-resolution float32 copies (`12 × H × W` bytes each, about 104 MB for a 300 dpi A4 page), so very large pages should be shrunk before training (API.md, "Augmentation").
- Randomness in a custom pipeline that comes from Python's `random` or NumPy is not seeded per draw.
- The pipeline name is recorded in the saved training metadata; nothing about augmentation is stored in the model bundle, because inference never augments.
