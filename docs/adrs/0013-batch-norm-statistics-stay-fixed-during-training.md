# ADR 0013 — Batch-Norm Statistics Stay Fixed During Representation Training

Status: Accepted

## Decision

During representation training, batch-normalisation layers run in eval mode by default: they normalise with the running statistics of the pretrained backbone and do not update them, while their scale and shift parameters are trained like every other weight. `TrainingArguments.freeze_batch_norm` (default `True`) controls this; `False` restores standard train-mode batch norm. Backbones without batch norm are unaffected, as are batch-norm layers built without running statistics (`track_running_stats=False`), which normalise with batch statistics in either mode.

The layers are put back into eval mode at the start of every epoch, after anything that re-encodes the training set (hard-negative mining), because encoding restores the backbone's previous mode.

## Context

In train mode, a batch-norm layer normalises each batch with that batch's own mean and variance and moves its running statistics towards them. Few-shot batches hold a few dozen images of a few classes, which gives noisy statistics and, over ten epochs, replaces the statistics the backbone was pretrained with.

Measured on EuroSAT with ImageNet ResNet-50 (`torchvision/resnet50`) at 8 shots per class: 112-pixel input, batch size 32, ten epochs, one seed (0), scored on 100 validation images per class. The frozen baseline scored 0.770. These are single-run figures, but the gaps are 6 to 13 points:

| Learning rate | Method | Train-mode batch norm | Batch norm fixed |
|---|---|---|---|
| 1e-5 | `end_to_end` | 0.694 | 0.802 |
| 1e-5 | `supcon` | 0.668 | 0.795 |
| 2.5e-4 | `end_to_end` | 0.804 | 0.866 |
| 2.5e-4 | `supcon` | 0.750 | 0.835 |

At 1e-5 the weights barely change, yet with train-mode batch norm both measured methods fell well below the frozen baseline: the loss came from the statistics, not from learning. With `end_to_end` the effect is also visible directly. Its linear probe (ADR 0012) had a training cross-entropy of 0.037, but the first epoch's loss was 1.44 in train mode and 0.038 with fixed statistics.

DINOv2 and other transformer encoders use layer normalisation and were not affected.

## Alternatives considered

- **Keep train-mode batch norm as the default and add an opt-in.** Rejected: the default would stay the setting that is wrong for the few-shot data this library is for, and a benchmark that forgets the option makes every trained method look worse than the frozen baseline on a batch-norm backbone.
- **Always fix the statistics, without an option.** Rejected: with large training sets and batches, adapting the statistics to the new domain can help, and users should be able to choose it.
- **Also freeze the scale and shift parameters.** Not needed: they are ordinary weights, trained at the backbone's learning rate, and fixing the statistics alone removed the degradation.

## Consequences

- Results for batch-norm backbones differ from those obtained before this change.
- Training and evaluation normalise features the same way, so the features that `end_to_end`'s linear probe is fitted on match those at the start of fine-tuning.
- `freeze_batch_norm=False` gives standard train-mode behaviour, including the sensitivity to small batches described in the benchmarking guide.
