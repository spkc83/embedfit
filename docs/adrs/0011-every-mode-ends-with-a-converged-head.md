# ADR 0011 — Every Training Mode Ends with a Converged Head

Status: Accepted. Amended by ADR 0012: `end_to_end` now starts from a linear probe, so the measurements below describe the earlier, randomly initialised baseline.

## Decision

After representation training, `Trainer` fits the classification head to convergence on the training embeddings of the final backbone in every mode, including `end_to_end`. In that mode the jointly trained linear classifier becomes a `LinearHead` and is then refitted (L-BFGS, starting from the trained weights); it is not used as it comes out of the training loop.

Validation loss during representation training is the mean over validation samples on batches that are identical in every epoch.

## Context

The first implementation of `end_to_end` used the jointly trained classifier directly and skipped head fitting, while `frozen`, `supcon`, and `pair` always received a head fitted to convergence by a solver. Measured on the synthetic benchmark (`dummy/tiny`, three classes, five seeds, ten epochs):

- With the learning rate from the documented example (2e-5) the jointly trained classifier never moved and `end_to_end` scored exactly chance (0.333 at 4, 8, and 16 shots, zero variance), while `frozen` and `supcon` scored about 0.68, because their heads do not depend on the representation learning rate.
- At a learning rate that does train (1e-2), refitting a converged head on `end_to_end`'s own unchanged backbone closed 26%, 63%, and 92% of its gap to `supcon` at 4, 8, and 16 shots. Most of the apparent advantage of contrastive training came from the head, not from the representation.

Separately, the validation batches of `end_to_end` were reshuffled every epoch. With an uneven last batch, identical weights scored differently from pass to pass (a range of 0.039 against a real improvement of 0.013 over eight epochs), and in four of five runs the best-epoch rule restored worse weights than the last epoch's.

## Rationale

`docs/research/BASELINES.md` makes strong baselines a precondition for any claim about sample efficiency. A comparison in which one arm is scored with a classifier that may not have converged measures the head-training procedure, not the representation. Giving every arm the same converged head isolates the variable the project is about: how the backbone was adapted.

With a converged head, a learning rate too small to train the backbone degrades every mode to the frozen baseline instead of sending one of them to chance.

The refit does not change what `end_to_end` learns: the backbone is still trained by cross-entropy through a linear classifier.

## Consequences

With converged heads, `end_to_end` matched or exceeded `supcon` on the synthetic benchmark (0.92 against 0.84 at 4 shots; both about 0.96 at 8 and 1.00 at 16). Synthetic bar images on an untrained backbone say nothing about real data, but they no longer suggest an advantage that was an artifact.

`end_to_end` costs one extra encoding pass and one L-BFGS fit, like the other modes.

A model trained end to end reports `head_name == "linear"` and its head is an ordinary `LinearHead`.
