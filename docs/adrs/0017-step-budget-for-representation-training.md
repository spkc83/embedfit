# ADR 0017 — A Step Budget for Representation Training

Status: Accepted

## Decision

`TrainingArguments.max_steps` (default `None`) budgets representation training in optimizer steps instead of epochs, in every mode that trains the backbone (`supcon`, `pair`, `end_to_end`); it is rejected in `frozen` mode.

- When `max_steps` is set, it replaces `num_epochs`, which is ignored whatever its value, as with `max_steps` in Hugging Face's `TrainingArguments`. Training runs as many epochs as it needs and stops directly after optimizer step `max_steps`, part-way through an epoch if that is where the budget ends.
- A step is an optimizer step: one per group of `gradient_accumulation_steps` batches (the last group of an epoch can be shorter). Under `fp16`, a step that the gradient scaler skips on overflow does not count, as before, so the budget is `max_steps` real updates.
- The warm-up and linear decay of the learning rate are sized on `max_steps`, so the rate reaches zero on the last step.
- Epochs keep their meaning. A final partial epoch counts as an epoch: its mean training loss is recorded, validation loss is measured after it, it can be the best epoch, and it is checkpointed (ADR 0016). Samplers still get `set_epoch` at the start of every epoch, and hard-negative mining still refreshes on its epoch schedule.
- The setup check of custom batch samplers (SPEC §26) iterates every epoch that training will run. That count is computed when the `Trainer` is created from the batches per epoch: the sampler's length, the drawn pairs per epoch in `pair` mode, or, with hard-negative mining, the largest number of pairs an epoch can hold, the same plan the learning-rate schedule uses without `max_steps`. Steps skipped under `fp16` can add epochs beyond that plan, and their batches are not checked in advance. A sampler that gives an epoch of the plan no optimizer step is rejected, since the budget could never be reached. The plan is computed only for the samplers that are checked, so `pair` mode draws no pairs ahead for it.
- `TrainOutput.representation_steps` equals `max_steps` after a completed run. The value is recorded with the other training arguments in saved models and checkpoints.

## Context

The budget was epochs only. In a few-shot run an epoch is a handful of batches: at 4 shots of 10 classes and batch size 32 a `supcon` epoch is two batches, so ten epochs were twenty optimizer steps, with the learning rate decaying to zero over them. The v0.1 benchmarks therefore gave the contrastive objectives very little optimisation, and the step count changed with the shot count, so methods and shot counts could not be given the same amount of training. A step budget fixes the number of updates directly.

## Alternatives considered

- **Reject `max_steps` together with a non-default `num_epochs`.** It would catch a user who sets both expecting one to cap the other, but it makes the default `num_epochs=5` a value that must be remembered as untouched, and benchmark code that builds arguments from a shared budget would have to drop `num_epochs` conditionally. Ignoring `num_epochs` matches Hugging Face, where users already know the rule.
- **Use `min(num_epochs, epochs needed for max_steps)`.** Two budgets that cap each other give a step count that still depends on the data size, which is what the step budget is meant to remove.
- **Finish the epoch in which the budget runs out.** The step count would again depend on the epoch length, and the schedule would have to run past zero or be sized on a number the user did not set.
- **Count batches instead of optimizer steps.** With gradient accumulation the number of updates would then be a fraction of `max_steps`, unlike every other step count in the library (`representation_steps`, the schedule).

## Consequences

- At few shots, training a fixed number of steps means many epochs over the same images. Validation loss after every epoch, or augmentation (ADR 0014), become more important against overfitting.
- `num_epochs` is no longer the training length when `max_steps` is set; logs report `epoch (step k/max_steps)`.
