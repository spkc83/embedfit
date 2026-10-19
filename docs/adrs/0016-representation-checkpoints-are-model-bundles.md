# ADR 0016 — Representation Checkpoints Are Model Bundles with an Unfitted Head

Status: Accepted

## Decision

With `TrainingArguments.checkpoint_dir` set (default `None`: nothing is written until `save_model`), representation training writes a checkpoint after every epoch and keeps two: the best epoch and the last.

```text
<checkpoint_dir>/
├── checkpoints.json   format_version, best_epoch, last_epoch, and each epoch's losses
├── epoch-0003/        model bundle of the best epoch
└── epoch-0005/        model bundle of the last epoch
```

- Each checkpoint is an ordinary model bundle written by `serialization.save_model` (ADR 0009): pickle-free, fingerprinted, loadable with `EmbedFitModel.from_pretrained`. Its head is unfitted and of the type head fitting will use: `linear` in `end_to_end` mode, `args.head` for a model created with a head name, or a copy of the model's head instance. `fit_head` on the training data then gives the model that `Trainer` produces from that epoch. The jointly trained `end_to_end` classifier is not stored; the head refit (ADR 0011) is a convex problem, so starting it from scratch reaches the same solution within solver tolerance.
- `training_metadata.json` in each checkpoint records the training arguments without `output_dir` and `checkpoint_dir`, and the epoch, step count, and losses.
- Write order per epoch: the bundle goes to a new directory (staged in a hidden `.epoch-NNNN.*` sibling and renamed, never replacing one); then `checkpoints.json` is written to a temporary file, flushed to disk, and renamed over the old one, and the directory is flushed (POSIX); only then are epochs other than the best and the last deleted.
- If the process dies (an exception, `SIGKILL`), `checkpoints.json` names only complete bundles. Left behind can be complete epoch directories that it no longer names (up to two when the best epoch moved), and, after a kill during a save, a hidden staging directory. A power loss or operating-system crash gives no such guarantee: the bundle files themselves are not flushed to disk before the index, so the index may name a bundle whose files were lost.
- `checkpoint_dir` must be missing or empty when training starts; a second `train()` into the same directory is refused. It may not be, contain, or lie inside `output_dir`, and `Trainer.save_model` refuses such a target too, because saving replaces its target directory as a whole. It is rejected in `frozen` mode and by `FewShotEvaluator`, whose runs would all write to one directory.
- Everything that can be checked before training is checked when the `Trainer` is created: the directory, labels that JSON cannot hold, a head that cannot be saved, and a head instance that is already fitted (it belongs to the untrained backbone).
- The best epoch is still restored from memory at the end of training; checkpoints are additional.
- Training cannot be resumed from a checkpoint.

## Context

The best epoch was kept only in memory, so a long run that failed lost everything, and intermediate epochs could not be inspected. SPEC §25 forbids writing training examples to checkpoints, and ADR 0009 rules out pickles.

## Alternatives considered

- **A backbone-only directory with its own small format.** No head question, but the user would have to rebuild the model (labels, positive label, normalisation) by hand, and it would need its own loader and checks: a second, weaker bundle format.
- **A training-state file** (weights, optimiser, scheduler, gradient scaler, random-generator states) for resuming. NumPy's generator state does not load with `torch.load(weights_only=True)`, and hard-negative mining would need the training embeddings, which are derived from samples and must not be written. Resuming with weights only restarts the learning-rate schedule and is not the same run.
- **Every epoch kept** (or a configurable limit). Disk use grows with the epochs, about 1.2 GB per checkpoint for a ViT-L backbone; best and last cover both crash recovery and model selection.
- **Fixed `best/` and `last/` directories replaced each epoch.** Replacing a directory atomically needs a platform-specific rename; `save_model`'s delete-then-rename leaves a moment without `last/`.
- **`<output_dir>/checkpoints`.** `save_model` refuses a non-empty directory that is not a bundle and replaces a bundle as a whole, which would delete the checkpoints.

## Consequences

- Writing a checkpoint costs one backbone save per epoch, and disk space for two backbones.
- A checkpoint contains what a saved model contains: weights, configuration, label names, and aggregate training facts (`SECURITY_PRIVACY.md`).
- A checkpoint can start a new run (`from_pretrained`, then a new `Trainer`), which is not a resumed run.
