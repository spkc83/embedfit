# Benchmarking Few-Shot Methods

`FewShotEvaluator` answers one question: at a given number of labelled examples per class, does adapting the representation beat the baselines, and by how much relative to the noise between runs?

```python
from embedfit import EmbedFitModel, FewShotEvaluator, TrainingArguments

budget = {"num_epochs": 10, "batch_size": 32}
learning_rates = {"end_to_end": 1e-5, "supcon": 5e-5}  # Each chosen on a validation split.
evaluator = FewShotEvaluator(shots=[4, 8, 16, 32, 64], seeds=[13, 42, 101, 7, 99])
records = evaluator.run(
    train_pool,
    test_dataset=test,
    methods=["frozen", "end_to_end", "supcon"],
    model_factory=lambda: EmbedFitModel.from_pretrained("torchvision/resnet18", labels=labels),
    training_arguments={
        mode: TrainingArguments(training_mode=mode, learning_rate=rate, **budget)
        for mode, rate in learning_rates.items()
    },
)
summary = evaluator.aggregate(records)
```

`records` is a list of dictionaries with the keys `method`, `backbone`, `shots`, `seed`, `metric`, and `value`, so `pandas.DataFrame(records)` works directly. `aggregate` returns the mean and sample standard deviation over seeds.

## What makes a comparison fair

**Same data for every method.** For each `(shots, seed)` the few-shot sample is drawn once and every method trains on it. Differences between methods are then differences in learning, not in luck.

**A test set that training never sees.** `test_dataset` is only passed to `Trainer.evaluate`. Build it before anything else, with a split that matches how the model will be used: grouped if related items exist, out-of-time if the data has a time order ([leakage guide](leakage.md)). The pool and the test set must not overlap; check with `find_split_overlap` when samples carry identifiers.

**No selection on the test set.** The benchmark passes no validation data to training, so representation training keeps its last epoch. Hyperparameters chosen by looking at benchmark results are tuned on the test set; choose them on a separate validation split first, then run the benchmark once.

**Strong baselines with an equal budget.** `frozen` costs nothing, and `end_to_end` is the method any practitioner would try first, so a representation-tuning method has to beat both. Give every trained method the same epochs and batch size, and choose each method's learning rate on validation data: the objectives differ in scale, so a single shared rate favours whichever method it happens to suit. `end_to_end` starts from a linear probe fitted on the pretrained embeddings (ADR 0012), and logs the probe's training cross-entropy when training starts; a rate large enough to disturb the probe shows up as a first epoch whose loss is above it. Every mode is scored with a head fitted to convergence on its final embeddings (ADR 0011), so the methods differ only in how the backbone was adapted.

**A learning rate that actually trains.** The representation learning rate has to suit the backbone: around 2e-5 for a pretrained encoder, around 1e-2 for the untrained demo network in the examples. If it is too small, every trained method quietly reduces to the frozen baseline and the benchmark compares nothing. Check `TrainOutput.train_losses` from a single run before launching a benchmark: for the contrastive modes a loss that barely moves means the comparison is void. `end_to_end` starts from the probe's loss, which can already be close to zero on separable few-shot data, so a small loss there is no sign of a problem; compare its accuracy with `frozen` instead. At 4 shots and a batch size of 32, an epoch of `supcon` or `end_to_end` is one batch for up to 8 classes and two batches for 9 to 16 classes, so ten epochs are only ten or twenty optimizer steps, and the schedule decays the rate linearly to zero over them. `max_steps` (`--max-steps` in `benchmarks/few_shot.py`) sets the number of optimizer steps instead, the same for every shot count and method (ADR 0017).

**The SupCon setup.** By default `supcon` applies its loss directly to the embedding the head uses, with one view per image. `projection_dim` (`--projection-dim`) computes the loss on a projection head that is discarded after training, and `contrastive_views` (`--contrastive-views`, with an augmentation) gives every image several augmented views, as in Khosla et al. (2020) (ADR 0018). In `benchmarks/few_shot.py` both apply to the `supcon` method only, as does `--temperature`, the SupCon temperature (default 0.07).

**Device.** `benchmarks/few_shot.py` encodes and trains on the CPU unless `--device` names another device, such as `cuda`; it never falls back to another device. The device is part of a results file's configuration, so a run is not resumed on a different one. `--disable-cudnn` runs CUDA convolutions without cuDNN, for a GPU that the installed cuDNN does not support; the other operations still run on the GPU.

**Batch statistics.** By default, batch-normalisation layers keep the pretrained backbone's running statistics during representation training (`freeze_batch_norm=True`, ADR 0013); only their scale and shift are trained. With `freeze_batch_norm=False` training normalises with the statistics of each few-shot batch and overwrites the running statistics with them. On EuroSAT with ResNet-50 at 8 shots (one seed, validation split), that put `end_to_end` and `supcon` below the frozen baseline at a learning rate of 1e-5, which barely changes the weights, and at 2.5e-4 they scored 6 and 9 points lower than with fixed statistics (ADR 0013). Transformer encoders such as ViT and DINOv2 use layer normalisation and are not affected.

**Partial unfreezing.** `unfreeze_last_blocks=N` (`--unfreeze-last-blocks` in `benchmarks/few_shot.py`) trains only the backbone's last N blocks in every trained method, so all of them adapt the same parameters (ADR 0015). What a block is depends on the provider and the architecture: mostly stages for convolutional networks (four for a ResNet) and layers for transformers, but Transformers (`hf`) EfficientNet counts each MBConv block (16 for B0), where torchvision and timm count its 7 stages. The same N therefore means different amounts of the network on different backbones; `model.backbone.num_blocks` gives the count.

**Enough seeds to see the noise.** Few-shot results vary a lot between draws. With three seeds a standard deviation is barely an estimate; five or more is better, and a gap smaller than the standard deviations of the two methods is not a result.

**Metrics that match the problem.** `Trainer.evaluate` reports accuracy, balanced accuracy, and macro and weighted scores, plus precision, recall, and F1 of the positive class, ROC-AUC, PR-AUC, and the Brier score when the model has a `positive_label`. For rare events read PR-AUC and balanced accuracy, not accuracy. A metric that is undefined for a run (for example, no positives in the test set) is NaN in the records and is counted in `num_undefined` by `aggregate` rather than averaged.

## Cost

The number of training runs is `len(methods) × len(shots) × len(seeds)`, and each run builds a new model with `model_factory`. For a pretrained backbone that means reloading its weights every run, which is deliberate: a shared model would carry trained weights from one run into the next, and the evaluator rejects a factory that returns the same model or backbone twice.

## A runnable version

[`examples/06_few_shot_benchmark.py`](../../examples/06_few_shot_benchmark.py) runs this comparison on synthetic images in a few seconds. [EuroSAT with DINOv2-small](../benchmarks/eurosat-dinov2-small.md) is a run on real images, including the learning-rate selection on a validation split.
