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

**Device.** `benchmarks/few_shot.py` encodes and trains on the CPU unless `--device` names another device, such as `cuda`; it never falls back to another device, and `auto` is rejected. An unavailable device is an error before any data is loaded. The device is part of a results file's configuration, so a run is not resumed on a different one. `--disable-cudnn` runs CUDA convolutions without cuDNN, for a GPU that the installed cuDNN does not support; the other operations still run on the GPU.

**Batch statistics.** By default, batch-normalisation layers keep the pretrained backbone's running statistics during representation training (`freeze_batch_norm=True`, ADR 0013); only their scale and shift are trained. With `freeze_batch_norm=False` training normalises with the statistics of each few-shot batch and overwrites the running statistics with them. On EuroSAT with ResNet-50 at 8 shots (one seed, validation split), that put `end_to_end` and `supcon` below the frozen baseline at a learning rate of 1e-5, which barely changes the weights, and at 2.5e-4 they scored 6 and 9 points lower than with fixed statistics (ADR 0013). Transformer encoders such as ViT and DINOv2 use layer normalisation and are not affected.

**Partial unfreezing.** `unfreeze_last_blocks=N` (`--unfreeze-last-blocks` in `benchmarks/few_shot.py`) trains only the backbone's last N blocks in every trained method, so all of them adapt the same parameters (ADR 0015). What a block is depends on the provider and the architecture: mostly stages for convolutional networks (four for a ResNet) and layers for transformers, but Transformers (`hf`) EfficientNet counts each MBConv block (16 for B0), where torchvision and timm count its 7 stages. The same N therefore means different amounts of the network on different backbones; `model.backbone.num_blocks` gives the count.

**Enough seeds to see the noise.** Few-shot results vary a lot between draws. With three seeds a standard deviation is barely an estimate; five or more is better, and a gap smaller than the standard deviations of the two methods is not a result.

**Metrics that match the problem.** `Trainer.evaluate` reports accuracy, balanced accuracy, and macro and weighted scores, plus precision, recall, and F1 of the positive class, ROC-AUC, PR-AUC, and the Brier score when the model has a `positive_label`. For rare events read PR-AUC and balanced accuracy, not accuracy. A metric that is undefined for a run (for example, no positives in the test set) is NaN in the records and is counted in `num_undefined` by `aggregate` rather than averaged.

## Datasets

`benchmarks/few_shot.py --dataset` and `benchmarks/embedding_quality.py --dataset` take one of six public datasets, chosen to cover domains near to and far from the backbones' pretraining data. Each downloads on first use into `--data-root` (by default the repository's git-ignored `.cache/`), under the folder in the last column. The torchvision downloads are verified against torchvision's MD5 checksums; the Hugging Face datasets are pinned to a revision, which is recorded as `dataset_revision` in the results file. Only aggregate metrics are committed, never images.

| `--dataset` | Domain | Classes | Source images | Default `--image-size` | Licence or terms | Folder |
|---|---|---|---|---|---|---|
| `eurosat` | Sentinel-2 satellite land cover | 10 | 64 × 64 | 112 | MIT (Sentinel data under the Copernicus terms) | `eurosat/` |
| `rvl-cdip` | scanned document pages | 16 | about 1000 px, grayscale | 224 | those of the Legacy Tobacco Document Library, the source of IIT-CDIP and RVL-CDIP | `huggingface/`, pages in `rvl-cdip/` |
| `dtd` | textures | 47 | 300–640 px | 224 | made available for research purposes (the authors' page) | `dtd/` |
| `pets` | cat and dog breeds | 37 | about 200–800 px | 224 | CC BY-SA 4.0; image copyright stays with the owners | `oxford-iiit-pet/` |
| `flowers` | flower species, fine-grained | 102 | shorter side 500 px | 224 | no licence stated by the authors; published for research evaluation | `flowers-102/` |
| `pcam` | lymph-node histopathology patches | 2 | 96 × 96 | 112 | CC0 1.0 | `pcam/` |

The default `--image-size` suits each dataset on its own. The cross-domain suite in the [experiment matrix](../research/EXPERIMENT_MATRIX.md#lift-rule) runs every dataset at `--image-size 112` instead, the size of the EuroSAT runs, so that the comparison is consistent and each benchmark costs about the same. This is a known limitation: at 112 px the fine texture of DTD and the fine-grained detail of Flowers-102 are reduced, which can lower every method's scores on them.

The splits come from each dataset's official ones. A benchmark needs three: the *pool* that few-shot samples are drawn from, a *validation* split to tune on, and a *test* split to report on.

| `--dataset` | Pool | Validation | Test |
|---|---|---|---|
| `eurosat` | `--pool-per-class` images per class | `--eval-per-class` per class | `--eval-per-class` per class |
| `rvl-cdip` | from the train split | the validation split | the test split |
| `dtd` | the train split of partition 1 (40 per class) | the val split (40 per class) | the test split (40 per class) |
| `pets` | the trainval split less the validation images (73–80 per class) | 20 images per class carved off trainval | the test split (88–100 per class) |
| `flowers` | the train and val splits together (20 per class) | 10 images per class carved off the test split | the rest of the test split (10–228 per class) |
| `pcam` | the first of the 13 train shards (about 20,000 images) | the valid split (32,768) | the test split (32,768) |

EuroSAT has no official split, so the three are disjoint random draws per class. For every other dataset, `--pool-per-class` and `--eval-per-class` images per class are drawn from the split; for `dtd`, `pets`, `flowers`, and `pcam` these counts are caps, and a class with fewer images in the split gives all of them (`rvl-cdip` exits instead). The carved validation images of `pets` and `flowers` are drawn with `--split-seed`, the same however `--evaluate-on` is set, so the pool and the test split do not change between tuning and reporting. Flowers-102 has only 10 training and 10 validation images per class, too few for 16 shots, so both form the pool and validation comes from the test split instead; the validation and test images stay disjoint. Evaluation images whose pixels match a pool image are dropped for every dataset.

Some tasks need more than a dataset offers:

- `open_set` holds `--unseen-classes` classes out and keeps at least two seen, so it needs `--unseen-classes` + 2 classes. On `pcam`, which has two, it exits before any training; run `--tasks rare_positive` there. It also needs `2 × shots` pool images of every seen class, so on `flowers` it runs up to 10 shots.
- `rare_positive` draws all `--negatives` from the other classes. On `pcam` they all come from the one other class, so `--pool-per-class` must be at least `--negatives` (for example `--pool-per-class 200` with the default 200); otherwise the script exits with the counts. On a two-class dataset the evaluated split's prevalence is about 0.5: the positive class is rare in training only.

## Cost

The number of training runs is `len(methods) × len(shots) × len(seeds)`, and each run builds a new model with `model_factory`. For a pretrained backbone that means reloading its weights every run, which is deliberate: a shared model would carry trained weights from one run into the next, and the evaluator rejects a factory that returns the same model or backbone twice.

## Embedding quality

Accuracy scores the head on the classes it was trained on. `benchmarks/embedding_quality.py` scores the embeddings themselves, on two tasks where a head alone cannot answer: retrieving and recognising classes that were never trained on, and ranking a rare positive class. It takes the same data, training, resume, and device options as `benchmarks/few_shot.py`, trains every method through `Trainer` exactly as `FewShotEvaluator` does, and adds `--tasks` (default `open_set rare_positive`), `--unseen-classes` (default 3, at least 2), and `--negatives` (default 200). The records are those of `FewShotEvaluator` with a `task` key, and the script prints the mean and sample standard deviation over seeds per task, metric, method, and shot count. Runs whose value is NaN are left out of the mean, and the cell then ends in `(k undefined)`.

For a given task, shot count, and seed, every method gets the same training set, the same unseen classes or positive class, and the same support set and held-out reference. The unseen classes, the positive class, and the negatives depend only on the seed, so they are the same at every shot count. Embeddings are those `model.encode` returns; similarities are cosine similarities. Images with identical pixels are dropped from the evaluated split after the first, because a copy would be retrieved at rank 1; the count is recorded as `evaluation_repeats_dropped` in the configuration, next to `evaluation_duplicates_dropped` (evaluation images that match a pool image).

**`open_set`.** Per seed, `--unseen-classes` classes are drawn at random and held out. At least two classes must be held out, because among a single unseen class retrieval and centroid accuracy are 1 by construction, and at least two must stay seen. Every method trains on `shots` pool images of each seen class (`sample_few_shot`), with the seen classes as the model's labels. The whole evaluated split, seen and unseen classes, is then encoded.

- `seen_accuracy`: the head's accuracy on the seen-class evaluation images.
- `seen_map`, `seen_precision_at_1`: leave-one-out retrieval among the seen-class evaluation images. Each image queries all the other seen-class evaluation images through a cosine `RetrievalIndex`; the relevant results are the images of its class. `seen_map` is `mean_average_precision` over the full ranking and `seen_precision_at_1` the mean of `retrieval_precision_at_k` at `k=1`. An image that is the only one of its class is not a query; with no query left, both are NaN.
- `unseen_map`, `unseen_precision_at_1`: the same among the unseen-class evaluation images only. This measures whether the geometry learned on the seen classes also groups classes it never saw.
- `unseen_centroid_accuracy`: `shots` pool images of each unseen class (the support set, drawn with `sample_few_shot`) are encoded with the trained model, and each unseen-class evaluation image is assigned to the class whose centroid (the mean of the class's unit-length support embeddings) is the most cosine-similar.
- `novelty_roc_auc`, `novelty_pr_auc`: every evaluation image gets the novelty score `1 − max cosine similarity to the training images' embeddings`; the unseen-class images are the positives. PR-AUC is average precision, so its chance level is the share of unseen-class images.
- `novelty_heldout_roc_auc`, `novelty_heldout_pr_auc`: the same, with `1 − max cosine similarity` to a held-out reference instead of the training images: `shots` further pool images of each seen class, disjoint from the training sample and the support set, drawn per shot count and seed and the same for every method; the pool therefore needs `2 × shots` images of every seen class. The trained methods were optimised on the training images, so novelty against those exact images partly measures how well a method memorised them; the held-out reference measures how well the seen classes are covered by the geometry.

**`rare_positive`.** Per seed, one class is drawn as the positive class. The training set holds `shots` positives (`sample_few_shot` of that class) and `--negatives` negatives drawn without replacement from the pool's other classes, as evenly as possible: the counts per class differ by at most one. The labels are `positive` and `negative`, with `positive` as the model's positive label. Evaluation ranks the whole evaluated split: the positive class's images against all others. The positive class is rare in training only: with the same number of evaluation images per class, the evaluated split's prevalence is one over the number of classes, 0.10 on EuroSAT. Recall at a false-positive rate of 1 % then rests on about 1 % of the negatives, about 9 of 900 on EuroSAT with 100 evaluation images per class, so it moves in coarse steps and varies a lot between seeds.

- `roc_auc`, `pr_auc`, `recall_at_fpr_0.01`, `recall_at_fpr_0.05`: from the head's predicted probability of `positive`, with the registry's `roc_auc` and `pr_auc` and `embedfit.evaluation.recall_at_fpr` (recall at the most permissive observed threshold whose false-positive rate is at most 1 % or 5 %).
- `knn_roc_auc`, `knn_pr_auc`, `knn_recall_at_fpr_0.01`, `knn_recall_at_fpr_0.05`: the same four metrics for a score that uses no head, the maximum cosine similarity to the training positives' embeddings ("similar to a known case").
- `prevalence`: the share of positive images in the evaluated split, the chance level of both PR-AUCs.

`supcon` draws class-balanced batches, so with two classes every batch holds half positives: up to half the batch size, and all of them when there are fewer, so the few positives recur in every batch while the negatives rotate. Its sampler needs at least two images per class, so `supcon` at one shot fails with a `ConfigurationError` before any training. `pair` trains at one shot: the single positive has no positive partner, but the negatives do, and every anchor still gets negative partners.

## A runnable version

[`examples/06_few_shot_benchmark.py`](../../examples/06_few_shot_benchmark.py) runs this comparison on synthetic images in a few seconds. [EuroSAT with DINOv2-small](../benchmarks/eurosat-dinov2-small.md) is a run on real images, including the learning-rate selection on a validation split.
