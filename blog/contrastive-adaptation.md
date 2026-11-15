<!--
Hugging Face community article (hf.co/blog/<user>).
Intended title: "Contrastive Fine-Tuning for Few-Shot Vision: A Test on Four Domains with a Decision Rule Fixed in Advance"
Set the title and the thumbnail in the web editor. FILL ME: thumbnail image.
-->

You have a pretrained vision backbone and a few labelled images per class. Should you fine-tune it with a supervised contrastive loss? We tested this on four image domains with [EmbedFit](https://github.com/spkc83/embedfit), with a decision rule fixed in advance. The protocol deviated from that plan in three ways: the step budget was not tuned, the learning-rate grids differ between methods, and the test split was used by earlier runs before the final ones (see [limitations](#limitations)).

The short answer:

- Joint cross-entropy and SupCon (`ce_supcon`) improved few-shot accuracy on three of four datasets. Apart from Pets at 4 shots (+3.4 points), the gains were 1.2 points or less.
- Neither contrastive method improved accuracy on scanned documents (RVL-CDIP). SupCon alone lost 3 to 4 points of accuracy there.
- On three of four datasets, the contrastive methods made nearest-neighbour search for rare positives worse than plain fine-tuning (`end_to_end`). On two of them, worse than the frozen pretrained backbone.
- Where contrastive training did help rare positives (RVL-CDIP), class-balanced batches with plain cross-entropy gave most of the same gain.

Read the [limitations](#limitations) before you act on any of this. The study has one backbone, one input size, settings at the edge of the tuning grid, a fixed 500-step budget, a test split that earlier runs had already used, and a lift rule with no minimum effect size.

## The question

There are four common ways to adapt a pretrained backbone with few labels:

| Name | What it does |
|---|---|
| `frozen` | Keep the backbone fixed. Fit a logistic regression on its embeddings. |
| `end_to_end` | Fine-tune the backbone with cross-entropy, starting from a linear probe (LP-FT). |
| `supcon` | Fine-tune the backbone with a supervised contrastive loss ([SupCon](https://arxiv.org/abs/2004.11362)) instead of cross-entropy, then fit a logistic regression. |
| `ce_supcon` | Fine-tune with cross-entropy plus a weighted SupCon term: `CE + w · SupCon`. |

The joint form follows [Gunel et al.](https://arxiv.org/abs/2011.01403), who report few-shot gains for language models from training a supervised contrastive term together with cross-entropy. The form `CE + w · SupCon`, with the cross-entropy at full weight, is the one of [Core-tuning](https://arxiv.org/abs/2102.06605). The design is in [ADR 0022](https://github.com/spkc83/embedfit/blob/main/docs/adrs/0022-joint-cross-entropy-and-supervised-contrastive-training.md).

We asked two things. Does contrastive adaptation beat the two standard baselines, `frozen` and `end_to_end`? And does it give better embeddings, not only a better classifier?

## The setup

| | |
|---|---|
| Backbone | `facebook/dinov2-small`, input 112 px, images first shrunk to a shorter side of 224 px |
| Datasets | Oxford-IIIT Pets (37 classes), Oxford Flowers-102 (102), DTD textures (47), RVL-CDIP scanned documents (16) |
| Shots | 4, 8, and 16 labelled images per class |
| Seeds | 5 per shot count. Every method trains on the same sample for a given seed (paired seeds). |
| Budget | 500 steps, batch size 32, natural-image augmentation, every method |
| Contrastive | Temperature 0.07, class-balanced batches, one view |
| Control | `end_to_end` with the class-balanced batches of the contrastive methods, tuned on its own |
| Hardware | One NVIDIA TITAN V |

**Tuning.** The learning rate, and for `ce_supcon` the contrastive weight, were chosen on the validation split. Tuning used seeds 0 and 1, all three shot counts, and mean accuracy. The grid was rates 3e-7, 1e-6, 3e-6, and 1e-5 (and 3e-5 for `end_to_end` and `supcon`), and weights 0.1, 1, and 10. The test split was not kept for one final run: earlier test runs exist (see [limitations](#limitations)).

**Metrics.** Four primary metrics, fixed in advance:

- `accuracy` on the test split.
- `unseen_map`: retrieval mean average precision among images of 3 classes held out of training. It asks whether the embedding groups classes it never saw.
- `novelty_heldout_pr_auc`: how well the embeddings flag images of those held-out classes as new, scored by distance to a held-out reference set of the seen classes.
- `knn_pr_auc` on a rare-positive task: the training set has a few positives of one class and 200 negatives from the others. Each test image is scored by its highest cosine similarity to a training positive.

The embedding metrics use no classifier head. On Flowers-102 the held-out-class metrics run at 4 and 8 shots only.

**The lift rule.** We fixed it before the runs. A method has *lift* on a metric at a shot count when both hold:

1. its mean over the 5 seeds is strictly above the means of `end_to_end` and `frozen`;
2. it beats `end_to_end` in at least 4 of the 5 paired seeds.

A method has lift on a dataset when it has lift at 2 or more of the 3 shot counts (both of 2 on Flowers-102 for the held-out-class metrics).

The rule has no effect-size floor. A difference of 0.002 passes if the seeds agree. So "lift" below means a consistent sign, not a large gain.

## Results

### Where each method has lift

| Metric | `supcon` | `ce_supcon` |
|---|---|---|
| `accuracy` | Pets | Pets, Flowers-102, DTD |
| `unseen_map` | Flowers-102, DTD | Pets, Flowers-102, DTD |
| `novelty_heldout_pr_auc` | Pets, Flowers-102 | Pets, Flowers-102, RVL-CDIP |
| `knn_pr_auc` | RVL-CDIP | none |

Accuracy uses each method's own head: logistic regression for `frozen` and `supcon`, linear for `end_to_end` and `ce_supcon`.

### Accuracy

Mean test accuracy over 5 seeds at 4 shots, each method with its own head:

| Dataset | `frozen` | `end_to_end` | `supcon` | `ce_supcon` |
|---|---|---|---|---|
| Pets | 0.764 | 0.790 | 0.820 | **0.825** |
| Flowers-102 | 0.932 | 0.965 | 0.967 | **0.972** |
| DTD | 0.553 | 0.590 | 0.597 | **0.602** |
| RVL-CDIP | 0.356 | **0.429** | 0.399 | 0.426 |

Difference from `end_to_end` at each shot count, with the number of seeds out of 5 in which the method wins:

| Dataset | Method | 4 shots | 8 shots | 16 shots | Lift |
|---|---|---|---|---|---|
| Pets | `supcon` | +0.030 (5/5) | +0.013 (5/5) | +0.003 (4/5) | yes |
| Pets | `ce_supcon` | +0.034 (5/5) | +0.012 (5/5) | +0.003 (4/5) | yes |
| Flowers-102 | `supcon` | +0.002 (5/5) | −0.001 (3/5) | +0.000 (3/5) | no |
| Flowers-102 | `ce_supcon` | +0.007 (5/5) | +0.002 (5/5) | +0.001 (5/5) | yes |
| DTD | `supcon` | +0.007 (3/5) | −0.004 (2/5) | −0.014 (0/5) | no |
| DTD | `ce_supcon` | +0.012 (5/5) | +0.007 (4/5) | +0.001 (3/5) | yes |
| RVL-CDIP | `supcon` | −0.029 (0/5) | −0.031 (0/5) | −0.030 (0/5) | no |
| RVL-CDIP | `ce_supcon` | −0.003 (3/5) | +0.002 (3/5) | +0.000 (2/5) | no |

The gains shrink as the shot count grows. At 16 shots they are 0.003 or less.

The verdict for `supcon` depends on the head. With the same linear head for every method, `supcon` also has lift on Flowers-102 and DTD. On DTD, both contrastive methods have lift with the linear head and not with logistic regression. If you compare methods, compare them at the same head as well as at their own.

On DTD, where both ran at 3e-7, `ce_supcon` and `supcon` give almost the same backbone. At the same head their accuracy differs by 0.002 or less in every seed and shot count. So the lift of `ce_supcon` and not `supcon` on DTD in the table above comes from the head, not the loss. With one head for both, they get the same verdict.

Weight interpolation (WiSE-FT), with the weight chosen on validation, does not change these verdicts. On RVL-CDIP it gives `ce_supcon` lift at 8 shots only, which is not enough for lift on the dataset.

### Unseen classes and novelty

- `novelty_heldout_pr_auc` improves on Pets and Flowers-102, by up to 0.041 and 0.105. For `ce_supcon` it also improves on RVL-CDIP, by up to 0.030. It does not improve on DTD.
- `unseen_map` has lift for `ce_supcon` on Pets, Flowers-102, and DTD, and for `supcon` on Flowers-102 and DTD. The gains are 0.007 or less.
- On RVL-CDIP, both methods are below `end_to_end` on `unseen_map` at every shot count. `supcon` is also below `frozen` there.

## Rare positives

This is the result we did not expect. Mean `knn_pr_auc` over 5 seeds (higher is better):

| Dataset | Shots | `frozen` | `end_to_end` | balanced control | `supcon` | `ce_supcon` |
|---|---|---|---|---|---|---|
| Pets | 4 | 0.708 | **0.870** | 0.869 | 0.402 | 0.535 |
| Pets | 16 | 0.831 | 0.969 | **0.970** | 0.782 | 0.801 |
| Flowers-102 | 4 | 0.832 | **0.910** | **0.910** | 0.707 | 0.718 |
| Flowers-102 | 16 | 0.929 | 0.977 | **0.988** | 0.865 | 0.877 |
| DTD | 4 | 0.455 | **0.570** | 0.559 | 0.540 | 0.546 |
| DTD | 16 | 0.599 | 0.730 | **0.731** | 0.631 | 0.640 |
| RVL-CDIP | 4 | 0.291 | 0.425 | 0.462 | **0.469** | 0.443 |
| RVL-CDIP | 16 | 0.273 | 0.554 | 0.597 | **0.609** | 0.595 |

The 8-shot rows follow the same pattern and are in the [full benchmark page](https://github.com/spkc83/embedfit/blob/main/docs/benchmarks/contrastive-adaptation.md#rare-positives). Each seed draws a different positive class, so the sample standard deviations over seeds are large: 0.02 to 0.29 on the first three datasets and 0.23 to 0.33 on RVL-CDIP.

- On Pets, Flowers-102, and DTD, neither contrastive method has lift at any shot count. Both are below `end_to_end` by 0.024 to 0.468. In paired seeds against `end_to_end`, they win 0 of 5 on Pets and Flowers-102, and 1 or 2 of 5 on DTD.
- On Pets and Flowers-102, both are also below `frozen` at every shot count. After contrastive training, the nearest-positive ranking is worse than with the pretrained embeddings.
- On RVL-CDIP, `supcon` has lift at all three shot counts. The next section shows where most of that gain comes from.

So a model tuned for accuracy can give worse embeddings for this kind of search. All settings here were chosen on accuracy, so these results describe those settings, not the best a contrastive method can do on rare positives.

### The Pets diagnostic

To see what drives the loss, we varied the step budget and the contrastive weight on Pets. This ran on the validation split with seeds 0 and 1 only, at fixed rates (1e-6 for `end_to_end` and `supcon`, 3e-7 for `ce_supcon`). The test split was not used. So the `ce_supcon` rows differ from `end_to_end` in the learning rate (3e-7 against 1e-6) as well as in the loss.

Mean `knn_pr_auc` at 4 / 16 shots. `frozen` scores 0.864 / 0.933.

| Method | 50 steps | 150 steps | 500 steps |
|---|---|---|---|
| `end_to_end` | 0.919 / 0.973 | 0.909 / 0.992 | 0.903 / 0.997 |
| `ce_supcon`, weight 0.1 | 0.886 / 0.952 | 0.834 / 0.940 | 0.653 / 0.878 |
| `ce_supcon`, weight 1 | 0.849 / 0.939 | 0.779 / 0.905 | 0.630 / 0.806 |
| `ce_supcon`, weight 10 | 0.831 / 0.927 | 0.684 / 0.871 | 0.603 / 0.789 |
| `supcon` | 0.620 / 0.856 | 0.612 / 0.804 | 0.412 / 0.831 |

What the table shows:

- `ce_supcon` gets worse as the step budget grows, at every weight. It gets worse as the weight grows, at every step budget.
- In this grid, the step budget mattered more than the weight. Weight 0.1 at 500 steps (0.653) is below weight 10 at 50 steps (0.831).
- Weight 10 at 500 steps is the setting tuning chose for accuracy on Pets. It is the worst `ce_supcon` setting tested here.
- Only weight 0.1 at 50 steps is above `frozen` at both shot counts, by about 0.02. It stays below `end_to_end`, which ran at a higher rate, so that gap is not due to the loss alone.
- `supcon` is far below `frozen` already at 50 steps.

What the table does not show:

- The accuracy cost of a short run. Accuracy was not measured at 50 or 150 steps.
- Whether this holds on Flowers-102, DTD, or RVL-CDIP. Only Pets was diagnosed.
- Small differences reliably. Two seeds is a small sample for a metric whose standard deviation over seeds reaches 0.17 on the Pets test split. Trust the ordering of the rows more than any single number.

**Hypothesis, not tested.** Cross-entropy on a few training images may saturate quickly, while the SupCon term on augmented views does not. The contrastive term would then dominate late in a long run, at any weight. In the rare-positive task there are only two labels, so SupCon pulls all negatives together, whatever their true class. That would erase the differences between negative classes that rare-positive ranking needs. The runs did not log the two loss terms separately or measure the spread within the negatives, so they cannot confirm this.

## Balanced batches against the contrastive loss

SupCon needs positives in every batch, so the contrastive methods use class-balanced batches. That is a second difference from `end_to_end`, besides the loss. The control isolates it: `end_to_end` with the same balanced batches.

**Accuracy.** On Pets, Flowers-102, and DTD, the control is within 0.002 of `end_to_end` at the linear head and within 0.003 at logistic regression, in all nine cells each. Balanced batches alone do not give the accuracy gain.

**Rare positives on Pets, Flowers-102, and DTD.** Balanced batches alone also do not give the `knn_pr_auc` loss. On Pets and DTD the control stays close to `end_to_end`. On Flowers-102 it is higher, but inconsistently (1 of 5 seeds), and at a different rate.

**Rare positives on RVL-CDIP.** Here the batches carry most of the gain:

| Shots | `end_to_end` | balanced control | `supcon` |
|---|---|---|---|
| 4 | 0.425 | 0.462 | 0.469 |
| 8 | 0.478 | 0.550 | 0.557 |
| 16 | 0.554 | 0.597 | 0.609 |

The control is above `end_to_end` by 0.037 to 0.072, in 4 of 5 seeds at each shot count. `supcon` is above the control by only 0.007 to 0.012, in 2 or 3 of 5 seeds.

A likely mechanism is simple arithmetic. In the rare-positive task, 4 of 204 training images are positive at 4 shots. A shuffled batch of 32 holds about 0.6 positives on average. A balanced batch over two labels holds every positive, up to 16. So on RVL-CDIP the gain over `end_to_end` comes mainly from the batches, not from the contrastive term.

**The rate confound.** Each method and the control ran at the rate its own tuning chose. On RVL-CDIP the control ran at 1e-5 and `supcon` at 3e-6. On every other dataset, at least one contrastive method also ran at a different rate from the control (see [limitations](#limitations)). So a gap between a contrastive method and the control mixes the loss and the rate. The control removes the difference in batches, not the difference in rate. The lift verdicts compare each method with `end_to_end` and `frozen` only and do not depend on the control.

## Practical guidance

These points follow from the results above. Each one is limited to one backbone at 112 px, 500 steps, and the four datasets tested.

- **For few-shot accuracy on natural images, `ce_supcon` is worth a try.** It had lift on Pets, Flowers-102, and DTD. Expect a small gain: the largest was at 4 shots, and at 16 shots it was 0.003 or less.
- **For documents, do not replace cross-entropy with SupCon.** On RVL-CDIP `supcon` lost about 0.03 accuracy at every shot count, and `ce_supcon` tied `end_to_end`.
- **Do not assume a better classifier gives better embeddings.** On three of four datasets the contrastive methods had lower `knn_pr_auc` than plain fine-tuning (`end_to_end`), and on Pets and Flowers-102 also lower than the frozen backbone. If you will use the embeddings for nearest-neighbour search, measure that use on your validation split.
- **Tune for the metric you will deploy.** Here every setting was chosen on accuracy. On Pets, the setting best for accuracy was the worst `ce_supcon` setting tested for rare positives.
- **Treat the step budget as a hyperparameter.** On Pets, rare-positive ranking fell with longer contrastive training. Whether a shorter run keeps the accuracy gain is not known from these runs.
- **Try balanced batches before a new loss.** If you see a gain from contrastive training, run `end_to_end` with the same batches. On RVL-CDIP that control gave most of the rare-positive gain.
- **Check the head.** Some verdicts changed between the linear head and logistic regression. Compare methods at the same head as well as at their own.
- **Use paired seeds and decide your rule first.** Five paired seeds and a fixed rule made small but consistent gains visible. Add an effect-size floor that matters for your use; this study had none.

## Limitations

Please read these before you act on the results.

- **Tuning at the edge of the grid.** The chosen `ce_supcon` setting is at a corner of the grid on every dataset: the lowest rate and highest weight on Pets, Flowers-102, and DTD, and the highest rate and lowest weight on RVL-CDIP. Better settings may lie outside the grid. On RVL-CDIP, `ce_supcon` ran at weight 0.1, not the weight 10 used elsewhere, and the Pets diagnostic does not cover that rate.
- **The step budget was not tuned.** Every tuning and test run used 500 steps. The lift rule said the step budget would be tuned per dataset with the rate. The diagnostic shows the budget matters for rare positives on Pets.
- **Gains are small, and the rule has no effect-size floor.** Apart from Pets at 4 shots, the accuracy gains of `ce_supcon` are 1.2 points or less. On Flowers-102 they are 0.007 or less, on scores already near 0.98. "Lift" means a consistent sign over seeds, not a large effect. The runs are also not bit-identical on the GPU: a rerun of RVL-CDIP with the same settings changed individual accuracy and `knn_pr_auc` values by up to 0.002. Verdicts that rest on gains of about 0.001, such as `ce_supcon` on Flowers-102 at 16 shots (+0.001, 5/5), may flip on a rerun.
- **One backbone, one resolution.** All results are for DINOv2-small at 112 px. At that size the fine detail of Flowers-102 and DTD, and the text of RVL-CDIP pages, is reduced for every method. Other backbones or sizes may behave differently.
- **The rate confound in the control comparison.** The contrastive methods and the control ran at different rates on most datasets (`end_to_end` / control / `supcon` / `ce_supcon`): Pets 1e-6 / 1e-6 / 1e-6 / 3e-7, Flowers-102 3e-7 / 1e-6 / 1e-6 / 3e-7, DTD 1e-6 / 1e-6 / 3e-7 / 3e-7, RVL-CDIP 1e-5 / 1e-5 / 3e-6 / 1e-5. Any comparison with the control mixes the loss and the rate.
- **Tuned on accuracy only.** The embedding results describe settings chosen for accuracy, not the best a contrastive method can do on each embedding task.
- **A small diagnostic.** One dataset, two seeds, the validation split, fixed rates, and rare-positive metrics only.
- **Rate grids differ.** `ce_supcon` and the control were tuned over four rates. `end_to_end` and `supcon` were also tuned over 3e-5, in an earlier sweep with the same settings, whose results record no code commit.
- **Four datasets, five seeds.** Other domains, and more seeds, may give different verdicts.
- **The test split was used before the final runs.** Earlier runs on all four datasets evaluated `frozen`, `end_to_end`, `supcon`, and a pair-based method on the test split, before `ce_supcon` and the control were added. The `end_to_end` and `supcon` arms here use the same rates and seeds, and reproduce their accuracy and `knn_pr_auc` to within 0.002 per value. They repeat the earlier result rather than confirm it independently. On Pets and DTD, a first test run came before the rate grid was extended down to 3e-7 and the rates were chosen again. The lift rule was fixed before any test run, but `ce_supcon` and the control were tuned when the other methods' test results were known. PatchCamelyon was also run earlier, with `supcon` and the pair-based method but without `ce_supcon` or the control; neither had lift, and it is not part of this study.
- **The backbone has seen these domains.** DINOv2's pretraining set, LVD-142M, was built partly by retrieving web images close to the training splits of Oxford-IIIT Pets, Flowers-102, and DTD ([Oquab et al., 2023](https://arxiv.org/abs/2304.07193), Table 15). So the "unseen" classes are new to the fine-tuning, not to the backbone. This may be why `frozen` already reaches an `unseen_map` of about 0.97 on Pets and 0.98 on Flowers-102. It affects how to read those results, not the paired comparisons between methods.
- **Smaller issues, the same for every method.** Rates were tuned on validation accuracy over all classes, including those later held out for the unseen-class metrics. RVL-CDIP used the natural-image augmentation, whose horizontal flips mirror document pages. The DTD pool has 3 pairs of identical images, and only exact copies between the pool and the evaluated split are dropped; near-duplicates are not detected.

## Reproducing it

The code is at [github.com/spkc83/embedfit](https://github.com/spkc83/embedfit). The full benchmark page, with every table, the tuning grid, and the per-dataset rates, is [`docs/benchmarks/contrastive-adaptation.md`](https://github.com/spkc83/embedfit/blob/main/docs/benchmarks/contrastive-adaptation.md).

The aggregate results are committed in the repository, one file per run, with each run's configuration and code commit. The validation runs of the tuning are not committed. The files hold per-seed metrics only, no images and no per-sample values:

- `benchmarks/results/contrastive_<dataset>_test.json`: accuracy of the four methods.
- `benchmarks/results/contrastive_<dataset>_balanced_test.json`: accuracy of the control.
- `benchmarks/results/contrastive_<dataset>_embedding.json` and `..._balanced_embedding.json`: the embedding metrics. On Flowers-102 they are split by task into `_embedding_open_set.json` and `_embedding_rare_positive.json`.
- `benchmarks/results/contrastive_pets_diagnostic_*.json`: the Pets diagnostic.

In these file names, `<dataset>` is `pets`, `flowers`, `dtd`, or `rvl_cdip`.

The commands for Pets follow. For another dataset, change `--dataset` to `flowers`, `dtd`, or `rvl-cdip`, and use its rates and weight from the benchmark page. On Flowers-102, run `open_set` with `--shots 4 8` and `rare_positive` with all three shot counts, as separate commands.

```bash
git clone https://github.com/spkc83/embedfit.git
cd embedfit
pip install -e ".[torchvision,hf,datasets]"
COMMON="--dataset pets --data-root .cache --backbone hf/facebook/dinov2-small --image-size 112 \
  --max-image-side 224 --shots 4 8 16 --augmentation natural --max-steps 500 --temperature 0.07 \
  --device cuda --disable-cudnn"
# Test split, with the rates chosen on validation:
python benchmarks/few_shot.py $COMMON --evaluate-on test --seeds 0 1 2 3 4 --interpolate 0.25 0.5 0.75 \
    --heads logistic_regression linear --methods frozen end_to_end supcon ce_supcon --contrastive-weight 10 \
    --learning-rate end_to_end=1e-6 supcon=1e-6 ce_supcon=3e-7 --output .runs/ces-test.json
python benchmarks/embedding_quality.py $COMMON --evaluate-on test --seeds 0 1 2 3 4 \
    --tasks open_set rare_positive --methods frozen end_to_end supcon ce_supcon --contrastive-weight 10 \
    --learning-rate end_to_end=1e-6 supcon=1e-6 ce_supcon=3e-7 --output .runs/ces-embedding.json
# The balanced control:
python benchmarks/few_shot.py $COMMON --evaluate-on test --seeds 0 1 2 3 4 --interpolate 0.25 0.5 0.75 \
    --heads logistic_regression linear --methods end_to_end --sampler end_to_end=balanced \
    --learning-rate end_to_end=1e-6 --output .runs/ces-balanced.json
python benchmarks/embedding_quality.py $COMMON --evaluate-on test --seeds 0 1 2 3 4 \
    --tasks open_set rare_positive --methods end_to_end --sampler end_to_end=balanced \
    --learning-rate end_to_end=1e-6 --output .runs/ces-balanced-embedding.json
```

The tuning commands are on the benchmark page. The scripts download the data and weights into the git-ignored `.cache/` folder on first use. On one shared TITAN V with cuDNN disabled, the test-split runs took between 407 minutes (DTD) and 640 minutes (Flowers-102) per dataset, including the control and the embedding metrics.
