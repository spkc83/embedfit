# Contrastive Adaptation Across Domains

`supcon` and `ce_supcon` (joint cross-entropy and SupCon, ADR 0022) against the `frozen` and `end_to_end` baselines on Oxford-IIIT Pets, Oxford Flowers-102, and DTD, under the lift rule of the [experiment matrix](../research/EXPERIMENT_MATRIX.md#lift-rule). A control run of `end_to_end` with class-balanced batches isolates the effect of the contrastive term from the effect of the batches. All runs use DINOv2-small at 112 px on a GPU.

RVL-CDIP belongs to this suite but is not included yet ([RVL-CDIP](#rvl-cdip)).

## Result

- **Accuracy.** `ce_supcon` has lift on all three datasets. With each method's own head, `supcon` has lift on Pets alone. With the same linear head for every method, it also has lift on Flowers-102 and DTD, so its result there depends on the head. The gain is largest at 4 shots: 0.034 on Pets, 0.012 on DTD, and 0.007 on Flowers-102 for `ce_supcon`. It is 0.003 or less at 16 shots.
- **Unseen classes and novelty.** `novelty_heldout_pr_auc` improves on Pets and Flowers-102, by up to 0.041 and 0.105, and not on DTD. `unseen_map` has lift for most cells, by 0.007 or less.
- **Rare positives.** Neither method has lift on `knn_pr_auc` at any shot count on any dataset. Both are below `end_to_end` by 0.024 to 0.468, and on Pets and Flowers-102 they are also below `frozen`.
- **Batches.** Class-balanced batches alone do not reproduce the accuracy gain or the `knn_pr_auc` loss. Both come with the contrastive term.
- **No setting is best for both kinds of task.** On a Pets diagnostic, a short run at a low contrastive weight recovers most of the rare-positive loss. The settings the suite chose on accuracy, the longest run and the highest weight, are the worst of those tested for rare positives.

The lift rule has no effect-size floor. A difference of 0.002 passes it if the seeds agree, as several cells below do, so "lift" here means a consistent sign, not a large gain.

## Accuracy

Test accuracy, mean ± sample standard deviation over 5 seeds, fully trained (no weight interpolation). `end_to_end balanced` is the control. Every method is scored with both heads (ADR 0019). A method's own head is logistic regression for `frozen` and `supcon`, and linear for `end_to_end` and `ce_supcon`.

Pets (37 classes):

| Method | Linear, 4 shots | 8 shots | 16 shots | Logistic, 4 shots | 8 shots | 16 shots |
|---|---|---|---|---|---|---|
| `frozen` | 0.771 ± 0.013 | 0.830 ± 0.008 | 0.860 ± 0.003 | 0.764 ± 0.014 | 0.821 ± 0.010 | 0.855 ± 0.005 |
| `end_to_end` | 0.790 ± 0.014 | 0.857 ± 0.005 | 0.891 ± 0.005 | 0.796 ± 0.017 | 0.857 ± 0.008 | 0.890 ± 0.005 |
| `end_to_end balanced` | 0.792 ± 0.014 | 0.857 ± 0.005 | 0.893 ± 0.006 | 0.798 ± 0.016 | 0.858 ± 0.008 | 0.891 ± 0.005 |
| `supcon` | 0.815 ± 0.009 | 0.869 ± 0.011 | 0.896 ± 0.004 | 0.820 ± 0.010 | 0.870 ± 0.010 | 0.894 ± 0.003 |
| `ce_supcon` | 0.825 ± 0.012 | 0.869 ± 0.008 | 0.895 ± 0.005 | 0.825 ± 0.013 | 0.867 ± 0.008 | 0.891 ± 0.005 |

Flowers-102 (102 classes):

| Method | Linear, 4 shots | 8 shots | 16 shots | Logistic, 4 shots | 8 shots | 16 shots |
|---|---|---|---|---|---|---|
| `frozen` | 0.953 ± 0.004 | 0.978 ± 0.001 | 0.984 ± 0.001 | 0.932 ± 0.007 | 0.964 ± 0.003 | 0.978 ± 0.001 |
| `end_to_end` | 0.965 ± 0.004 | 0.984 ± 0.002 | 0.989 ± 0.001 | 0.954 ± 0.006 | 0.978 ± 0.001 | 0.986 ± 0.001 |
| `end_to_end balanced` | 0.963 ± 0.005 | 0.984 ± 0.001 | 0.990 ± 0.001 | 0.956 ± 0.005 | 0.980 ± 0.001 | 0.988 ± 0.001 |
| `supcon` | 0.972 ± 0.004 | 0.987 ± 0.001 | 0.991 ± 0.001 | 0.967 ± 0.004 | 0.984 ± 0.001 | 0.989 ± 0.001 |
| `ce_supcon` | 0.972 ± 0.004 | 0.987 ± 0.001 | 0.990 ± 0.000 | 0.962 ± 0.004 | 0.982 ± 0.002 | 0.987 ± 0.001 |

DTD (47 classes):

| Method | Linear, 4 shots | 8 shots | 16 shots | Logistic, 4 shots | 8 shots | 16 shots |
|---|---|---|---|---|---|---|
| `frozen` | 0.560 ± 0.012 | 0.623 ± 0.013 | 0.673 ± 0.009 | 0.553 ± 0.017 | 0.607 ± 0.004 | 0.654 ± 0.006 |
| `end_to_end` | 0.590 ± 0.012 | 0.658 ± 0.011 | 0.708 ± 0.013 | 0.588 ± 0.019 | 0.656 ± 0.008 | 0.700 ± 0.005 |
| `end_to_end balanced` | 0.591 ± 0.014 | 0.658 ± 0.011 | 0.709 ± 0.011 | 0.588 ± 0.018 | 0.654 ± 0.005 | 0.701 ± 0.007 |
| `supcon` | 0.602 ± 0.013 | 0.666 ± 0.009 | 0.709 ± 0.012 | 0.597 ± 0.017 | 0.654 ± 0.003 | 0.694 ± 0.004 |
| `ce_supcon` | 0.602 ± 0.013 | 0.665 ± 0.009 | 0.710 ± 0.012 | 0.598 ± 0.018 | 0.653 ± 0.003 | 0.693 ± 0.005 |

### Lift on accuracy

The training methods section of the matrix pairs `frozen` and `supcon` with logistic regression, and `ce_supcon` is scored with the head of `end_to_end`. Each cell of this table is therefore a comparison at each method's own head: the mean difference from `end_to_end`, then the seeds out of 5 in which the method beats it.

| Dataset | Method | 4 shots | 8 shots | 16 shots | Lift |
|---|---|---|---|---|---|
| Pets | `supcon` | +0.030 (5/5) | +0.013 (5/5) | +0.003 (4/5) | yes, 3 of 3 |
| Pets | `ce_supcon` | +0.034 (5/5) | +0.012 (5/5) | +0.003 (4/5) | yes, 3 of 3 |
| Flowers-102 | `supcon` | +0.002 (5/5) | −0.001 (3/5) | +0.000 (3/5) | no, 1 of 3 |
| Flowers-102 | `ce_supcon` | +0.007 (5/5) | +0.002 (5/5) | +0.001 (5/5) | yes, 3 of 3 |
| DTD | `supcon` | +0.007 (3/5) | −0.004 (2/5) | −0.014 (0/5) | no, 0 of 3 |
| DTD | `ce_supcon` | +0.012 (5/5) | +0.007 (4/5) | +0.001 (3/5) | yes, 2 of 3 |

The verdict for `supcon` changes with the head, so the same rule is also given with one head for all methods, which compares the backbones only:

| Dataset | Method | Own heads | All linear | All logistic |
|---|---|---|---|---|
| Pets | `supcon` | yes, 3 of 3 | yes, 3 of 3 | yes, 3 of 3 |
| Pets | `ce_supcon` | yes, 3 of 3 | yes, 3 of 3 | yes, 2 of 3 |
| Flowers-102 | `supcon` | no, 1 of 3 | yes, 3 of 3 | yes, 3 of 3 |
| Flowers-102 | `ce_supcon` | yes, 3 of 3 | yes, 3 of 3 | yes, 2 of 3 |
| DTD | `supcon` | no, 0 of 3 | yes, 2 of 3 | no, 1 of 3 |
| DTD | `ce_supcon` | yes, 2 of 3 | yes, 2 of 3 | no, 1 of 3 |

On DTD, both contrastive methods have lift with the linear head and not with logistic regression. At 8 and 16 shots, their linear-head accuracy is 0.012 to 0.017 above their logistic-regression accuracy. The gain there depends on the head, not only on the backbone.

### Weight interpolation

Test accuracy at each interpolation weight, at each method's own head, for 4 / 8 / 16 shots. Weight 1 is the fully trained backbone in the tables above.

| Dataset | Method | α = 0.25 | α = 0.5 | α = 0.75 |
|---|---|---|---|---|
| Pets | `end_to_end` | 0.785 / 0.846 / 0.877 | 0.793 / 0.853 / 0.885 | 0.795 / 0.855 / 0.890 |
| Pets | `supcon` | 0.811 / 0.854 / 0.882 | 0.826 / 0.867 / 0.891 | 0.827 / 0.871 / 0.894 |
| Pets | `ce_supcon` | 0.801 / 0.852 / 0.879 | 0.815 / 0.863 / 0.888 | 0.821 / 0.868 / 0.893 |
| Flowers-102 | `end_to_end` | 0.957 / 0.980 / 0.986 | 0.961 / 0.982 / 0.987 | 0.963 / 0.984 / 0.988 |
| Flowers-102 | `supcon` | 0.952 / 0.975 / 0.985 | 0.962 / 0.980 / 0.987 | 0.965 / 0.983 / 0.988 |
| Flowers-102 | `ce_supcon` | 0.962 / 0.982 / 0.986 | 0.967 / 0.985 / 0.988 | 0.970 / 0.986 / 0.990 |
| DTD | `end_to_end` | 0.574 / 0.634 / 0.684 | 0.583 / 0.645 / 0.694 | 0.588 / 0.654 / 0.702 |
| DTD | `supcon` | 0.569 / 0.625 / 0.669 | 0.584 / 0.640 / 0.682 | 0.593 / 0.646 / 0.690 |
| DTD | `ce_supcon` | 0.579 / 0.636 / 0.683 | 0.591 / 0.649 / 0.695 | 0.599 / 0.661 / 0.704 |

Validation chose a weight below 1 in three cases only: `end_to_end` (0.5) and `supcon` (0.75) on Pets at 4 shots, and `supcon` (0.75) on Flowers-102 at 8 shots. With those weights, every verdict in the own-heads column above stays the same.

## Unseen classes and novelty

Both metrics come from `benchmarks/embedding_quality.py`, and neither uses a head. Flowers-102 runs them at 4 and 8 shots only, and lift there needs both ([lift rule](../research/EXPERIMENT_MATRIX.md#lift-rule)). `frozen` does not depend on the shot count for `unseen_map`. The standard deviations are large because each seed holds out different classes, but the seeds are paired.

`unseen_map`, mean over 5 seeds:

| Dataset | Shots | `frozen` | `end_to_end` | balanced | `supcon` | `ce_supcon` |
|---|---|---|---|---|---|---|
| Pets | 4 | 0.967 | 0.975 | 0.975 | 0.976 (3/5) | 0.979 (5/5) |
| Pets | 8 | 0.967 | 0.980 | 0.980 | 0.981 (3/5) | 0.982 (4/5) |
| Pets | 16 | 0.967 | 0.982 | 0.982 | 0.985 (4/5) | 0.983 (3/5) |
| Flowers-102 | 4 | 0.978 | 0.983 | 0.984 | 0.988 (5/5) | 0.988 (5/5) |
| Flowers-102 | 8 | 0.978 | 0.983 | 0.984 | 0.987 (5/5) | 0.986 (5/5) |
| DTD | 4 | 0.776 | 0.801 | 0.802 | 0.808 (4/5) | 0.808 (4/5) |
| DTD | 8 | 0.776 | 0.815 | 0.815 | 0.813 (3/5) | 0.813 (3/5) |
| DTD | 16 | 0.776 | 0.813 | 0.812 | 0.821 (5/5) | 0.820 (4/5) |

`novelty_heldout_pr_auc`, mean over 5 seeds:

| Dataset | Shots | `frozen` | `end_to_end` | balanced | `supcon` | `ce_supcon` |
|---|---|---|---|---|---|---|
| Pets | 4 | 0.150 | 0.138 | 0.137 | 0.151 (4/5) | 0.179 (5/5) |
| Pets | 8 | 0.158 | 0.176 | 0.176 | 0.210 (4/5) | 0.208 (4/5) |
| Pets | 16 | 0.161 | 0.199 | 0.201 | 0.240 (5/5) | 0.224 (5/5) |
| Flowers-102 | 4 | 0.261 | 0.288 | 0.291 | 0.370 (5/5) | 0.379 (5/5) |
| Flowers-102 | 8 | 0.326 | 0.371 | 0.387 | 0.476 (5/5) | 0.470 (5/5) |
| DTD | 4 | 0.128 | 0.155 | 0.154 | 0.148 (1/5) | 0.148 (1/5) |
| DTD | 8 | 0.124 | 0.145 | 0.143 | 0.147 (3/5) | 0.147 (3/5) |
| DTD | 16 | 0.138 | 0.175 | 0.179 | 0.169 (2/5) | 0.169 (2/5) |

The bracket is the number of seeds out of 5 in which the method beats `end_to_end`.

| Dataset | Metric | `supcon` | `ce_supcon` |
|---|---|---|---|
| Pets | `unseen_map` | no, 1 of 3 | yes, 2 of 3 |
| Pets | `novelty_heldout_pr_auc` | yes, 3 of 3 | yes, 3 of 3 |
| Flowers-102 | `unseen_map` | yes, 2 of 2 | yes, 2 of 2 |
| Flowers-102 | `novelty_heldout_pr_auc` | yes, 2 of 2 | yes, 2 of 2 |
| DTD | `unseen_map` | yes, 2 of 3 | yes, 2 of 3 |
| DTD | `novelty_heldout_pr_auc` | no, 0 of 3 | no, 0 of 3 |

## Rare positives

`knn_pr_auc` on the `rare_positive` task: the training set has `shots` positives from one class and 200 negatives from the others, with the labels `positive` and `negative`, and the score is the maximum cosine similarity to the training positives. Each seed draws a different positive class, so the standard deviations are large and depend on that class.

| Dataset | Shots | `frozen` | `end_to_end` | balanced | `supcon` | `ce_supcon` |
|---|---|---|---|---|---|---|
| Pets | 4 | 0.708 ± 0.174 | **0.870 ± 0.067** | 0.869 ± 0.071 | 0.402 ± 0.136 | 0.535 ± 0.077 |
| Pets | 8 | 0.828 ± 0.095 | **0.942 ± 0.034** | 0.938 ± 0.033 | 0.619 ± 0.136 | 0.719 ± 0.064 |
| Pets | 16 | 0.831 ± 0.080 | 0.969 ± 0.022 | **0.970 ± 0.023** | 0.782 ± 0.066 | 0.801 ± 0.018 |
| Flowers-102 | 4 | 0.832 ± 0.227 | **0.910 ± 0.118** | **0.910 ± 0.107** | 0.707 ± 0.285 | 0.718 ± 0.226 |
| Flowers-102 | 8 | 0.916 ± 0.150 | 0.961 ± 0.082 | **0.977 ± 0.035** | 0.789 ± 0.192 | 0.848 ± 0.204 |
| Flowers-102 | 16 | 0.929 ± 0.125 | 0.977 ± 0.044 | **0.988 ± 0.020** | 0.865 ± 0.178 | 0.877 ± 0.214 |
| DTD | 4 | 0.455 ± 0.219 | **0.570 ± 0.119** | 0.559 ± 0.103 | 0.540 ± 0.162 | 0.546 ± 0.159 |
| DTD | 8 | 0.544 ± 0.234 | **0.678 ± 0.138** | 0.669 ± 0.140 | 0.586 ± 0.252 | 0.593 ± 0.249 |
| DTD | 16 | 0.599 ± 0.200 | 0.730 ± 0.137 | **0.731 ± 0.142** | 0.631 ± 0.170 | 0.640 ± 0.171 |

Neither contrastive method has lift at any shot count. Against `end_to_end` they win 0 of 5 seeds on Pets and Flowers-102, and 1 or 2 of 5 on DTD. On Pets and Flowers-102 they also score below `frozen` at every shot count, so the training removed information that the pretrained embeddings had. On DTD they stay above `frozen`.

The head-based `pr_auc`, a secondary metric, has the same pattern on Pets and Flowers-102. On DTD at 4 shots it is the exception: `end_to_end` (0.530) is below `frozen` (0.568), and `ce_supcon` (0.554) is above `end_to_end`.

### Step budget and contrastive weight

A diagnostic on Pets varies the step budget and the contrastive weight, on the validation split, with seeds 0 and 1 only. The test split was not used. Rates are fixed at 1e-6 for `end_to_end` and `supcon`, and at 3e-7 for `ce_supcon`, so the `ce_supcon` rows do not differ from `end_to_end` by the contrastive term alone. Other settings are as in [Setup](#setup).

`knn_pr_auc` at 4 / 16 shots, with `frozen` at 0.864 / 0.933:

| Method | 50 steps | 150 steps | 500 steps |
|---|---|---|---|
| `end_to_end` | 0.919 / 0.973 | 0.909 / 0.992 | 0.903 / 0.997 |
| `ce_supcon`, weight 0.1 | 0.886 / 0.952 | 0.834 / 0.940 | 0.653 / 0.878 |
| `ce_supcon`, weight 1 | 0.849 / 0.939 | 0.779 / 0.905 | 0.630 / 0.806 |
| `ce_supcon`, weight 10 | 0.831 / 0.927 | 0.684 / 0.871 | 0.603 / 0.789 |
| `supcon` | 0.620 / 0.856 | 0.612 / 0.804 | 0.412 / 0.831 |

- `end_to_end` stays above `frozen` at every step budget.
- `ce_supcon` falls as the step budget grows, at every weight, and falls as the weight grows, at every step budget. Only weight 0.1 at 50 steps is above `frozen` at both shot counts, by about 0.02. It stays below `end_to_end`.
- The loss from the step budget is larger than the loss from the weight. Weight 0.1 at 500 steps (0.653) is below weight 10 at 50 steps (0.831).
- `supcon` is already far below `frozen` at 50 steps. Its 16-shot row is not monotonic.
- Two seeds is a small sample for a metric whose standard deviation over seeds reaches 0.17 on the Pets test split. The ordering of the rows is more reliable than any single difference.

**Hypothesis, not tested by these runs.** The pattern would follow if cross-entropy on a few training images saturates quickly while the SupCon term on augmented views does not, so that the contrastive term dominates late in a long run at any weight. With two labels, SupCon pulls all negatives together, whatever their true class, which would remove the differences between the negative classes that rare-positive ranking needs. The runs show the dependence on steps and weight. They do not record the two loss terms separately or measure the spread within the negatives, so they do not show the cause.

## Sampler control

`end_to_end balanced` is `end_to_end` with the class-balanced batches of `supcon` and `ce_supcon`, tuned over the same rates. Its accuracy is within 0.002 of `end_to_end` in all nine cells at the linear head, and within 0.003 at logistic regression. So the accuracy gain does not come from the batches. On Pets and DTD its `unseen_map`, `novelty_heldout_pr_auc`, and `knn_pr_auc` are close to `end_to_end` as well.

Flowers-102 differs slightly: `knn_pr_auc` rises by about 0.02 at 8 shots and 0.01 at 16 shots, and `novelty_heldout_pr_auc` by about 0.02 at 8 shots. Tuning chose different rates for the two runs there, 1e-6 with balanced batches and 3e-7 without, so these differences mix the batches and the rate. They are in the direction opposite to the `knn_pr_auc` loss of the contrastive methods, so balanced batches do not explain that loss.

## Setup

| | |
|---|---|
| Datasets | Oxford-IIIT Pets (37 classes), Oxford Flowers-102 (102), and DTD (47), with the splits and pool sizes of the [benchmarking guide](../guides/benchmarking.md#datasets): pool up to 100 and evaluation up to 100 images per class |
| Backbone | `hf/facebook/dinov2-small`, `--image-size 112`, images shrunk once to a shorter side of 224 px |
| Shots and seeds | 4, 8, and 16 per class; seeds 0 to 4 on the test split, paired across methods |
| Budget | 500 steps, batch size 32, `natural` augmentation, every method |
| Contrastive | Temperature 0.07, class-balanced batches, one view |
| `open_set` | 3 held-out classes; on Flowers-102, 4 and 8 shots only |
| `rare_positive` | 200 negatives |
| Hardware | One NVIDIA TITAN V with cuDNN disabled, shared with other jobs |
| Code | Commit `f6e41ab` for Pets and `3a418ed` for Flowers-102, DTD, and the diagnostic; the second adds input checks, not behaviour used by these runs |

### Tuning

Rates and the contrastive weight were chosen on the validation split, before the test runs, over seeds 0 and 1 and all three shot counts, by mean accuracy of the fully trained model at each method's own head. `end_to_end` and `supcon` reuse an earlier sweep with the same settings.

| Dataset | `end_to_end` | balanced | `supcon` | `ce_supcon` rate | `ce_supcon` weight |
|---|---|---|---|---|---|
| Pets | 1e-6 | 1e-6 | 1e-6 | 3e-7 | 10 |
| Flowers-102 | 3e-7 | 1e-6 | 1e-6 | 3e-7 | 10 |
| DTD | 1e-6 | 1e-6 | 3e-7 | 3e-7 | 10 |

Mean validation accuracy for `ce_supcon`:

| Dataset | Weight | 3e-7 | 1e-6 | 3e-6 | 1e-5 |
|---|---|---|---|---|---|
| Pets | 0.1 | 0.8669 | 0.8662 | 0.8536 | 0.8270 |
| Pets | 1 | 0.8716 | 0.8707 | 0.8579 | 0.8295 |
| Pets | 10 | **0.8732** | 0.8716 | 0.8552 | 0.8264 |
| Flowers-102 | 0.1 | 0.9773 | 0.9773 | 0.9753 | 0.9745 |
| Flowers-102 | 1 | 0.9796 | 0.9796 | 0.9775 | 0.9739 |
| Flowers-102 | 10 | **0.9810** | 0.9802 | 0.9786 | 0.9755 |
| DTD | 0.1 | 0.6343 | 0.6408 | 0.6319 | 0.6240 |
| DTD | 1 | 0.6423 | 0.6374 | 0.6299 | 0.6170 |
| DTD | 10 | **0.6430** | 0.6382 | 0.6202 | 0.6012 |

The chosen `ce_supcon` setting is the same corner of the grid on all three datasets: the lowest rate and the highest weight. Better settings may lie outside the grid. At the chosen rate, weight 1 is within 0.002 of weight 10 on all three datasets.

## RVL-CDIP

Not included: the suite's RVL-CDIP run at 112 px has not finished, so this page has no results for it. [RVL-CDIP with DINOv2-small](rvl-cdip-dinov2-small.md) is a separate benchmark at 224 px with another budget, and its numbers are not comparable with the tables here.

## Limits

- Three datasets, one small backbone, one input size, five seeds. At 112 px the fine detail of Flowers-102 and DTD is reduced for every method.
- Lift has no effect-size floor (see [Result](#result)). The Pets accuracy gain at 4 shots is large against its standard deviation. On Flowers-102, the accuracy gains of `ce_supcon` are 0.007 or less, on scores already near 0.98.
- Settings were chosen on accuracy, for all tasks. The embedding results therefore describe those settings, not the best a contrastive method can do on each task. The diagnostic shows that the rare-positive loss depends on them.
- The `ce_supcon` setting is at the edge of the tuning grid on every dataset.
- `benchmarks/embedding_quality.py` scores only each method's own head, so head-based `pr_auc` is not available with one head for all methods. The primary embedding metrics use no head.
- The diagnostic covers one dataset, two seeds, and the validation split, at fixed rates tuned for classification.
- Validation for `end_to_end` and `supcon` comes from an earlier sweep, whose results record no code commit.

## Cost

The test-split runs took 473 minutes for Pets, 639 for Flowers-102, and 407 for DTD, including the control and the embedding metrics. The GPU and the machine were shared with other jobs, so these are rough figures.

## Reproducing

```bash
pip install -e ".[hf,datasets]"
# From the repository root. Shared settings:
COMMON="--dataset pets --data-root .cache --backbone hf/facebook/dinov2-small --image-size 112 \
  --max-image-side 224 --shots 4 8 16 --augmentation natural --max-steps 500 --temperature 0.07"
# Tune on validation, one command per rate and weight:
python benchmarks/few_shot.py $COMMON --evaluate-on validation --seeds 0 1 --interpolate 0.25 0.5 0.75 \
    --methods ce_supcon --learning-rate ce_supcon=3e-7 --contrastive-weight 10 --output .runs/ces-tune.json
# Then run the test split once with the chosen settings:
python benchmarks/few_shot.py $COMMON --evaluate-on test --seeds 0 1 2 3 4 --interpolate 0.25 0.5 0.75 \
    --heads logistic_regression linear --methods frozen end_to_end supcon ce_supcon --contrastive-weight 10 \
    --learning-rate end_to_end=1e-6 supcon=1e-6 ce_supcon=3e-7 --output .runs/ces-test.json
python benchmarks/embedding_quality.py $COMMON --evaluate-on test --seeds 0 1 2 3 4 \
    --tasks open_set rare_positive --methods frozen end_to_end supcon ce_supcon --contrastive-weight 10 \
    --learning-rate end_to_end=1e-6 supcon=1e-6 ce_supcon=3e-7 --output .runs/ces-embedding.json
# The control:
python benchmarks/few_shot.py $COMMON --evaluate-on test --seeds 0 1 2 3 4 --interpolate 0.25 0.5 0.75 \
    --heads logistic_regression linear --methods end_to_end --sampler end_to_end=balanced \
    --learning-rate end_to_end=1e-6 --output .runs/ces-balanced.json
```

The scripts download the data and the weights into the git-ignored `.cache/` folder on first use. On Flowers-102, run `open_set` with `--shots 4 8` and `rare_positive` with all three shot counts, as separate commands.
