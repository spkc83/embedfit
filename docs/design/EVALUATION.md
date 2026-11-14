# Evaluation Design

## Why rare-event evaluation differs

In rare-positive review scenarios, a model can have excellent accuracy while capturing few positives.

Prioritize:

- PR-AUC
- recall at operational FPR
- precision / recall at review capacity
- value-weighted capture

## Recall at fixed FPR

Given risk scores and binary truth, `recall_at_fpr` takes the lowest observed score threshold whose false-positive rate is within the target, without interpolation, and reports recall at it.

Tied scores are handled deterministically; see [Conventions](#conventions-implemented-in-embedfitevaluation).

## Review-rate metrics

If the operation can review 1% of items:

```text
sort by risk descending
take top ceil(N * 0.01)
compute precision / recall
```

The rounding is defined once, in `review_count`, which every review-rate metric and `select_threshold` use; see [Conventions](#conventions-implemented-in-embedfitevaluation).

## Value-weighted recall (`dollar_recall`)

```text
captured positive amount / total positive amount
```

Only positive examples contribute to numerator/denominator.

Negative amounts are rejected with `ConfigurationError`.

## Conventions implemented in `embedfit.evaluation`

Metric functions take arrays of labels and scores and never call a model.

**Positive class.** Every binary metric requires `positive_label`. A sample is positive only when its label equals `positive_label` in both value and type, so `1`, `1.0`, and `True` are distinct. Labels may be a list, a 1-D numpy array, or a 1-D torch tensor of scalar labels; a string, a 2-D array, or array-valued elements raise `ConfigurationError`, because they would otherwise match nothing and be scored as all-negative. A type mismatch between `positive_label` and well-formed labels leaves no positives, which the recall-type metrics report as undefined.

**Score direction.** Higher scores mean "more likely positive". The ranking metrics (`roc_auc`, `pr_auc`, `recall_at_fpr`, and the top-K and review-rate metrics) accept distance-like scores with `greater_is_riskier=False`. `select_threshold` requires higher-is-riskier scores, so that the returned threshold has one meaning (`score >= threshold` flags); negate distance-like scores before calling it. Calibration functions take probabilities of the positive class and have no direction.

**Thresholds and ties.** A threshold flags every item with `score >= threshold`, so tied items are always flagged together. Candidate thresholds are the observed score values; nothing is interpolated.

- `recall_at_fpr` uses the lowest threshold whose FPR is within the target. When even the top tied group exceeds the target, nothing is flagged and recall is 0.
- Top-K metrics (`precision_at_k`, `recall_at_k`, the review-rate metrics, `dollar_recall_at_review_rate`) treat tied items as interchangeable and report the expected value under random tie-breaking: a tied group of `g` items of which `m` fit under the cut-off contributes `m / g` of its positives (or positive amount). Results never depend on input order.

**Review-rate rounding.** `review_rate` lies in (0, 1] and reviews `max(1, ceil(N × review_rate))` items. The product is rounded to 9 decimals before the ceiling so that `100 × 0.07` reviews 7 items, not 8; a genuine fractional part below 5e-10 is ignored as a result, which only matters beyond roughly a billion samples. The count never exceeds `N`.

**Undefined values.** A metric with no defined value raises `UndefinedMetricError`: recall-type metrics and PR-AUC without positives, ROC-AUC and recall-at-FPR without positives or without negatives, dollar recall when positives have zero total amount, and any metric on an empty sample. Precision at K and the review-rate precision stay defined (0) without positives, and the calibration functions do not need positives at all. `Trainer.evaluate` converts these errors to NaN and logs a warning, so that one degenerate split does not abort a benchmark.

**Amounts.** Amounts must be finite and non-negative; amounts of negative samples are ignored.

## Metric registry

`Trainer.evaluate(metrics=[...])` and `FewShotEvaluator.run(metrics=[...])` add registered metrics to the default results. A registered metric is a callable `metric(y_true, scores, *, positive_label) -> float`, where `scores` is the predicted probability of `positive_label`. Every ranking function in `embedfit.evaluation` except the dollar metrics has this form, and so do `brier_score` and `expected_calibration_error` (`calibration_bins` returns bins, not a number), so a parameterised one registers with `functools.partial`:

```python
from functools import partial
from embedfit.evaluation import recall_at_fpr, register_metric

register_metric("recall_at_fpr_1pct", partial(recall_at_fpr, target_fpr=0.01))
trainer.evaluate(test_dataset, metrics=["recall_at_fpr_1pct"])
```

- `roc_auc`, `pr_auc`, and `brier_score` are registered built-ins and are computed by default whenever the model has a `positive_label`. They, and every name `classification_metrics` returns, are reserved: they cannot be registered or overwritten, so a reported `roc_auc` is always ROC-AUC.
- Requested metrics need a model with a `positive_label`. Unknown, repeated, or default names, and requests without a `positive_label`, raise `ConfigurationError` before anything is predicted (in `FewShotEvaluator.run`, before any model is trained).
- A requested metric that raises `UndefinedMetricError` is reported as NaN with a logged warning, like the built-ins.
- Metrics over hard predictions or the full probability matrix (multiclass metrics such as top-k accuracy) and the dollar metrics, which need amounts, cannot be registered in v0.1; compute them from `Trainer.predict`.

**PR-AUC** is average precision, the uninterpolated step sum `Σ (Rₙ − Rₙ₋₁) Pₙ`, which equals the positive rate for a random ranking.

**Macro averages** run over every declared class, scoring an absent class as 0, so results are comparable across samples; `balanced_accuracy` averages recall over the classes present in `y_true`.

**Threshold selection.** `select_threshold` supports `max_f1` (ties go to the higher threshold), `target_precision` (highest recall subject to the precision floor, ties going to the higher threshold; raises if unreachable), `target_recall` (highest qualifying threshold), `target_fpr`, and `review_rate` (both: lowest qualifying threshold). When a target can only be met by flagging nothing, the threshold is `inf`. It must only ever see validation data.

## Splits

Recommended order of realism:

1. random stratified — debugging baseline
2. grouped — prevents direct entity/item leakage
3. temporal out-of-time — best proxy for future deployment when available

The corresponding helpers and the overlap check are described in the [leakage guide](../guides/leakage.md).

## Model selection

Validation set:

- early stopping
- checkpoint choice
- threshold selection

Test set:

- final evaluation only
