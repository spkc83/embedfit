# Rare-Event Risk Workflows

This guide is for problems where the class of interest is rare and costly: a small fraction of items is risky, labels arrive late, and only a limited number of items can be reviewed. Rare-positive review of document images is the reference case ([domain notes](../domain/RARE_POSITIVE_REVIEW_REFERENCE.md)), but nothing below is specific to it, and EmbedFit's core contains no domain logic (ADR 0005).

EmbedFit produces scores, embeddings, and evaluation numbers. Decisions, business rules, and case handling sit outside it.

[`examples/05_synthetic_document_risk.py`](../../examples/05_synthetic_document_risk.py) runs every step below on generated images.

## 1. Describe each sample fully

```python
ImageSample(image=path, label="risk", sample_id="...", group_id="...",
            timestamp=observed_at, amount=1250.00)
```

- `group_id` links samples that must not be separated: scans of one physical item, items from one source or template.
- `timestamp` is when the item was observed, not when its label became known.
- `amount` is only used by value-weighted metrics. Metadata is never a model feature.

Keep anything unavailable at decision time out of the image and out of training: outcome codes, investigation notes, later events.

## 2. Split before anything else

Hold out the most recent period with `temporal_split`, then divide the rest with `grouped_split` into training and validation data, and confirm the result with `find_split_overlap` ([leakage guide](leakage.md)). With rare positives, check that each split actually contains some: both split functions log a warning when a class is missing from one side.

## 3. State the positive class

```python
model = EmbedFitModel.from_pretrained(backbone, labels=["routine", "risk"], positive_label="risk")
```

The positive class is never inferred from label order. `risk_score` returns the probability of `positive_label`, and every ranking metric takes `positive_label` explicitly.

## 4. Adapt the representation, then fit a class-weighted head

`supcon` training samples classes evenly, so with two classes the rare positives are in every batch. `pair` training shuffles its pairs instead; on imbalanced data rare-class samples still take part in many more pairs than others ([losses and sampling](../design/LOSSES_AND_SAMPLING.md)). The head is fitted on the natural class distribution, where `LogisticRegressionHead(class_weight="balanced")` keeps the rare class from being ignored. Pass validation data as `eval_dataset` so that the best epoch is chosen on it; never pass test data there.

Few positives make everything noisy. With a handful of positive examples, compare against the frozen baseline over several seeds before trusting an improvement ([benchmarking guide](benchmarking.md)).

## 5. Measure what review capacity buys

Accuracy is close to the negative rate for any model, including one that flags nothing. Use instead:

| Question | Metric |
| --- | --- |
| How good is the ranking overall? | `pr_auc` (equals the positive rate for a random ranking) |
| What is caught at a tolerable false-alarm rate? | `recall_at_fpr(target_fpr=...)` |
| What does reviewing the top 1% yield? | `precision_at_review_rate`, `recall_at_review_rate` |
| How much of the value at risk is caught? | `dollar_recall_at_review_rate`, `dollar_recall` |
| Can the scores be read as probabilities? | `brier_score`, `calibration_bins`, `expected_calibration_error` |

A model can catch most risky items and still miss most of the value, when the few missed items are the large ones; report count-based and value-weighted capture side by side. Metrics that are undefined for a split, such as recall without any positives, raise `UndefinedMetricError` rather than returning a number.

## 6. Choose the threshold on validation data

```python
chosen = select_threshold(val_labels, val_scores, positive_label="risk",
                          objective="target_fpr", target=0.01)
flagged = test_scores >= chosen.threshold
```

Apply the threshold to the test set unchanged and report what it achieves there. A threshold tuned on the test set makes every reported number optimistic. Expect the realised false-positive rate on new data to differ from the target, more so when validation negatives are few.

## 7. Use the embeddings for context

The same embeddings support retrieval: index past items with `RetrievalIndex` and, for a flagged item, retrieve its nearest neighbours with their labels. When evaluating retrieval on indexed items, use `exclude_self` and `exclude_same_group` so that an item, or another scan of it, does not count as evidence.

## 8. Handle sensitive data deliberately

Images and metadata may contain personal and other sensitive information. EmbedFit keeps image content, metadata, and sample identifiers out of its logs and exception messages, and saved models contain no training samples ([security and privacy requirements](../../SECURITY_PRIVACY.md)). Class label names and the ids of retrieval results do appear in messages, reprs, and saved models, so do not encode sensitive values in them. What is done with the scores and embeddings afterwards is the caller's responsibility: embeddings of sensitive images are themselves sensitive.
