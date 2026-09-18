# Evaluation Design

## Why rare-event evaluation differs

In deposit-risk scenarios, a model can have excellent accuracy while capturing little fraud.

Prioritize:

- PR-AUC
- recall at operational FPR
- precision / recall at review capacity
- dollar-weighted capture

## Recall at fixed FPR

Given risk scores and binary truth, select the score threshold that meets the target false-positive rate according to a documented interpolation/tie policy, then compute recall.

Tie behavior must be deterministic and tested.

## Review-rate metrics

If the operation can review 1% of deposits:

```text
sort by risk descending
take top ceil(N * 0.01)
compute precision / recall
```

Define rounding policy centrally.

## Dollar recall

```text
captured fraud amount / total fraud amount
```

Only fraudulent examples contribute to numerator/denominator.

Reject negative amounts unless explicitly supported.

## Splits

Recommended order of realism:

1. random stratified — debugging baseline
2. grouped — prevents direct entity/item leakage
3. temporal out-of-time — best proxy for future deployment when available

## Model selection

Validation set:

- early stopping
- checkpoint choice
- threshold selection

Test set:

- final evaluation only
