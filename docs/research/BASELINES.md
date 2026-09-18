# Required Baselines

Complex metric-learning methods are not meaningful without strong baselines.

## Baseline 1 — Frozen features + LR

Required for every experiment.

```text
pretrained encoder
→ frozen embeddings
→ logistic regression
```

## Baseline 2 — End-to-end CE

```text
pretrained encoder
→ linear head
→ cross-entropy fine-tuning
```

Allow either full or explicitly configured partial unfreezing.

## Baseline 3 — Nearest centroid

Optional but useful as an embedding diagnostic.

## Interpretation rule

Do not claim the SetFit-style procedure provides sample-efficiency unless it improves against these baselines under identical few-shot splits and repeated seeds.
