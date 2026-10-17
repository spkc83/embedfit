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
→ linear head fitted on frozen embeddings (linear probe)
→ cross-entropy fine-tuning of head and backbone together
```

Starting from a converged probe (LP-FT, ADR 0012) keeps the baseline from being trained through a random classifier at a learning rate chosen for the backbone.

The whole backbone is unfrozen for fine-tuning; partial unfreezing is not available.

## Baseline 3 — Nearest centroid

Optional but useful as an embedding diagnostic. It is not built in: a nearest-centroid classifier can be written as a `ClassificationHead` and registered with `register_head`, which makes it selectable through `TrainingArguments.head`.

## Interpretation rule

Do not claim the SetFit-style procedure provides sample-efficiency unless it improves against these baselines under identical few-shot splits and repeated seeds.
