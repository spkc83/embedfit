# Losses and Sampling

## Supervised Contrastive Loss

Primary v0.1 representation objective.

Requirements:

- accepts `[B, D]`;
- supports optional `[B, V, D]` multi-view representation;
- normalized embeddings by default;
- configurable temperature;
- excludes anchor self-comparison;
- stable numerics;
- clear behavior when an anchor has no positive in batch.

The balanced batch sampler should make missing positives uncommon.

## Balanced class batches

Config concept:

```python
BalancedClassBatchSampler(
    classes_per_batch=4,
    samples_per_class=8,
    seed=42,
)
```

Batch size derives from product when this sampler is active.

## Pairwise loss

Represent pair:

```python
@dataclass
class ImagePair:
    left: Any
    right: Any
    target: int  # 1 similar, 0 dissimilar
    weight: float = 1.0
    relationship: str | None = None
```

Use bounded pair generation.

## Pair sampling defaults

A reasonable initial default:

```text
positives_per_example = 4
negatives_per_example = 4
```

Cap if dataset structure makes this impossible.

## Hard negatives

Later milestone:

- encode train split without gradients;
- nearest neighbor search;
- choose different-class nearest items;
- refresh periodically;
- never inspect validation/test data.
