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

### Implemented behaviour (`embedfit.losses.SupConLoss`)

For anchor `i` with positives `P(i)` (other samples of the same class) and all other samples `A(i)`:

```text
L_i = -1/|P(i)| · Σ_{p ∈ P(i)} log( exp(z_i·z_p / τ) / Σ_{a ∈ A(i)} exp(z_i·z_a / τ) )
```

The loss is the mean of `L_i` over anchors that have at least one positive. An anchor without a positive still serves as a negative for the others but contributes no term. A batch in which no sample has a positive raises `ConfigurationError`, because returning zero would hide a sampler misconfiguration.

Embeddings are L2-normalised inside the loss by default, so the result does not depend on embedding scale. The anchor's self-similarity is masked to −∞, which removes it from the comparison. The loss goes through `log_softmax`, which keeps it and its gradients finite for near-duplicate vectors and very small temperatures. Half-precision inputs are raised to float32 first; float32 and float64 inputs keep their precision.

`[B, V, D]` input is flattened to `[B·V, D]` with labels repeated per view, so views of one sample are positives of each other.

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

### Implemented behaviour (`embedfit.sampling.BalancedClassBatchSampler`)

- Classes are drawn without replacement from a shuffled queue that refills when empty, so classes are visited equally often even when only some fit in a batch.
- Within a class, samples are drawn the same way: every sample of a class is used before any is repeated.
- A class with fewer than `samples_per_class` samples contributes all of its samples and no duplicates, because a duplicated image is a trivial positive pair. Every class needs at least two samples.
- `samples_per_class` must be at least 2.
- An epoch has `ceil(N / (classes_per_batch × samples_per_class))` batches unless `batches_per_epoch` is given: about `N` draws, which on imbalanced data is not one pass, since samples of rare classes recur and samples of frequent classes are skipped.
- Batches are a function of `seed` and the epoch set with `set_epoch`; the global random state is not used.

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

### Implemented behaviour (`embedfit.sampling.PairSampler`, `embedfit.losses.CosinePairLoss`)

`ImagePair` holds two dataset positions rather than images, so a pair list stays small; `weight` and `relationship` record what a rule decided.

For each anchor the sampler builds a positive pool (same class) and a negative pool (other classes), lets the relationship rules move samples between the pools, and draws up to `positives_per_example` and `negatives_per_example` partners. Pools are computed one anchor at a time, so memory stays `O(N)`; the full `N²` pair set is never built. A class with few samples simply yields fewer positive pairs: partners are never repeated to fill a quota. Partners already paired with the anchor earlier in the epoch are skipped first, so the quota goes to new pairs.

`CosinePairLoss` is the weighted mean of `(cos(a, b) − goal)²` with `goal = 1` for positive pairs and `negative_target` (default 0) for negative pairs. Half-precision inputs are computed in float32.

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
