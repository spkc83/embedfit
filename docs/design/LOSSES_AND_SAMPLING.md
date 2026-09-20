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

For each anchor the sampler builds a positive pool (same class) and a negative pool (other classes), lets the relationship rules move samples between the pools, and draws up to `positives_per_example` and `negatives_per_example` partners. Pools are computed one anchor at a time, so memory stays `O(N)` and the full `N²` pair set is never built; time is `O(N²)` per epoch in vectorised array operations (a few seconds at 10 000 samples). A class with few samples simply yields fewer positive pairs: partners are never repeated to fill a quota. A partner already paired with the anchor earlier in the epoch is excluded, so an anchor can finish below its quota and the pair count varies slightly from epoch to epoch.

Quotas are per anchor, so the positive/negative split is only roughly even. On imbalanced data a small class runs out of positive partners while its samples are drawn often as negatives for the larger classes, so rare-class samples take part in many more pairs than others.

A rule's `field` is looked up as a sample field first and as a `metadata` key second. A missing value matches neither `same` nor `different`, and a rule whose field has no value on any sample is rejected as a likely typo.

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

### Implemented behaviour (`PairSampler.sample(..., embeddings=, hard_negative_fraction=)`)

Mining changes only *which* negatives an anchor is paired with. For each anchor, `round(negatives_per_example × hard_negative_fraction)` of its negative partners are the members of its negative pool with the highest cosine similarity to it under the current backbone (ties by dataset position); the rest of the quota, and all positives, stay random. A mix is the default recommendation because training on hardest negatives alone is unstable and keeps returning the same few pairs. The negative pool is the one the class labels and relationship rules define, so mining never turns a positive into a negative.

The sampler can only rank its own samples, and `Trainer` builds it from `train_dataset`, so validation and test data cannot be mined. During training the embeddings come from `model.encode` on the training images, which runs without gradients in eval mode and restores training mode, and they are refreshed every `hard_negative_refresh_epochs` epochs. Validation pairs are always random, which keeps the validation loss comparable across epochs and between runs with and without mining.

Mined pairs depend on the weights at the time, so they cannot be drawn ahead of training. The learning-rate schedule is therefore planned on the largest number of pairs an epoch can hold, and the rate ends slightly above zero.
