# Losses and Sampling

## Supervised Contrastive Loss

Primary v0.1 representation objective.

Properties:

- accepts `[B, D]`, and `[B, V, D]` for multiple views;
- normalises embeddings by default (`normalize=True`);
- configurable temperature (`temperature`, default 0.07);
- excludes anchor self-comparison;
- stable numerics;
- defined behaviour when an anchor has no positive in its batch.

With `BalancedClassBatchSampler`, every class in a batch contributes at least two samples, so every anchor has a positive.

### Implemented behaviour (`embedfit.losses.SupConLoss`)

For anchor `i` with positives `P(i)` (other samples of the same class) and all other samples `A(i)`:

```text
L_i = -1/|P(i)| · Σ_{p ∈ P(i)} log( exp(z_i·z_p / τ) / Σ_{a ∈ A(i)} exp(z_i·z_a / τ) )
```

The loss is the mean of `L_i` over anchors that have at least one positive. An anchor without a positive still serves as a negative for the others but contributes no term. A batch in which no sample has a positive raises `ConfigurationError`, because returning zero would hide a sampler misconfiguration.

Embeddings are L2-normalised inside the loss by default, so the result does not depend on embedding scale. The anchor's self-similarity is masked to −∞, which removes it from the comparison. The loss goes through `log_softmax`, which keeps it and its gradients finite for near-duplicate vectors and very small temperatures. Half-precision inputs are raised to float32 first; float32 and float64 inputs keep their precision.

`[B, V, D]` input is flattened to `[B·V, D]` with labels repeated per view, so views of one sample are positives of each other. `supcon` training passes `[B, V, D]` when `TrainingArguments.contrastive_views` is above 1 (ADR 0018).

## Balanced class batches

With the default `balanced` sampler, `Trainer` derives the batch shape from `TrainingArguments.batch_size` with `batch_shape`:

```python
classes_per_batch = min(num_classes, batch_size // 2)
samples_per_class = batch_size // classes_per_batch
```

`num_classes` counts the classes present in the training data, and `batch_size` must be at least 4 (two classes with two samples each). The realised batch can be smaller than `batch_size`: the division can leave a remainder, and a class with fewer than `samples_per_class` samples contributes all it has. The sampler's seed is `TrainingArguments.seed`.

### Implemented behaviour (`embedfit.sampling.BalancedClassBatchSampler`)

- Classes are drawn without replacement from a shuffled queue that refills when empty, so classes are visited equally often even when only some fit in a batch.
- Within a class, samples are drawn the same way: every sample of a class is used before any is repeated.
- A class with fewer than `samples_per_class` samples contributes all of its samples and no duplicates, because a duplicated image is a trivial positive pair. Every class needs at least two samples.
- `samples_per_class` must be at least 2.
- An epoch has `ceil(N / (classes_per_batch × samples_per_class))` batches unless `batches_per_epoch` is given: about `N` draws, which on imbalanced data is not one pass, since samples of rare classes recur and samples of frequent classes are skipped.
- Batches are a function of `seed` and the epoch set with `set_epoch`; the global random state is not used.
- With `TrainingArguments.augmentation`, each draw is augmented with a seed of its own, so a rare-class sample that recurs within an epoch gets a different view each time (ADR 0014).

## Batch-sampler registry

`supcon` and `end_to_end` training draw their batches from a registered batch sampler, selected with `TrainingArguments.sampler`. `None` selects the mode's default: `"balanced"` (`BalancedClassBatchSampler` shaped by `batch_shape`) for `supcon`, `"random"` (uniform shuffling, reshuffled every epoch from the seed) for `end_to_end`. The option is rejected in `frozen` and `pair` mode; `pair` batches come from `PairSampler`, which is not pluggable.

```python
from embedfit.sampling import register_sampler

register_sampler("my_sampler", lambda targets, *, batch_size, seed: MySampler(targets, batch_size, seed))
args = TrainingArguments(training_mode="supcon", sampler="my_sampler")
```

The factory is called as `factory(targets, *, batch_size, seed)`, where `targets` is the integer class index of every sample in dataset order; it never sees images or metadata. It returns a `BatchSampler`: an object with `__iter__` yielding lists of dataset positions, `__len__` equal to the number of batches one iteration yields, and `set_epoch(epoch)`, which training calls before every epoch. Options of a custom sampler go in a closure or `functools.partial`.

The same sampler, with a separate seed, serves the validation batches: validation always uses the batches of epoch 0. Batches must depend only on the seed and the epoch, because the `Trainer` checks them before training: it iterates a separate instance for every training epoch (with `max_steps`, every epoch needed to take that many steps; ADR 0017) and epoch 0 for validation on positions only, and rejects an epoch with no batches, an empty batch, a position outside the data, a sampler whose length differs from its batch count or from one epoch to the next (the learning-rate schedule is sized from it once), and, for `supcon`, a batch without two samples of one class. A sampler therefore cannot fail part-way through training. The check costs one pass over the positions per epoch. The built-in `balanced` sampler is valid by construction (every class has at least two samples, and each batch draws at least two per class), so it is not iterated; a sampler registered over the name `balanced` is.

## Pairwise loss

Represent pair:

```python
@dataclass(frozen=True)
class ImagePair:
    left: int  # dataset position
    right: int
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

`PairSampler` and `TrainingArguments` both default to:

```text
positives_per_example = 4
negatives_per_example = 4
```

An anchor with fewer available partners gets fewer pairs; partners are never repeated to fill a quota.

## Hard negatives

Hard-negative mining for `pair` training:

- encodes the training split without gradients;
- ranks each anchor's negative pool by cosine similarity;
- pairs each anchor with the most similar members of its negative pool;
- refreshes the embeddings periodically;
- never inspects validation or test data.

### Implemented behaviour (`PairSampler.sample(..., embeddings=, hard_negative_fraction=)`)

Mining changes only *which* negatives an anchor is paired with. For each anchor, `negatives_per_example × hard_negative_fraction` of its negative partners, rounded half up (a positive fraction that rounds to zero is rejected), are the members of its negative pool with the highest cosine similarity to it under the current backbone (ties by dataset position); the rest of the quota, and all positives, stay random. A mix keeps the negatives varied: mining alone returns the same few pairs every epoch. On the synthetic setup of `examples/06_few_shot_benchmark.py` (`dummy/tiny`, three classes, 15 epochs, batch size 12, learning rate 1e-2, seeds 1 to 5) mining gave no clear gain. Test accuracy at fractions 0, 0.5, and 1.0 was 1.00 at every fraction with 8 shots, 0.98 at every fraction with 4 shots, and 0.97, 0.95, and 0.94 with 3 shots; with 2 shots it was 0.67, 0.72, and 0.74, but the spread between seeds there (about ±0.12) is as large as the difference. No run diverged. Whether it helps on real data, where hard negatives carry more information, is an open question (`docs/research/EXPERIMENT_MATRIX.md`). The negative pool is the one the class labels and relationship rules define, so mining never turns a positive into a negative.

The sampler can only rank its own samples, and `Trainer` builds it from `train_dataset`, so validation and test data cannot be mined. During training the embeddings come from `model.encode` on the training images, which runs without gradients in eval mode and restores training mode, and they are refreshed every `hard_negative_refresh_epochs` epochs. Validation pairs are always random, which keeps the validation loss comparable across epochs and between runs with and without mining.

Mined pairs depend on the weights at the time, so they cannot be drawn ahead of training. With the epoch budget, the learning-rate schedule is therefore planned on the largest number of pairs an epoch can hold, and the rate ends above zero: by under 1% of the base rate once the data has a few dozen samples per class, but by as much as 15% on very small sets where de-duplication removes many pairs. With `max_steps` the schedule is sized on the step budget and training continues until it is spent, so the rate reaches zero (ADR 0017).

Similarities are computed as one matrix product per block of 512 anchors, which makes drawing an epoch's pairs with mining about 1.8 times as slow as without it (`PairSampler.sample` on 5 000 samples with 32-dimensional embeddings: 5.1 s against 2.8 s with two threads on a loaded machine). Training on the pairs costs far more than drawing them: an epoch trains on about 40 000 pairs, while one forward pass over the 5 000 images took 2 to 3 s. The pairs are re-mined every epoch from the most recent embeddings; `hard_negative_refresh_epochs` only controls how often those embeddings are recomputed.
