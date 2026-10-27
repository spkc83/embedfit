# ADR 0019 — The Benchmarks Can Score Every Method with the Same Head

Status: Accepted. Corrects the rationale of ADR 0011, which said that every arm gets the same head.

## Decision

`benchmarks/few_shot.py --heads HEAD [HEAD ...]`, with `HEAD` one of `logistic_regression` and `linear`, scores every method with each named head. For every method, shot count, and seed, the backbone is trained once. At the trained weights and at each interpolation weight (`--interpolate`), which only sets the backbone between the pretrained and the trained weights, each head is then fitted to convergence on the training embeddings of that backbone and the model is scored with it. Every record has a `head` key that names the head it was scored with.

- The head the method was already given is used as it is: for `end_to_end` the `LinearHead` warm-started from the jointly trained classifier (ADR 0011), for the other methods the default `logistic_regression` head.
- Every other head is new: `linear` is a `LinearHead` seeded and configured like the linear probe of `end_to_end` (ADR 0012), and `logistic_regression` has the default settings, like the other methods' own head. The training embeddings are encoded once for all new heads.

Without `--heads`, every method is scored with its own head only, as before the option. Then the summary rows do not name the head. The option is part of a results file's configuration, as a sorted list or `null`. A file from before the option resumes as one without it, and each of its records gets the head its method used.

The library does not change. `Trainer` still fits `args.head` (default `logistic_regression`) in `frozen`, `supcon`, and `pair`, and a `LinearHead` in `end_to_end`.

## Context

ADR 0011 made every mode end with a head fitted to convergence, and said that every arm then gets the same head. That is not true: the heads are different, and so is their regularisation.

- `frozen`, `supcon`, and `pair` get `LogisticRegressionHead(C=1.0)`. scikit-learn minimises `0.5 ||W||² + C Σ loss`, which is the mean cross-entropy plus `0.5 / (C N) ||W||²` for `N` training samples.
- `end_to_end` gets `LinearHead(weight_decay=1e-4)`, which minimises the mean cross-entropy plus `0.5 × 1e-4 ||W||²`.

With 40 to 160 training samples (4 to 16 shots of 10 classes), the logistic-regression penalty is `1 / (1e-4 N)`, about 60 to 250 times larger. On unit-length embeddings, where logits are small unless the weights are large, the penalty decides how confident and how separable the head can be. A difference between `end_to_end` and the other methods therefore mixes two variables: how the backbone was adapted, and which head scored it.

## Alternatives considered

- **Change the default head of one side.** Rejected: it changes the library's defaults and every earlier result, and it removes the comparison with the head each method would get in normal use.
- **One head for all methods, chosen by a single option.** `--heads` with one name does this. With two names, the same training run also gives each method's result with both heads, so the size of the head effect is measured, not assumed.
- **A separate run per head.** Rejected: the representation training dominates the cost, and it would be repeated for nothing.

## Consequences

- With `--heads`, differences between methods at the same head come from the backbone only. At `--interpolate 0` every trained method is the pretrained backbone, so with the same head it scores exactly as `frozen` does.
- For every method and weight, the additional heads together cost one encoding pass over the training set, and each costs one fit on those embeddings and one encoding pass over the evaluated split. The training is not repeated.
- The trained `end_to_end` model's `linear` head is warm-started from the jointly trained classifier, and other methods' `linear` heads start from a seeded initialisation. Both are fitted with L-BFGS to the same convex objective, so they reach about the same optimum.
- `benchmarks/embedding_quality.py` has no `--heads` option. Most of its metrics use the embeddings only. Its head-based metrics (`seen_accuracy` and the `rare_positive` ranking by the head's probability) still use each method's own head.
- Results files from before the option are unchanged and resume as before.
