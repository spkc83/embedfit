# ADR 0022 — Joint Cross-Entropy and Supervised Contrastive Training

Status: Accepted. Amends ADR 0002 and ADR 0012.

## Decision

`TrainingArguments.contrastive_weight` (default 0.0, a finite number >= 0) adds a supervised contrastive term to `end_to_end` training. With a weight `w` above 0, the loss of every batch is

```text
CE(classifier(e)) + w · SupCon(z; temperature)
```

- `e` is the backbone embedding, as the model produces it (normalised or not), and `classifier` the jointly trained linear layer of `end_to_end`.
- `z` is `e`, or the output of the SupCon projection head of ADR 0018 when `projection_dim` is set. `SupConLoss` L2-normalises `z`, and its temperature is `TrainingArguments.temperature`.
- Everything else is `end_to_end` as before: the classifier starts from a converged linear probe (ADR 0012), backbone, classifier, and projection head share one AdamW optimiser and schedule, gradient accumulation and mixed precision work as in every mode, the best epoch restores backbone, classifier, and projection head together, and after training the classifier becomes a `LinearHead` that is refitted to convergence (ADR 0011). The projection head is discarded, as in `supcon`.

With a weight above 0:

- **Batches.** The default sampler is `balanced`, the one `supcon` uses, so every sample has a positive in its batch. `sampler` still overrides it, and a sampler other than the built-in `balanced` is checked before training like a `supcon` sampler: a training batch without two samples of one class fails (with one view), as does every such validation batch. A batch without any positive during training raises `ConfigurationError` from `SupConLoss`, as in `supcon`.
- **Views.** `projection_dim` and `contrastive_views` are accepted. With `V` views, the `V × B` augmented images of a batch are embedded in one pass, cross-entropy is the mean over all of them (each view carries its image's label), and SupCon receives `[B, V, D]`, so the views of an image are positives of each other.
- **Validation loss.** The same combined objective, on one view of the original images, over the fixed epoch-0 batches of the same sampler: with the default sampler a class-balanced sample of the validation data, not every validation sample as for plain `end_to_end` (ADR 0011). It is comparable only between runs with the same weight, temperature, and `projection_dim`, and selects the best epoch as in every mode.

With the weight at 0 (the default), `end_to_end` is unchanged: the same batches, the same computations, and bit-identical weights for the same seed. `temperature`, `projection_dim`, and `contrastive_views` then play no part, and the latter two are rejected, as is a weight above 0 in any other mode.

`benchmarks/few_shot.py` and `benchmarks/embedding_quality.py` add the method `ce_supcon`: `end_to_end` with `--contrastive-weight` (default 1.0), its own `--learning-rate ce_supcon=...`, and the contrastive options `--projection-dim`, `--contrastive-views`, and `--temperature`, which it shares with `supcon`. Under `--interpolate` it is refitted with the head of `end_to_end`, a `LinearHead` seeded like the linear probe, so at weight 0 both are exactly that probe.

## Context

ADR 0002 makes two-stage training the canonical flow: adapt the representation with a contrastive loss that *replaces* cross-entropy, freeze it, and fit a head. End-to-end cross-entropy is kept as a baseline. ADR 0012 then made that baseline LP-FT and chose to give it no hyperparameter shared with the contrastive losses.

The few-shot results in which a contrastive term helps fine-tuning come mostly from training it *together with* cross-entropy, not instead of it:

- Gunel et al. (ICLR 2021, arXiv:2011.01403) fine-tune RoBERTa-Large with `(1 − λ) · CE + λ · SCL` and report the largest gains in few-shot settings (20 to 1,000 labelled examples). They search λ in {0.1, 0.3, 0.5, 0.7, 0.9, 1.0} and τ in {0.1, 0.3, 0.5, 0.7}; their best models overwhelmingly use λ = 0.9 and τ = 0.3.
- Core-tuning (Zhang et al., NeurIPS 2021, arXiv:2102.06605) fine-tunes contrastively pretrained ResNet-50s with `CE + η · L_con` on a 256-dimensional two-layer projection at τ = 0.07, tuning η in {0.1, 1, 10} per dataset; η = 0.1 was slightly best on ImageNet20, with the best value differing between datasets.
- Islam et al. (ICCV 2021, arXiv:2103.13517) find that pretraining with a joint objective, `CE + α · L_SelfSupCon`, transfers better than either loss alone, best at α = 2. That is pretraining with a self-supervised term, not fine-tuning with a supervised one, so it supports the joint form only indirectly.

EmbedFit could run the replacement (`supcon`) but not the joint objective.

## Rationale

- **An option of `end_to_end`, not a new mode.** The joint objective needs exactly what `end_to_end` already has: the classifier, the LP-FT start, the refit at the end, the `LinearHead` result, and the head checks in `Trainer`. A separate mode would duplicate all of it, and every place that treats `end_to_end` specially (checkpoints, the benchmark's WiSE refit) would need a second name. A weight of 0 being the baseline exactly also makes the comparison between the two clean: the only difference is the added term.
- **`CE + w · SupCon` rather than `(1 − λ) · CE + λ · SupCon`.** Core-tuning's form keeps the cross-entropy at full weight, so the weight is the only new parameter, 0 is the baseline, and the cross-entropy part of the loss keeps the scale the baseline's learning rate was tuned for. The two forms differ only in scale: λ corresponds to `w = λ / (1 − λ)` with the whole loss multiplied by `1 − λ`, a factor that Adam's updates are nearly invariant to.
- **Class-balanced batches by default.** SupCon needs positives in every batch. In uniformly shuffled few-shot batches many anchors have none: at 4 shots of 10 classes, about 54% of the anchors in a batch of 8 have no other sample of their class in it. This is the sampler `supcon` uses, so the two contrastive methods see the same batches.
- **`contrastive_views` and `projection_dim` reused, not duplicated.** They mean the same thing as in `supcon` (ADR 0018), and the projection head is built, trained, restored, and discarded by the same code. It is one of the modules trained besides the backbone, like the classifier, so `head_learning_rate` (ADR 0020) sets its rate together with the classifier's, and `trainable_parameters` (ADR 0021) never freezes it.

## Relationship to ADR 0002 and ADR 0012

ADR 0002 still holds for what it decides: two-stage training is the default and every model ends with a frozen representation and a head fitted on it (ADR 0011 extends that to `end_to_end`). What changes is the framing that end-to-end training is *only* a baseline. With a contrastive weight, `end_to_end` is one of the representation-learning methods under test, as the literature above suggests it should be; with the weight at 0 it is the cross-entropy baseline, unchanged, and remains the reference every method is compared against (`docs/research/EXPERIMENT_MATRIX.md`).

ADR 0012 rejected a temperature-scaled classifier partly because it would couple the baseline to a contrastive hyperparameter. The baseline stays uncoupled: `temperature` affects `end_to_end` only through the contrastive term, which is off by default. The classifier itself is unchanged.

## Alternatives considered

- **A new training mode (`ce_supcon`).** Rejected for the duplication described above. The name is used in the benchmarks instead, where methods are compared side by side.
- **Default weight in the library above 0.** Rejected: it would change the baseline for every existing user and every earlier result.
- **Cross-entropy on the first view only.** Every view is a valid labelled sample of its image, and averaging cross-entropy over all of them uses the augmentation for the classifier too; with one view the two are identical.
- **A uniform sampler with anchors without positives skipped.** `SupConLoss` already skips them, but a batch without any positive would have to be skipped silently or fail mid-training, and with few anchors per batch the contrastive gradient would come from a handful of samples.

## Consequences

- The benchmark default `--contrastive-weight 1.0` is equal weighting: the middle of Core-tuning's grid on a log scale, and between its best value on ImageNet20 (0.1) and the equivalent of Gunel et al.'s few-shot optimum (λ = 0.9, `w` = 9). The literature does not support one value across domains; tune it on the validation split with the learning rate.
- `ce_supcon` is not in the benchmarks' default `--methods`, so a results file from before this change resumes unchanged; such a file resumes as one with the default `--contrastive-weight`.
- A weight above 0 costs one similarity matrix per batch (`(V × B)²`), plus the projection head when it is used: small next to the backbone.
- The default `balanced` sampler needs at least two training samples per class, and `batch_size` at least 4, so `end_to_end` with a contrastive weight, like `supcon`, cannot train at one shot unless `sampler` is set.
