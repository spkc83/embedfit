# ADR 0012 — The End-to-End Baseline Starts from a Linear Probe

Status: Accepted. Amends ADR 0011, whose measurements describe the earlier, randomly initialised baseline. Amended by ADR 0022: `contrastive_weight` adds a supervised contrastive term, which uses `temperature`; at its default of 0 the baseline is unchanged.

## Decision

`end_to_end` training is linear-probe-then-fine-tune (LP-FT; Kumar et al., 2022, "Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution"). Before any backbone update, a `LinearHead` is fitted to convergence on the training embeddings of the untouched backbone, with the same settings as the head refitted after training (ADR 0011: default weight decay and iterations, seeded with `args.seed`). Its weights initialise the classifier that is then trained jointly with the backbone under cross-entropy, in the same loop, optimiser, and schedule as before.

There is no option to switch this off and no new hyperparameter.

## Context

The classifier used to start from a random initialisation and was trained with the backbone's optimiser at the backbone's learning rate. For a pretrained encoder that rate is small (1e-5 to 5e-5), and a few-shot run takes a few dozen steps. Measured on EuroSAT with DINOv2-small at 8 shots per class (ten epochs, one seed, validation split):

- Cross-entropy fell only from 2.30 to 2.09 at 1e-5 and to 1.92 at 5e-5 (ln 10 = 2.30): the classifier barely learned, so the backbone was trained on gradients through a nearly random classifier. Validation accuracy was 0.766 and 0.719, below the frozen baseline's 0.786.
- At 2.5e-4 the backbone was destroyed (0.343).

A baseline in this state makes any representation-learning method look better than it is. The comparison would measure the classifier's initialisation, not the value of adapting the backbone, which is the same problem ADR 0011 fixed for the final head.

## Alternatives considered

- **A separate, larger learning rate for the classifier** (parameter groups). Also a common practice and it would train the classifier, but it adds a hyperparameter that has to be tuned on validation data for every backbone and shot count, and it does nothing about early updates driven by a poor classifier.
- **A temperature-scaled (cosine) classifier**: logits on normalised embeddings divided by `args.temperature`, the usual remedy for the small logits of unit-length embeddings. Measured on the synthetic benchmark with the randomly initialised classifier, it lowered few-shot test accuracy (0.53 against 0.66 at 4 shots, 0.66 against 0.77 at 8 shots, five seeds), and it couples the baseline to a contrastive-loss hyperparameter. Rejected: `end_to_end` does not use `temperature`.
- **Benchmark the baseline as it was and document the weakness.** Rejected: the benchmark exists to answer whether representation learning beats standard fine-tuning, and a known-weak baseline cannot answer it.

LP-FT needs no new setting, starts fine-tuning from the best linear classifier the pretrained features allow, and in Kumar et al.'s analysis distorts pretrained features less than fine-tuning from a random head.

## Consequences

- An extra encoding pass over the training set and one L-BFGS fit at the start of `end_to_end` training; the embeddings are held in memory for the fit. Both are small for few-shot data.
- With a learning rate too small to change the backbone, `end_to_end` is the frozen baseline with a linear head, by construction.
- The probe's classifier has a large weight norm on well-separated features, so gradients reaching the backbone stay informative although the starting loss is low. The flip side: a learning rate that suited a random start can be too large. On the synthetic example (`examples/06_few_shot_benchmark.py` with its defaults: untrained demo network, 15 epochs, batch size 12, learning rate 1e-2; seeds 1 to 5) the first epoch disturbs the probe, and accuracy over five seeds went from 0.93 ± 0.05 to 0.86 ± 0.08 at 4 shots and from 1.00 to 0.97 ± 0.06 at 8 shots (unchanged at 16). An untrained network has no pretrained features for a probe to exploit, so that setting says little about LP-FT, but learning rates tuned for the old baseline must be tuned again.
- The probe is fitted on eval-mode features. With the default `freeze_batch_norm=True` (ADR 0013), batch-norm layers stay in eval mode during fine-tuning too, so the probe sees the features fine-tuning starts from; with `False` they switch to batch statistics, as in Kumar et al.
- A class with no training sample is rejected before training by `validate_representation_setup` as well as by `Trainer`, because the probe needs every class.
- Results of `end_to_end` from before this change are not comparable with later ones.
