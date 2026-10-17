# RVL-CDIP with DINOv2-small

The `frozen`, `end_to_end`, and `supcon` training modes on scanned document pages from RVL-CDIP with DINOv2-small, run on a CPU. It follows the protocol of [EuroSAT with DINOv2-small](eurosat-dinov2-small.md) on document imagery, which is far from both EuroSAT and DINOv2's pretraining data.

## Result

| Method | 4 shots | 8 shots | 16 shots |
|---|---|---|---|
| `frozen` | 0.399 ± 0.008 | 0.423 ± 0.007 | 0.459 ± 0.016 |
| `supcon` | 0.397 ± 0.015 | 0.433 ± 0.008 | 0.496 ± 0.009 |
| `end_to_end` | **0.422 ± 0.005** | **0.459 ± 0.020** | **0.514 ± 0.008** |

Test accuracy on 16 classes (chance is 0.0625), mean ± sample standard deviation over three seeds. Balanced accuracy is within 0.001 of accuracy in every run. Macro F1 is lower than accuracy in every run, by 0.009 to 0.060.

- `end_to_end` (cross-entropy fine-tuning from a linear probe, ADR 0012) has the highest accuracy on every one of the nine few-shot samples, although on two of them it leads `supcon` by less than 0.004 (3 and 5 pages). Its mean is above `frozen` by 0.023, 0.036, and 0.055 at 4, 8, and 16 shots.
- `supcon` improves on `frozen` at 16 shots, by 0.028 to 0.043 on each of the three seeds. At 4 and 8 shots it does not separate from `frozen`: its seed-by-seed differences from `frozen` run from −0.013 to +0.021, and the 8-shot gain is +0.003 and +0.004 on two of the three seeds.
- Every method is far from solving the task: the best cell is 0.514 at 16 shots.

On EuroSAT with the same backbone, `end_to_end` was also best on all nine samples, and `supcon` did not separate from `frozen` at any shot count.

## Setup

| | |
|---|---|
| Dataset | RVL-CDIP through the Hugging Face dataset `jinhybr/rvl_cdip_400_train_val_test`, pinned to revision `13462a42e3375e80a9bd46a64c58e1bcaba77874`: scanned office documents in 16 classes, with 400 training, 100 validation, and 100 test pages per class |
| Splits | The dataset's own splits. A training pool of 100 pages per class is drawn from its training split with seed 0; the validation and test sets are 100 pages per class from its validation and test splits |
| Duplicates | Evaluation pages whose pixels exactly match a pool page are dropped. One test page and one validation page were, which leaves 1,599 in each |
| Pages | Converted to grayscale and shrunk to a shorter side of 512 px before use |
| Backbone | `hf/facebook/dinov2-small` (22M parameters), the provider's default pooling, input 224×224 with the checkpoint's preprocessing: shorter side resized to 256 px, then a centre crop |
| Shots | 4, 8, and 16 per class, drawn from the pool; seeds 0, 1, 2 |
| Budget | 10 epochs, batch size 16 with 2 gradient-accumulation steps (32 samples per update), AdamW, linear decay, no warm-up, weight decay 0.01, gradient clipping at 1.0 |
| Head | Logistic regression for `frozen` and `supcon`; the fine-tuned linear classifier refitted to convergence for `end_to_end` (ADR 0011) |
| Hardware | CPU only, 4 threads, on a shared machine |

The input is DINOv2's default 224 px, not the 112 px of the EuroSAT runs, because document classes depend on layout and text that a smaller input would blur. The batch is split into two accumulation steps because larger batches at 224 px exceeded the memory available on this machine. The steps average gradients, so the optimiser update covers 32 samples, but `supcon`'s contrastive loss is computed within each 16-sample step: with 16 classes, an anchor has few or no positives in its step. On EuroSAT, `supcon` compared 32 samples across 10 classes.

`pair` was not run. On EuroSAT its representation training took about 3.3 times as long as `supcon`'s, which here would have added well over ten hours.

Every method trains on the same few-shot sample for a given shot count and seed. No validation data is passed to training, so each run keeps its last epoch.

### Learning rates

Each trained method's learning rate was chosen on the validation split, at 8 shots with seed 0, before any test-split run.

| Method | 2e-6 | 1e-5 | 5e-5 | 2.5e-4 | Chosen |
|---|---|---|---|---|---|
| `end_to_end` | 0.461 | **0.485** | 0.266 | 0.130 | 1e-5 |
| `supcon` | **0.450** | 0.445 | 0.160 | 0.121 | 2e-6 |

Validation accuracy of the frozen baseline in the same setting: 0.442.

The rates are lower than on EuroSAT, where they were 1e-5 for `end_to_end` and 5e-5 for `supcon`. From 5e-5 up, training damages the pretrained features: at 5e-5, `end_to_end`'s training loss in the first epoch was 3.59, above the 2.77 of a uniform guess over 16 classes, against 0.91 at 1e-5.

`supcon`'s chosen rate is the lowest one tested, and lower rates were not tried. As the rate approaches zero, `supcon` approaches the frozen encoder, and the three values at that end (0.442 for `frozen`, 0.450 at 2e-6, 0.445 at 1e-5) are 0.009 apart at most, below the standard error of one accuracy on 1,599 pages (about 0.012). All three are scored on the same pages, so this is only a rough yardstick for their differences. The sweep does not tell 2e-6 from 1e-5 for `supcon`. Tuning used one seed at one shot count, so the chosen rates are not necessarily best at 4 or 16 shots.

The records of the tuning runs are in the results file.

## Limits

- One dataset, one small backbone, three seeds, and rates tuned on one run per method.
- The centre crop keeps the middle of the page: for a typical page about 1.33 times taller than wide, about two thirds of its height and seven eighths of its width. The top and bottom of the page, where letterheads, titles, and form headers usually are, are cut off. A preprocessing that keeps the whole page might score higher for every method.
- `supcon`'s contrastive loss sees 16 samples at a time (see Setup), half as many as on EuroSAT; larger contrastive batches might change its result.
- Ten epochs is a small budget, and contrastive objectives are often trained for longer.
- The 8-shot, seed-0 training sample used for tuning is also the one in the 8-shot, seed-0 test cell. The rates were scored on the validation split, so the test images were not involved, but that cell is not fully independent of the tuning.

## Cost

The test-split run took 642 minutes on this machine for all three methods, shots, and seeds: 75 minutes for `frozen`, 285 for `end_to_end`, and 282 for `supcon`, each including encoding the test set once per few-shot sample. The four validation runs for tuning took another 309 minutes. The machine was under load from other jobs.

## Reproducing

```bash
pip install -e ".[hf,datasets]"
# From the repository root. Choose learning rates on the validation split, one command per candidate rate:
python benchmarks/few_shot.py --dataset rvl-cdip --evaluate-on validation \
    --shots 8 --seeds 0 --methods frozen end_to_end supcon \
    --learning-rate end_to_end=1e-5 supcon=1e-5 \
    --batch-size 16 --gradient-accumulation-steps 2 --output .runs/rvl-tune-1e-5.json
# Then run the test split once with the chosen rates:
python benchmarks/few_shot.py --dataset rvl-cdip --evaluate-on test --methods frozen end_to_end supcon \
    --learning-rate end_to_end=1e-5 supcon=2e-6 \
    --batch-size 16 --gradient-accumulation-steps 2 --output .runs/rvl-test.json
```

The script downloads the dataset (about 1.1 GB) and the DINOv2-small weights into the repository's git-ignored `.cache/` folder on first use and writes the shrunk pages there. The records behind the tables are in [`benchmarks/results/rvl_cdip_dinov2_small.json`](../../benchmarks/results/rvl_cdip_dinov2_small.json), with the runs' configuration and the test run's code commit.
