# EuroSAT with DINOv2-small

A few-shot comparison of the four training modes on real images with a pretrained encoder, run on a CPU. It answers research questions 1–3 of [`EXPERIMENT_MATRIX.md`](../research/EXPERIMENT_MATRIX.md) for one backbone and one dataset.

## Result

| Method | 4 shots | 8 shots | 16 shots |
|---|---|---|---|
| `frozen` | 0.722 ± 0.005 | 0.770 ± 0.016 | 0.836 ± 0.004 |
| `supcon` | 0.670 ± 0.036 | 0.780 ± 0.028 | 0.822 ± 0.019 |
| `pair` | 0.696 ± 0.044 | 0.786 ± 0.020 | 0.853 ± 0.027 |
| `end_to_end` | **0.757 ± 0.003** | **0.847 ± 0.005** | **0.902 ± 0.019** |

Test accuracy, mean ± sample standard deviation over three seeds. The classes are balanced, so balanced accuracy is identical and macro F1 is within 0.01 of accuracy in every cell.

- `end_to_end` (cross-entropy fine-tuning from a linear probe, ADR 0012) has the highest accuracy on every one of the nine few-shot samples, not only on average.
- Neither contrastive mode separates from the frozen baseline. Their differences from `frozen` go both ways (from −0.052 for `supcon` at 4 shots to +0.017 for `pair` at 16) and none exceeds 1.5 of the contrastive mode's own standard deviations.
- The contrastive modes are also the least stable across seeds (standard deviations up to 0.044, against at most 0.019 for `end_to_end`).

On this task, supervised contrastive adaptation does not improve over the frozen encoder, and cross-entropy fine-tuning improves on both.

## Setup

| | |
|---|---|
| Dataset | EuroSAT RGB: 27,000 Sentinel-2 land-cover tiles of 64×64 px, 10 classes |
| Splits | Disjoint and drawn per class with seed 0: a training pool of 100, a validation set of 100, and a test set of 100 images per class (1,000 test images) |
| Backbone | `hf/facebook/dinov2-small` (22M parameters), the provider's default pooling, input 112×112 |
| Shots | 4, 8, and 16 per class, drawn from the pool; seeds 0, 1, 2 |
| Budget | 10 epochs, batch size 32, AdamW, linear decay, no warm-up, weight decay 0.01, gradient clipping at 1.0 |
| Pair mode | One positive and one negative partner per sample per epoch |
| Head | Logistic regression for `frozen`, `supcon`, and `pair`; the fine-tuned linear classifier refitted to convergence for `end_to_end` (ADR 0011) |
| Hardware | CPU only, 4 threads, on a shared machine |

The input is 112 px rather than DINOv2's default 224 px. The tiles are 64 px, so the smaller input discards no image detail, and it made the runs about four times faster on this machine.

Every method trains on the same few-shot sample for a given shot count and seed. No validation data is passed to training, so each run keeps its last epoch.

### Learning rates

Each trained method's learning rate was chosen on the validation split, at 8 shots with seed 0, before any test-split run:

| Method | 2.5e-6 | 1e-5 | 5e-5 | 2.5e-4 | Chosen |
|---|---|---|---|---|---|
| `end_to_end` | 0.832 | **0.844** | 0.767 | — | 1e-5 |
| `supcon` | — | 0.827 | **0.843** | 0.316 | 5e-5 |
| `pair` | 0.750 | **0.807** | 0.578 | 0.307 | 1e-5 |

Validation accuracy of the frozen baseline in the same setting: 0.786. The chosen rate is inside the tested range for every method. Tuning used one seed at one shot count, so the chosen rates are not necessarily best at 4 or 16 shots.

The `end_to_end` sweep was run after the change in ADR 0012. Before it, the baseline's classifier started from random weights at the backbone's learning rate and scored 0.766, 0.719, and 0.343 at 1e-5, 5e-5, and 2.5e-4, below the frozen baseline at every rate.

## Limits

- One dataset, one small backbone, three seeds. EuroSAT is a satellite-image task that is far from DINOv2's pretraining data; the ordering may differ on documents or other domains.
- Ten epochs is a small budget, and contrastive objectives are often trained for longer. More epochs might change the contrastive results; they were not tested here.
- Pair mode sees four images per training sample per epoch against one for the other modes, so its compute per epoch is about four times theirs.
- Seed 0 is used for both the split and the first few-shot draw.

## Cost

On this machine, `frozen`, `supcon`, and `pair` together took 131 minutes for all shots and seeds, and `end_to_end` took 39 minutes. A separate throughput test at the same input size measured about 18 images per second for encoding and 5.5 for training.

## Reproducing

```bash
pip install -e ".[hf,torchvision]"
# Choose learning rates on the validation split, one command per candidate rate:
python benchmarks/eurosat_few_shot.py --evaluate-on validation --shots 8 --seeds 0 \
    --methods frozen end_to_end supcon pair \
    --learning-rate end_to_end=1e-5 supcon=1e-5 pair=1e-5 --output tune_1e-5.json
# Then run the test split once with the chosen rates:
python benchmarks/eurosat_few_shot.py --evaluate-on test \
    --learning-rate end_to_end=1e-5 supcon=5e-5 pair=1e-5 --output results.json
```

The script downloads EuroSAT (about 90 MB) through torchvision on first use. The records behind the table are in [`benchmarks/results/eurosat_dinov2_small.json`](../../benchmarks/results/eurosat_dinov2_small.json).
