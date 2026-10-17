# EuroSAT with ResNet-50

The four training modes on EuroSAT with an ImageNet ResNet-50, run on a CPU. The protocol is the one of [EuroSAT with DINOv2-small](eurosat-dinov2-small.md), with a convolutional backbone that uses batch normalisation in place of a vision transformer, so that the two can be compared.

## Result

| Method | 4 shots | 8 shots | 16 shots |
|---|---|---|---|
| `frozen` | 0.687 ± 0.003 | 0.752 ± 0.025 | 0.812 ± 0.014 |
| `supcon` | 0.711 ± 0.019 | 0.801 ± 0.014 | 0.831 ± 0.014 |
| `pair` | 0.721 ± 0.028 | 0.789 ± 0.014 | 0.841 ± 0.031 |
| `end_to_end` | **0.739 ± 0.063** | **0.848 ± 0.010** | **0.878 ± 0.004** |

Test accuracy, mean ± sample standard deviation over three seeds. The classes are balanced, so balanced accuracy is identical, and macro F1 is within 0.02 of accuracy in every run.

- `end_to_end` (cross-entropy fine-tuning from a linear probe, ADR 0012) has the highest mean accuracy at every shot count, and the highest accuracy on eight of the nine few-shot samples. On the ninth (4 shots, seed 0) it has the lowest, 0.667 against 0.684 to 0.692 for the other methods, which is why its 4-shot spread is large.
- Both contrastive modes improve on the frozen baseline: by 0.02 to 0.05 on average in each column, and on 16 of their 18 seed-by-seed comparisons with `frozen`. The two exceptions are at 16 shots (−0.011 for `supcon`, −0.006 for `pair`).
- `supcon` and `pair` are level with each other: neither is ahead in every column, and their differences (at most 0.012) are smaller than their spreads across seeds.

The frozen ResNet-50 is below the frozen DINOv2-small in every column (0.687, 0.752, and 0.812 against 0.722, 0.770, and 0.836), and fine-tuning closes most of that gap. With DINOv2-small, contrastive adaptation did not separate from the frozen encoder; with ResNet-50 it improved on it. The two backbones differ in architecture, size, and pretraining, so this comparison does not say which of these causes the difference.

## Setup

| | |
|---|---|
| Dataset | EuroSAT RGB: 27,000 Sentinel-2 land-cover tiles of 64×64 px, 10 classes |
| Splits | Disjoint and drawn per class with seed 0: a training pool of 100, a validation set of 100, and a test set of 100 images per class (1,000 test images); the same images as the DINOv2-small benchmark |
| Backbone | `torchvision/resnet50` (ImageNet weights, 25M parameters, 2048-dimensional embedding), input 112×112 |
| Batch norm | Running statistics fixed during representation training (`freeze_batch_norm=True`, the default; ADR 0013) |
| Shots | 4, 8, and 16 per class, drawn from the pool; seeds 0, 1, 2 |
| Budget | 10 epochs, batch size 32, AdamW, linear decay, no warm-up, weight decay 0.01, gradient clipping at 1.0 |
| Pair mode | One positive and one negative partner per sample per epoch |
| Head | Logistic regression for `frozen`, `supcon`, and `pair`; the fine-tuned linear classifier refitted to convergence for `end_to_end` (ADR 0011) |
| Hardware | CPU only, 4 threads, on a shared machine |

Every method trains on the same few-shot sample for a given shot count and seed. No validation data is passed to training, so each run keeps its last epoch.

### Learning rates

Each trained method's learning rate was chosen on the validation split, at 8 shots with seed 0 and with batch-norm statistics fixed, before any test-split run.

| Method | 1e-5 | 5e-5 | 2.5e-4 | 1.25e-3 | Chosen |
|---|---|---|---|---|---|
| `end_to_end` | 0.802 | 0.836 | **0.866** | 0.122 | 2.5e-4 |
| `supcon` | 0.795 | 0.821 | **0.835** | 0.318 | 2.5e-4 |
| `pair` | 0.767 | **0.811** | 0.671 | 0.145 | 5e-5 |

Validation accuracy of the frozen baseline in the same setting: 0.770. The chosen rate is inside the tested range for every method. The records of these tuning runs were not kept; the 1e-5 and 2.5e-4 figures for `end_to_end` and `supcon` are also reported in ADR 0013. Tuning used one seed at one shot count, so the chosen rates are not necessarily best at 4 or 16 shots.

The best rates are about 5 to 25 times those for DINOv2-small (1e-5 for `end_to_end` and `pair`, 5e-5 for `supcon`).

With train-mode batch norm, every trained method was below the frozen baseline at low rates; ADR 0013 gives the measurements.

## Limits

- One dataset, three seeds, and a rate tuned on one run per method. EuroSAT is far from ImageNet; the ordering may differ in other domains.
- Ten epochs is a small budget, and contrastive objectives are often trained for longer.
- The 8-shot, seed-0 training sample used for tuning is also the one in the 8-shot, seed-0 test cell. The rates were scored on the validation split, so the test images were not involved, but that cell is not fully independent of the tuning.
- The comparison with DINOv2-small changes several things at once (architecture, size, pretraining data, normalisation layers, learning rates).

## Cost

On this machine all four methods together took 296 minutes for all shots and seeds, about 1.7 times the DINOv2-small run (170 minutes), under heavy load from other jobs.

## Reproducing

```bash
pip install -e ".[torchvision]"
# From the repository root. Choose learning rates on the validation split, one command per candidate rate:
python benchmarks/few_shot.py --dataset eurosat --backbone torchvision/resnet50 --evaluate-on validation \
    --shots 8 --seeds 0 --methods frozen end_to_end supcon pair \
    --learning-rate end_to_end=2.5e-4 supcon=2.5e-4 pair=2.5e-4 --output .runs/r50-tune-2.5e-4.json
# Then run the test split once with the chosen rates:
python benchmarks/few_shot.py --dataset eurosat --backbone torchvision/resnet50 --evaluate-on test \
    --learning-rate end_to_end=2.5e-4 supcon=2.5e-4 pair=5e-5 --output .runs/r50-test.json
```

The script downloads EuroSAT (about 90 MB) through torchvision and the ResNet-50 weights (about 100 MB) through torch hub on first use, both into the repository's git-ignored `.cache/` folder. The records behind the table are in [`benchmarks/results/eurosat_resnet50.json`](../../benchmarks/results/eurosat_resnet50.json), with the run's configuration and code commit.
