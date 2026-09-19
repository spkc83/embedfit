# Examples

Each script is self-contained, uses generated images only, and runs offline in a few seconds on the `dummy/tiny` backbone (a small untrained CNN). Run them from this directory:

```bash
cd examples
python 01_frozen_baseline.py
```

| Script | Shows |
| --- | --- |
| [`01_frozen_baseline.py`](01_frozen_baseline.py) | Frozen embeddings + logistic regression: `from_pretrained`, `Trainer` in `frozen` mode, `predict`, `predict_proba`, `evaluate` |
| [`02_supcon_few_shot.py`](02_supcon_few_shot.py) | Supervised contrastive adaptation against the frozen baseline, with a validation split choosing the epoch |
| [`03_pair_training_with_relationships.py`](03_pair_training_with_relationships.py) | Pair training with a `RelationshipRule` on a field the library knows nothing about |
| [`04_retrieval.py`](04_retrieval.py) | Indexing adapted embeddings and scoring retrieval with `exclude_self`, so an item never retrieves itself |
| [`05_synthetic_document_risk.py`](05_synthetic_document_risk.py) | A rare-event workflow: out-of-time and grouped splits, a class-weighted head, ranking and value-weighted metrics, a threshold chosen on validation data, and retrieval of similar items |
| [`06_few_shot_benchmark.py`](06_few_shot_benchmark.py) | `FewShotEvaluator`: frozen vs end-to-end cross-entropy vs SupCon over shots and seeds |

Common options: `--backbone`, `--epochs`, `--learning-rate`, `--seed`, `--quiet`.

To use a pretrained encoder, install the extra and pass its identifier. Pretrained backbones need a much smaller learning rate than the untrained demo network:

```bash
pip install 'embedfit[torchvision]'
python 02_supcon_few_shot.py --backbone torchvision/resnet18 --learning-rate 2e-5 --epochs 5
```

The synthetic images make every task easy, so the numbers printed here say nothing about real data. `05_synthetic_document_risk.py` in particular is about the order of operations, which is where rare-event projects usually go wrong: split first, tune on validation only, and report ranking metrics rather than accuracy.
