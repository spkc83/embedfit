# Research Experiment Matrix

EmbedFit is a generic framework for few-shot adaptation of image embeddings. The experiments ask where supervised contrastive adaptation (`supcon`, `pair`) improves on the two baselines any practitioner would try, the frozen backbone and end-to-end fine-tuning, and on which kinds of images.

## Research questions

1. Does SupCon adaptation improve over frozen pretrained features?
2. Does it improve over standard cross-entropy fine-tuning?
3. At which label counts does representation adaptation help?
4. In which image domains does representation adaptation add lift, and which backbone family transfers best to each?
5. Does hard-negative mining improve separation?
6. Does retrieval identify recurring risk patterns beyond classification? `benchmarks/embedding_quality.py` measures retrieval and novelty detection on held-out classes and similarity ranking of rare positives ([benchmarking guide](../guides/benchmarking.md#embedding-quality)).
7. How much performance drops on out-of-time data?

## Domains

Each domain is a public dataset that `benchmarks/few_shot.py` and `benchmarks/embedding_quality.py` download at run time ([benchmarking guide](../guides/benchmarking.md#datasets) for sizes, licences, and splits). They range from close to the backbones' pretraining data to far from it:

- natural objects: Oxford-IIIT Pets (`pets`, 37 classes)
- fine-grained natural objects: Oxford Flowers-102 (`flowers`, 102 classes)
- textures: Describable Textures (`dtd`, 47 classes)
- satellite imagery: EuroSAT RGB (`eurosat`, 10 classes)
- scanned documents: RVL-CDIP (`rvl-cdip`, 16 classes)
- histopathology: PatchCamelyon (`pcam`, 2 classes; `rare_positive` only, as `open_set` needs at least four classes)

## Lift rule

Fixed before the runs. Every dataset runs at `--image-size 112` with DINOv2-small, the size of the EuroSAT runs. The published RVL-CDIP run used 224 px, so RVL-CDIP enters the suite only through a run at 112 px.

**Primary metrics.** Classification: test `accuracy` from `benchmarks/few_shot.py`. Embedding: `unseen_map`, `novelty_heldout_pr_auc` (`open_set` task), and `knn_pr_auc` (`rare_positive` task) from `benchmarks/embedding_quality.py`. A dataset with too few classes for `open_set` (PCam) uses `accuracy` and `knn_pr_auc` only. Every other metric is secondary and descriptive only.

**Lift on a metric at a shot count.** Over 5 seeds, every method trains on the same few-shot sample per seed (paired seeds). A method has lift on a primary metric at a shot count when both hold:

- its mean over the 5 seeds is higher than the means of `end_to_end` and of `frozen`;
- its paired difference against `end_to_end` is positive in at least 4 of the 5 seeds.

**Lift on a dataset.** A method has lift on a dataset for a metric when it has lift on that metric at 2 or more of the 3 shot counts (4, 8, 16). Classification lift and embedding lift are reported separately and are not pooled.

**Tuning.** The learning rate and the step budget are tuned per dataset on its validation split, with the same small grid for every method; the test split is used once, with the chosen settings. Weight-interpolated (WiSE-FT) variants are reported at every interpolation weight run, but a variant counts for lift only if its weight was chosen on the validation split before the test run.

## Backbone matrix

Initial candidates:

- ResNet50 ImageNet
- ConvNeXt Tiny ImageNet
- ViT-B/16 ImageNet
- DINOv2 Base or current comparable self-supervised vision encoder

## Training methods

- frozen + logistic regression
- end-to-end cross-entropy
- SupCon + logistic regression
- pair contrastive + logistic regression

## Shots

- 4
- 8
- 16
- 32
- 64
- 128

## Seeds

Minimum 3; prefer 5.

Do not report only the best seed.

## Metrics

- PR-AUC
- ROC-AUC
- macro F1
- recall @ fixed FPR
- recall @ review rate
- dollar recall @ review rate when amount exists

## Fairness of comparison

All methods must receive:

- identical labeled examples per seed/shot
- identical train/validation/test boundaries
- identical preprocessing where checkpoint requirements permit
- documented hyperparameter budgets
