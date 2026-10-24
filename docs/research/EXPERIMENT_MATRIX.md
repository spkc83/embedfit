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

Fixed before the runs. A method *adds lift* on a dataset only if, against both `end_to_end` and `frozen`:

- its mean over 5 paired seeds is higher, and
- the paired difference (the same few-shot sample for both methods) is positive in at least 4 of the 5 seeds,

on the classification test split (`benchmarks/few_shot.py`) and on the embedding-quality tasks (`benchmarks/embedding_quality.py`).

The learning rate and the step budget are tuned per dataset on its validation split, with the same small grid for every method; the test split is used once, with the chosen settings.

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
