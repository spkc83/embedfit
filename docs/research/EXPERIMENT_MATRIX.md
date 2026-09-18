# Research Experiment Matrix

## Research questions

1. Does SupCon adaptation improve over frozen pretrained features?
2. Does it improve over standard cross-entropy fine-tuning?
3. At which label counts does representation adaptation help?
4. Which backbone family transfers best to check/document imagery?
5. Does hard-negative mining improve separation?
6. Does retrieval identify recurring risk patterns beyond classification?
7. How much performance drops on out-of-time data?

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
