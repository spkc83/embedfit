"""Frozen features + logistic regression: the baseline every other method must beat.

python examples/01_frozen_baseline.py
"""

from _common import parse_args

from embedfit import EmbedFitModel, ImageDataset, Trainer, TrainingArguments
from embedfit.testing import BAR_CLASSES, make_bar_images


def main(argv: list[str] | None = None) -> dict[str, float]:
    args = parse_args(__doc__, argv)
    labels = list(BAR_CLASSES[:3])
    train = ImageDataset.from_lists(*make_bar_images(12, num_classes=3, seed=args.seed))
    test = ImageDataset.from_lists(*make_bar_images(20, num_classes=3, seed=args.seed + 1))

    model = EmbedFitModel.from_pretrained(args.backbone, labels=labels)
    trainer = Trainer(model, TrainingArguments(training_mode="frozen", seed=args.seed), train)
    trainer.train()  # The backbone is untouched; only the head is fitted.

    metrics = trainer.evaluate(test)
    print(f"embedding dimension: {model.embedding_dim}")
    print(f"test accuracy {metrics['accuracy']:.2f}, macro F1 {metrics['macro_f1']:.2f}")
    print("first prediction:", model.predict(test.images[0])[0])
    probabilities = model.predict_proba(test.images[0])[0]
    print(
        "its probabilities:",
        {label: round(float(p), 2) for label, p in zip(labels, probabilities, strict=False)},
    )
    return metrics


if __name__ == "__main__":
    main()
