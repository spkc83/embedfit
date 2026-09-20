"""Pairwise contrastive training, with a relationship rule on a field other than the label.

Samples that share a ``group_id`` are declared similar whatever their class. The library does
not know what a group is; the rule only names the field.

    python examples/03_pair_training_with_relationships.py
"""

from collections import Counter

from _common import parse_args

from embedfit import EmbedFitModel, ImageDataset, Trainer, TrainingArguments
from embedfit.data import ImageSample
from embedfit.sampling import PairSampler, RelationshipRule
from embedfit.testing import BAR_CLASSES, make_bar_images


def main(argv: list[str] | None = None) -> dict[str, float]:
    args = parse_args(__doc__, argv)
    labels = list(BAR_CLASSES[:3])
    images, targets = make_bar_images(10, num_classes=3, seed=args.seed)
    train = ImageDataset(
        [
            ImageSample(image, label, group_id=f"group-{position % 6}")
            for position, (image, label) in enumerate(zip(images, targets, strict=True))
        ]
    )
    test = ImageDataset.from_lists(*make_bar_images(20, num_classes=3, seed=args.seed + 1))
    rule = RelationshipRule(field="group_id", relation="same", pair_type="positive", weight=2.0)

    pairs = PairSampler(train, seed=args.seed, rules=[rule]).sample(epoch=0)
    print(
        f"{len(pairs)} pairs for {len(train)} samples (all pairs would be "
        f"{len(train) * (len(train) - 1) // 2}):",
        dict(Counter((pair.target, pair.relationship) for pair in pairs)),
    )

    model = EmbedFitModel.from_pretrained(args.backbone, labels=labels)
    training = TrainingArguments(
        training_mode="pair",
        num_epochs=args.epochs,
        batch_size=32,
        learning_rate=args.learning_rate,
        num_workers=0,
        seed=args.seed,
    )
    trainer = Trainer(model, training, train, relationship_rules=[rule])
    trainer.train()
    metrics = trainer.evaluate(test)
    print(f"test accuracy {metrics['accuracy']:.2f}")
    return metrics


if __name__ == "__main__":
    main()
