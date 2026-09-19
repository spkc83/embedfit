"""Supervised contrastive adaptation, then the same lightweight head, compared with frozen.

python examples/02_supcon_few_shot.py
"""

from _common import parse_args

from embedfit import EmbedFitModel, ImageDataset, Trainer, TrainingArguments
from embedfit.data import stratified_split
from embedfit.testing import BAR_CLASSES, make_bar_images


def main(argv: list[str] | None = None) -> dict[str, float]:
    args = parse_args(__doc__, argv)
    labels = list(BAR_CLASSES[:3])
    development = ImageDataset.from_lists(*make_bar_images(16, num_classes=3, seed=args.seed))
    test = ImageDataset.from_lists(*make_bar_images(20, num_classes=3, seed=args.seed + 1))
    # Validation picks the best epoch. It comes out of the development data, never the test set.
    train, validation = stratified_split(development, test_size=0.25, seed=args.seed)

    results = {}
    for mode in ("frozen", "supcon"):
        model = EmbedFitModel.from_pretrained(args.backbone, labels=labels)
        training = TrainingArguments(
            training_mode=mode,
            num_epochs=args.epochs,
            batch_size=12,
            learning_rate=args.learning_rate,
            num_workers=0,
            seed=args.seed,
        )
        trainer = Trainer(model, training, train, eval_dataset=validation)
        output = trainer.train()
        results[mode] = trainer.evaluate(test)["accuracy"]
        if output.train_losses:
            print(
                f"{mode}: loss {output.train_losses[0]:.2f} -> {output.train_losses[-1]:.2f}, "
                f"kept epoch {output.best_epoch} of {args.epochs}"
            )
    print({mode: round(accuracy, 2) for mode, accuracy in results.items()})
    return results


if __name__ == "__main__":
    main()
