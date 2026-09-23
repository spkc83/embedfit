"""Frozen vs end-to-end cross-entropy vs SupCon at several shot counts over repeated seeds.

python examples/06_few_shot_benchmark.py
"""

from _common import parse_args

from embedfit import EmbedFitModel, FewShotEvaluator, ImageDataset, TrainingArguments
from embedfit.testing import BAR_CLASSES, make_bar_images

METHODS = ["frozen", "end_to_end", "supcon"]


def main(argv: list[str] | None = None, shots=(4, 8, 16), seeds=(1, 2, 3)) -> list[dict]:
    args = parse_args(__doc__, argv, epochs=15)
    labels = list(BAR_CLASSES[:3])
    pool = ImageDataset.from_lists(*make_bar_images(24, num_classes=3, seed=args.seed))
    test = ImageDataset.from_lists(*make_bar_images(20, num_classes=3, seed=args.seed + 1))

    # One budget, learning rate included, keeps this demo short. In a real benchmark choose each
    # method's learning rate on a validation split (docs/guides/benchmarking.md).
    budget = {
        "num_epochs": args.epochs,
        "batch_size": 12,
        "learning_rate": args.learning_rate,
        "num_workers": 0,
    }
    arguments = {mode: TrainingArguments(training_mode=mode, **budget) for mode in METHODS[1:]}

    evaluator = FewShotEvaluator(shots=list(shots), seeds=list(seeds))
    records = evaluator.run(
        pool,
        test_dataset=test,
        methods=METHODS,
        training_arguments=arguments,
        model_factory=lambda: EmbedFitModel.from_pretrained(args.backbone, labels=labels),
    )
    summary = [row for row in evaluator.aggregate(records) if row["metric"] == "accuracy"]
    print(f"{'method':12s}" + "".join(f"{f'{n} shots':>16s}" for n in shots))
    for method in METHODS:
        cells = [
            f"{row['mean']:.2f} ± {row['std']:.2f}" for row in summary if row["method"] == method
        ]
        print(f"{method:12s}" + "".join(f"{cell:>16s}" for cell in cells))
    return summary


if __name__ == "__main__":
    main()
