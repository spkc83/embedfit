"""Few-shot benchmark on EuroSAT (RGB) with a pretrained backbone.

Compares training modes with ``FewShotEvaluator`` on the public EuroSAT land-cover dataset
(10 classes, 64x64 Sentinel-2 RGB patches). The dataset is downloaded once by torchvision into
``--data-root``; nothing from it is written into the repository.

Three disjoint per-class splits are drawn from the full dataset with ``--split-seed``: a
training pool that few-shot samples come from, a validation set for choosing learning rates,
and a test set. Tune with ``--evaluate-on validation``; report with ``--evaluate-on test``.

    python benchmarks/eurosat_few_shot.py --evaluate-on validation --shots 8 --seeds 0 \\
        --methods supcon --learning-rate supcon=1e-5 --output tune.json
"""

import argparse
import json
import logging
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import EuroSAT

import embedfit
from embedfit import EmbedFitModel, FewShotEvaluator, ImageDataset, TrainingArguments

TRAINED_METHODS = ("end_to_end", "supcon", "pair")


def load_splits(
    root: Path, *, pool: int, validation: int, test: int, seed: int
) -> tuple[dict[str, ImageDataset], list[str]]:
    """Draw disjoint per-class pool/validation/test splits and load their images."""
    source = EuroSAT(str(root), download=True)
    targets = np.asarray(source.targets)
    rng = np.random.default_rng(seed)
    chosen: dict[str, list[int]] = {"pool": [], "validation": [], "test": []}
    for label in range(len(source.classes)):
        order = rng.permutation(np.flatnonzero(targets == label))
        cuts = np.cumsum([pool, validation, test])
        if cuts[-1] > len(order):
            raise SystemExit(
                f"Class {label} has {len(order)} images, fewer than the {cuts[-1]} requested."
            )
        for name, part in zip(chosen, np.split(order[: cuts[-1]], cuts[:-1]), strict=True):
            chosen[name].extend(part.tolist())
    splits = {}
    for name, indices in chosen.items():
        images, labels = [], []
        for index in indices:
            image, target = source[index]
            images.append(image.convert("RGB"))
            labels.append(source.classes[target])
        splits[name] = ImageDataset.from_lists(images, labels)
    return splits, list(source.classes)


def parse_learning_rates(values: list[str], default: float) -> dict[str, float]:
    rates = dict.fromkeys(TRAINED_METHODS, default)
    for value in values:
        method, _, rate = value.partition("=")
        if method not in rates or not rate:
            raise SystemExit(
                f"--learning-rate expects METHOD=RATE with METHOD in {TRAINED_METHODS}"
            )
        rates[method] = float(rate)
    return rates


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-root", type=lambda value: Path(value).expanduser(), default="~/.cache/embedfit"
    )
    parser.add_argument("--backbone", default="hf/facebook/dinov2-small")
    parser.add_argument(
        "--image-size", type=int, default=112, help="input side; EuroSAT tiles are 64x64"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["frozen", *TRAINED_METHODS],
        default=["frozen", *TRAINED_METHODS],
    )
    parser.add_argument("--shots", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--learning-rate", nargs="*", default=[], help="per-method rates, e.g. supcon=1e-5"
    )
    parser.add_argument("--default-learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--pairs-per-example",
        type=int,
        default=1,
        help="pair mode: positive and negative partners per sample per epoch",
    )
    parser.add_argument("--evaluate-on", choices=["validation", "test"], required=True)
    parser.add_argument("--pool-per-class", type=int, default=100)
    parser.add_argument("--eval-per-class", type=int, default=100)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True, help="JSON file for the records")
    args = parser.parse_args(argv)
    try:  # Before loading any data, so that a typo fails at once.
        rates = parse_learning_rates(args.learning_rate, args.default_learning_rate)
    except ValueError:
        parser.error("--learning-rate expects METHOD=RATE with a number as RATE")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(args.threads)

    splits, labels = load_splits(
        args.data_root,
        pool=args.pool_per_class,
        validation=args.eval_per_class,
        test=args.eval_per_class,
        seed=args.split_seed,
    )
    budget = {"num_epochs": args.epochs, "batch_size": args.batch_size, "num_workers": 0}
    arguments = {
        method: TrainingArguments(training_mode=method, learning_rate=rates[method], **budget)
        for method in args.methods
        if method in TRAINED_METHODS
    }
    if "pair" in arguments:
        arguments["pair"] = TrainingArguments(
            training_mode="pair",
            learning_rate=rates["pair"],
            positives_per_example=args.pairs_per_example,
            negatives_per_example=args.pairs_per_example,
            **budget,
        )

    evaluator = FewShotEvaluator(shots=args.shots, seeds=args.seeds)
    started = time.perf_counter()
    records = evaluator.run(
        splits["pool"],
        test_dataset=splits[args.evaluate_on],
        methods=args.methods,
        model_factory=lambda: EmbedFitModel.from_pretrained(
            args.backbone, labels=labels, image_size=args.image_size
        ),
        training_arguments=arguments,
    )
    elapsed = time.perf_counter() - started

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in {"data_root", "output"}
    }
    config["learning_rates"] = {method: rates[method] for method in arguments}
    environment = {
        "embedfit": embedfit.__version__,
        "torch": torch.__version__,
        "device": "cpu",
        "threads": args.threads,
        "processor": platform.processor() or platform.machine(),
        "elapsed_seconds": round(elapsed, 1),
    }
    args.output.write_text(
        json.dumps({"config": config, "environment": environment, "records": records}, indent=1)
    )

    summary = evaluator.aggregate(records)
    print(
        f"\n{args.evaluate_on} accuracy (mean ± std over {len(args.seeds)} seeds), "
        f"{elapsed / 60:.1f} min"
    )
    print(f"{'method':12s}" + "".join(f"{f'{n} shots':>16s}" for n in args.shots))
    for method in args.methods:
        cells = [
            f"{row['mean']:.3f} ± {row['std']:.3f}"
            for row in summary
            if row["method"] == method and row["metric"] == "accuracy"
        ]
        print(f"{method:12s}" + "".join(f"{cell:>16s}" for cell in cells))


if __name__ == "__main__":
    main()
