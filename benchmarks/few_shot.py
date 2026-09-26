"""Few-shot benchmark on a public image dataset with a pretrained backbone.

Compares training modes with ``FewShotEvaluator``. Two datasets are supported, both downloaded
on first use into a cache outside the repository; nothing from them is written into it:

- ``eurosat``: EuroSAT RGB land cover (10 classes, 64x64 Sentinel-2 tiles), via torchvision into
  ``--data-root``. It has no official split, so disjoint per-class pool, validation, and test
  sets are drawn with ``--split-seed``.
- ``rvl-cdip``: document images from RVL-CDIP (16 classes), the Hugging Face dataset
  ``jinhybr/rvl_cdip_400_train_val_test`` (400 training, 100 validation, and 100 test images
  per class). Its own splits are kept: the pool is drawn from the training split. Pages are
  shrunk to a shorter side of 512 px when loaded and kept in grayscale, which bounds memory;
  512 px is still larger than the model input.

Few-shot samples come from the pool. Tune with ``--evaluate-on validation`` and report with
``--evaluate-on test``. Only the pool and the evaluated split are loaded. An evaluation image
whose pixels exactly match a pool image is dropped, and the count is recorded.

    python benchmarks/few_shot.py --dataset eurosat --evaluate-on validation --shots 8 \\
        --seeds 0 --methods supcon --learning-rate supcon=1e-5 --output tune.json
"""

import argparse
import hashlib
import json
import logging
import platform
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import embedfit
from embedfit import EmbedFitModel, FewShotEvaluator, ImageDataset, TrainingArguments

TRAINED_METHODS = ("end_to_end", "supcon", "pair")
DOCUMENT_SHORT_SIDE = 512
DEFAULT_IMAGE_SIZE = {"eurosat": 112, "rvl-cdip": 224}

Split = list[tuple[Image.Image, str]]


def _per_class(labels: np.ndarray, count: int, rng: np.random.Generator) -> list[int]:
    """Up to ``count`` random positions of every class, in class order."""
    chosen = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        if count > len(members):
            raise SystemExit(f"Class {label} has {len(members)} images, fewer than {count}.")
        chosen.extend(rng.permutation(members)[:count].tolist())
    return chosen


def eurosat_splits(
    root: Path, *, pool: int, evaluation: int, evaluate_on: str, seed: int
) -> tuple[dict[str, Split], list[str]]:
    from torchvision.datasets import EuroSAT

    source = EuroSAT(str(root), download=True)
    targets = np.asarray(source.targets)
    rng = np.random.default_rng(seed)
    chosen: dict[str, list[int]] = {"pool": [], "validation": [], "test": []}
    for label in range(len(source.classes)):
        order = rng.permutation(np.flatnonzero(targets == label))
        cuts = np.cumsum([pool, evaluation, evaluation])
        if cuts[-1] > len(order):
            raise SystemExit(
                f"Class {label} has {len(order)} images, fewer than the {cuts[-1]} requested."
            )
        for name, part in zip(chosen, np.split(order[: cuts[-1]], cuts[:-1]), strict=True):
            chosen[name].extend(part.tolist())
    splits = {
        name: [
            (source[index][0].convert("RGB"), source.classes[source[index][1]])
            for index in chosen[name]
        ]
        for name in ("pool", evaluate_on)
    }
    return splits, list(source.classes)


def rvl_cdip_splits(
    root: Path, *, pool: int, evaluation: int, evaluate_on: str, seed: int
) -> tuple[dict[str, Split], list[str]]:
    from datasets import load_dataset

    source = load_dataset("jinhybr/rvl_cdip_400_train_val_test")
    names = source["train"].features["label"].names

    def take(split: str, count: int, stream: int) -> Split:
        data = source[split]
        rng = np.random.default_rng([seed, stream])  # Independent of which splits are loaded.
        rows = data.select(_per_class(np.asarray(data["label"]), count, rng))
        return [(_shrink(row["image"]), names[row["label"]]) for row in rows]

    splits = {"pool": take("train", pool, 0)}
    splits[evaluate_on] = take(evaluate_on, evaluation, 1 if evaluate_on == "validation" else 2)
    return splits, names


def _shrink(image: Image.Image) -> Image.Image:
    scale = DOCUMENT_SHORT_SIDE / min(image.size)
    if scale < 1:
        size = (round(image.width * scale), round(image.height * scale))
        image = image.resize(size, Image.Resampling.BICUBIC)
    return image.convert("L")


LOADERS: dict[str, Callable[..., tuple[dict[str, Split], list[str]]]] = {
    "eurosat": eurosat_splits,
    "rvl-cdip": rvl_cdip_splits,
}


def _digest(image: Image.Image) -> str:
    return hashlib.sha256(f"{image.mode}{image.size}".encode() + image.tobytes()).hexdigest()


def drop_pool_duplicates(splits: dict[str, Split], evaluate_on: str) -> int:
    """Remove evaluation images whose pixels match a pool image; return how many."""
    pool = {_digest(image) for image, _ in splits["pool"]}
    kept = [(image, label) for image, label in splits[evaluate_on] if _digest(image) not in pool]
    dropped = len(splits[evaluate_on]) - len(kept)
    splits[evaluate_on] = kept
    return dropped


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
    parser.add_argument("--dataset", choices=sorted(LOADERS), required=True)
    parser.add_argument("--backbone", default="hf/facebook/dinov2-small")
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help=f"model input side; defaults per dataset: {DEFAULT_IMAGE_SIZE}",
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
    parser.add_argument(
        "--train-mode-batch-norm",
        action="store_true",
        help="update batch-norm statistics during training (freeze_batch_norm=False, ADR 0013)",
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
    if args.image_size is None:
        args.image_size = DEFAULT_IMAGE_SIZE[args.dataset]

    loaded, labels = LOADERS[args.dataset](
        args.data_root,
        pool=args.pool_per_class,
        evaluation=args.eval_per_class,
        evaluate_on=args.evaluate_on,
        seed=args.split_seed,
    )
    duplicates = drop_pool_duplicates(loaded, args.evaluate_on)
    splits = {
        name: ImageDataset.from_lists([image for image, _ in items], [label for _, label in items])
        for name, items in loaded.items()
    }
    budget = {
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": 0,
        "freeze_batch_norm": not args.train_mode_batch_norm,
    }
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

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in {"data_root", "output"}
    }
    config["learning_rates"] = {method: rates[method] for method in arguments}
    config["freeze_batch_norm"] = not args.train_mode_batch_norm
    config["evaluation_duplicates_dropped"] = duplicates
    config["evaluation_size"] = len(splits[args.evaluate_on])

    # Results are written after every (shots, seed) group, and a rerun with the same output
    # file and configuration skips the groups already there. Each group's few-shot sample
    # depends only on its shots and seed, so a resumed run gives the same records.
    records: list[dict] = []
    elapsed = 0.0
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        if previous["config"] != config:
            raise SystemExit(f"{args.output} holds results for a different configuration.")
        records = previous["records"]
        elapsed = previous["environment"]["elapsed_seconds"]
    done = {(r["shots"], r["seed"]) for r in records}

    def model_factory() -> EmbedFitModel:
        return EmbedFitModel.from_pretrained(
            args.backbone, labels=labels, image_size=args.image_size
        )

    for shots in args.shots:
        for seed in args.seeds:
            if (shots, seed) in done:
                continue
            started = time.perf_counter()
            records += FewShotEvaluator(shots=[shots], seeds=[seed]).run(
                splits["pool"],
                test_dataset=splits[args.evaluate_on],
                methods=args.methods,
                model_factory=model_factory,
                training_arguments=arguments,
            )
            elapsed += time.perf_counter() - started
            environment = {
                "embedfit": embedfit.__version__,
                "torch": torch.__version__,
                "device": "cpu",
                "threads": args.threads,
                "processor": platform.processor() or platform.machine(),
                "elapsed_seconds": round(elapsed, 1),
            }
            payload = {"config": config, "environment": environment, "records": records}
            staging = args.output.with_suffix(".partial")
            staging.write_text(json.dumps(payload, indent=1))
            staging.replace(args.output)  # A crash mid-write never corrupts the results.

    evaluator = FewShotEvaluator(shots=args.shots, seeds=args.seeds)
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
