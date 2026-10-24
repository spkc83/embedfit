"""Embedding-quality benchmark: how reusable are the embeddings each training mode produces?

``few_shot.py`` scores the classification head. This script scores the embedding geometry on two
tasks that a head trained on the known classes cannot answer by itself:

- ``open_set``: ``--unseen-classes`` classes, chosen per seed, are held out of training. The
  embeddings are then used to retrieve, recognise, and flag images of classes never trained on.
- ``rare_positive``: one class, chosen per seed, is the positive class; the model trains on
  ``shots`` positives and ``--negatives`` negatives from the other classes, and ranks the whole
  evaluated split by the head's positive probability and by similarity to the known positives.

Data, training options (``--interpolate`` among them), resuming, and the environment record are
those of ``few_shot.py``; see
its documentation and ``docs/guides/benchmarking.md`` for every metric.

    python benchmarks/embedding_quality.py --dataset eurosat --evaluate-on validation \\
        --shots 8 --seeds 0 --methods frozen supcon --output .runs/quality.json

Run it from the repository root, and keep run output in the git-ignored ``.runs/`` folder.
"""

import argparse
import logging
import math
import time
from collections.abc import Callable, Hashable, Sequence
from dataclasses import replace
from pathlib import Path

import few_shot
import numpy as np

from embedfit import EmbedFitModel, FewShotEvaluator, ImageDataset, Trainer, TrainingArguments
from embedfit.evaluation import get_metric, recall_at_fpr
from embedfit.retrieval import RetrievalIndex, mean_average_precision, retrieval_precision_at_k
from embedfit.sampling import sample_few_shot

logger = logging.getLogger(__name__)

TASKS = ("open_set", "rare_positive")
POSITIVE, NEGATIVE = "positive", "negative"
TARGET_FPRS = (0.01, 0.05)
MEASURED = (*few_shot.MEASURED, "evaluation_repeats_dropped")  # Known after loading.

Evaluation = Callable[[EmbedFitModel], dict[str, float]]


def partition_classes(
    classes: Sequence[str], unseen: int, *, seed: int
) -> tuple[list[str], list[str]]:
    """Seen and unseen classes, each in the order of ``classes``; they depend only on ``seed``."""
    if unseen < 2:
        raise SystemExit(
            f"--unseen-classes {unseen}: the open_set task needs at least 2 unseen classes."
        )
    if len(classes) - unseen < 2:
        raise SystemExit(
            f"--unseen-classes {unseen} leaves {max(len(classes) - unseen, 0)} of the "
            f"{len(classes)} classes seen; the open_set task needs at least 2 seen classes, so "
            f"at least {unseen + 2} classes."
        )
    held_out = set(np.random.default_rng([seed, 0]).choice(len(classes), unseen, replace=False))
    seen = [name for position, name in enumerate(classes) if position not in held_out]
    return seen, [name for position, name in enumerate(classes) if position in held_out]


def _of_classes(dataset: ImageDataset, classes: Sequence[Hashable]) -> ImageDataset:
    wanted = set(classes)
    return dataset.subset(i for i, label in enumerate(dataset.labels) if label in wanted)


def open_set_sample(
    pool: ImageDataset, classes: Sequence[str], *, unseen: int, shots: int, seed: int
) -> tuple[list[str], ImageDataset, ImageDataset, ImageDataset]:
    """The seen classes, the training sample, the support set, and the held-out reference.

    Each holds ``shots`` pool images per class: the training sample and the held-out reference
    of every seen class, disjoint from each other, and the support set of every unseen class.
    """
    seen, held_out = partition_classes(classes, unseen, seed=seed)
    seen_pool = _of_classes(pool, seen)
    train = sample_few_shot(seen_pool, shots=shots, seed=seed)
    support = sample_few_shot(_of_classes(pool, held_out), shots=shots, seed=seed)
    trained = {id(sample) for sample in train}  # subset shares the pool's sample objects.
    rest = seen_pool.subset(i for i, sample in enumerate(seen_pool) if id(sample) not in trained)
    rng = np.random.default_rng([seed, 2])
    labels = np.asarray(rest.labels, dtype=object)
    chosen: list[int] = []
    for name in seen:
        members = np.flatnonzero(labels == name)
        if shots > len(members):
            raise SystemExit(
                f"The held-out reference needs {shots} more images of a seen class, which has "
                f"{len(members)} left in the pool after the training sample."
            )
        chosen.extend(rng.choice(members, shots, replace=False).tolist())
    return seen, train, support, rest.subset(sorted(chosen))


def rare_positive_sample(
    pool: ImageDataset, classes: Sequence[str], *, shots: int, negatives: int, seed: int
) -> tuple[str, ImageDataset]:
    """The positive class and a training set of ``shots`` positives and ``negatives`` negatives.

    The negatives are spread over the other classes as evenly as possible (counts differ by at
    most one) and depend only on ``seed``, not on ``shots``. Labels are ``positive`` and
    ``negative``.
    """
    rng = np.random.default_rng([seed, 1])
    positive = classes[int(rng.integers(len(classes)))]
    others = [name for name in classes if name != positive]
    quota = np.full(len(others), negatives // len(others))
    quota[rng.choice(len(others), negatives % len(others), replace=False)] += 1
    labels = np.asarray(pool.labels, dtype=object)
    chosen: list[int] = []
    for name, count in zip(others, quota.tolist(), strict=True):
        members = np.flatnonzero(labels == name)
        if count > len(members):
            raise SystemExit(
                f"--negatives asks for {count} images of a class that has {len(members)} in the "
                "pool."
            )
        chosen.extend(rng.choice(members, count, replace=False).tolist())
    positives = sample_few_shot(_of_classes(pool, [positive]), shots=shots, seed=seed)
    images = [*positives.images, *(pool.images[i] for i in sorted(chosen))]
    train = ImageDataset.from_lists(images, [POSITIVE] * shots + [NEGATIVE] * len(chosen))
    return positive, train


def _unit(embeddings: np.ndarray) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float64)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def retrieval_quality(embeddings: np.ndarray, labels: Sequence[Hashable]) -> tuple[float, float]:
    """Leave-one-out mean average precision and precision at 1 by cosine similarity.

    Every item queries all the others; the relevant ones share its label. Items whose label no
    other item has are not scored. NaN for both when no item can be scored.
    """
    by_label: dict[Hashable, set[int]] = {}
    for position, label in enumerate(labels):
        by_label.setdefault(label, set()).add(position)
    queries = [i for i, label in enumerate(labels) if len(by_label[label]) > 1]
    if not queries:
        return math.nan, math.nan
    index = RetrievalIndex("cosine")
    index.fit(np.asarray(embeddings, dtype=np.float32))
    results = index.search(
        np.asarray(embeddings, dtype=np.float32)[queries],
        k=len(labels) - 1,
        query_ids=queries,
        exclude_self=True,
    )
    relevant = [by_label[labels[i]] - {i} for i in queries]
    precision = [
        retrieval_precision_at_k(ranked, wanted, k=1)
        for ranked, wanted in zip(results, relevant, strict=True)
    ]
    return mean_average_precision(results, relevant), float(np.mean(precision))


def max_similarity(queries: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Each query's largest cosine similarity to any reference embedding."""
    return (_unit(queries) @ _unit(reference).T).max(axis=1)


def centroid_accuracy(
    support: np.ndarray,
    support_labels: Sequence[Hashable],
    queries: np.ndarray,
    query_labels: Sequence[Hashable],
) -> float:
    """Accuracy of assigning each query to the class whose mean unit support embedding is the
    most cosine-similar."""
    classes = list(dict.fromkeys(support_labels))
    owners = np.asarray([classes.index(label) for label in support_labels])
    unit = _unit(support)
    centroids = np.stack([unit[owners == c].mean(axis=0) for c in range(len(classes))])
    predicted = (_unit(queries) @ _unit(centroids).T).argmax(axis=1)
    return float(np.mean(np.asarray(classes, dtype=object)[predicted] == list(query_labels)))


def ranking_metrics(is_positive: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """ROC-AUC, PR-AUC, and recall at fixed false-positive rates of a score."""
    metrics = {
        name: get_metric(name)(is_positive, scores, positive_label=True)
        for name in ("roc_auc", "pr_auc")
    }
    for target in TARGET_FPRS:
        metrics[f"recall_at_fpr_{target}"] = recall_at_fpr(
            is_positive, scores, positive_label=True, target_fpr=target
        )
    return metrics


def open_set_evaluation(
    train: ImageDataset,
    support: ImageDataset,
    heldout: ImageDataset,
    evaluation: ImageDataset,
    seen: Sequence[str],
) -> Evaluation:
    """The ``open_set`` metrics of a model trained on ``train``."""
    labels = np.asarray(evaluation.labels, dtype=object)
    is_seen = np.isin(labels, list(seen))
    seen_labels, unseen_labels = labels[is_seen].tolist(), labels[~is_seen].tolist()

    def evaluate(model: EmbedFitModel) -> dict[str, float]:
        embeddings = model.encode(evaluation.images)
        predicted = model.predict(embeddings=embeddings[is_seen])
        metrics = {"seen_accuracy": float(np.mean(predicted == np.asarray(seen_labels)))}
        for prefix, mask, names in (
            ("seen", is_seen, seen_labels),
            ("unseen", ~is_seen, unseen_labels),
        ):
            mean_ap, first = retrieval_quality(embeddings[mask], names)
            metrics[f"{prefix}_map"], metrics[f"{prefix}_precision_at_1"] = mean_ap, first
        metrics["unseen_centroid_accuracy"] = centroid_accuracy(
            model.encode(support.images), support.labels, embeddings[~is_seen], unseen_labels
        )
        for prefix, images in (("novelty", train.images), ("novelty_heldout", heldout.images)):
            novelty = 1 - max_similarity(embeddings, model.encode(images))
            for name in ("roc_auc", "pr_auc"):
                metric = get_metric(name)
                metrics[f"{prefix}_{name}"] = metric(~is_seen, novelty, positive_label=True)
        return metrics

    return evaluate


def rare_positive_evaluation(
    train: ImageDataset, evaluation: ImageDataset, positive: str
) -> Evaluation:
    """The ``rare_positive`` metrics of a model trained on ``train``."""
    is_positive = np.asarray([label == positive for label in evaluation.labels])
    positives = _of_classes(train, [POSITIVE]).images

    def evaluate(model: EmbedFitModel) -> dict[str, float]:
        embeddings = model.encode(evaluation.images)
        assert model.label_mapping is not None
        column = model.label_mapping.index_of(POSITIVE)
        head = model.predict_proba(embeddings=embeddings)[:, column]
        similarity = max_similarity(embeddings, model.encode(positives))
        metrics = ranking_metrics(is_positive, head)
        metrics |= {f"knn_{k}": v for k, v in ranking_metrics(is_positive, similarity).items()}
        metrics["prevalence"] = float(is_positive.mean())
        return metrics

    return evaluate


def drop_repeated_images(dataset: ImageDataset) -> tuple[ImageDataset, int]:
    """Keep the first of every set of images with identical pixels; return how many were dropped.

    A repeated image would retrieve its copy at rank 1 and inflate the retrieval metrics.
    """
    seen: set[str] = set()
    kept = []
    for position, image in enumerate(dataset.images):
        digest = few_shot.digest(image)
        if digest not in seen:
            seen.add(digest)
            kept.append(position)
    return dataset.subset(kept), len(dataset) - len(kept)


def read_previous(output: Path, config: dict) -> dict | None:
    """``few_shot.read_previous``, also requiring a known task in every record."""
    previous = few_shot.read_previous(output, config, measured=MEASURED)
    try:
        known = previous is None or all(record["task"] in TASKS for record in previous["records"])
    except (KeyError, TypeError):
        known = False
    if not known:
        raise SystemExit(f"{output} is not a results file written by this script.")
    return previous


def parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    """The command line, with every check that needs no data; exits on an error."""
    parser = few_shot.training_parser(__doc__.splitlines()[0])
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument(
        "--unseen-classes",
        type=few_shot.positive_int,
        default=3,
        help="open_set: classes held out of training, chosen per seed",
    )
    parser.add_argument(
        "--negatives",
        type=few_shot.positive_int,
        default=200,
        help="rare_positive: negatives in the training set, spread over the other classes",
    )
    args = parser.parse_args(argv)
    few_shot.check_arguments(parser, args)
    if args.unseen_classes < 2:
        parser.error(
            f"--unseen-classes must be at least 2, got {args.unseen_classes}: among a single "
            "unseen class, retrieval and centroid accuracy are 1 by construction"
        )
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("--tasks names a task twice")
    for option, value, default, task in (
        ("--unseen-classes", args.unseen_classes, 3, "open_set"),
        ("--negatives", args.negatives, 200, "rare_positive"),
    ):
        if value != default and task not in args.tasks:
            parser.error(f"{option} applies to the {task} task only, and it is not selected")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)
    rates = few_shot.parse_learning_rates(args.learning_rate, args.default_learning_rate)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    few_shot.prepare_environment(args)
    arguments = few_shot.training_arguments(args, rates)

    # Resumed as in few_shot.py, by (task, shots, seed) group: every sample, class choice, and
    # support set depends only on the task, shots, and seed.
    config = few_shot.resume_configuration(args)
    config["learning_rates"] = {method: rates[method] for method in arguments}
    previous = read_previous(args.output, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = previous["records"] if previous else []
    sessions: list[dict] = previous["environment"]["sessions"] if previous else []
    done = {(r["task"], r["shots"], r["seed"]) for r in records}
    elapsed = sum(entry["elapsed_seconds"] for entry in sessions)

    splits, classes, duplicates = few_shot.load_splits(args)
    pool = splits["pool"]
    evaluation, repeats = drop_repeated_images(splits[args.evaluate_on])
    config["evaluation_duplicates_dropped"] = duplicates
    config["evaluation_repeats_dropped"] = repeats
    config["evaluation_size"] = len(evaluation)
    if previous and previous["config"] != config:
        raise SystemExit(f"{args.output} holds results for different evaluation data.")

    def draw(
        task: str, shots: int, seed: int
    ) -> tuple[list[str], str | None, ImageDataset, Evaluation]:
        """The model's labels and positive label, the training set, and the evaluation."""
        if task == "open_set":
            seen, train, support, heldout = open_set_sample(
                pool, classes, unseen=args.unseen_classes, shots=shots, seed=seed
            )
            evaluate = open_set_evaluation(train, support, heldout, evaluation, seen)
            return seen, None, train, evaluate
        positive, train = rare_positive_sample(
            pool, classes, shots=shots, negatives=args.negatives, seed=seed
        )
        evaluate = rare_positive_evaluation(train, evaluation, positive)
        return [NEGATIVE, POSITIVE], POSITIVE, train, evaluate

    def options(method: str, seed: int) -> TrainingArguments:
        return replace(arguments.get(method, TrainingArguments(training_mode=method)), seed=seed)

    def model_factory(labels: list[str], positive_label: str | None) -> EmbedFitModel:
        return EmbedFitModel.from_pretrained(
            args.backbone,
            labels=labels,
            positive_label=positive_label,
            image_size=args.image_size,
            device=args.device,
        )

    # Fail before any training: the pool must cover the largest shot count and every method
    # must accept the smallest, for every task.
    for task in args.tasks:
        draw(task, max(args.shots), args.seeds[0])
        labels, positive_label, smallest, _ = draw(task, min(args.shots), args.seeds[0])
        for method in args.methods:
            Trainer(model_factory(labels, positive_label), options(method, args.seeds[0]), smallest)

    session = few_shot.new_session(args)
    sessions.append(session)
    for task in args.tasks:
        for shots in args.shots:
            for seed in args.seeds:
                if (task, shots, seed) in done:
                    continue
                started = time.perf_counter()
                labels, positive_label, train, evaluate = draw(task, shots, seed)
                for method in args.methods:
                    model = model_factory(labels, positive_label)
                    trainer = Trainer(model, options(method, seed), train)
                    for alpha in few_shot.trained_weights(trainer, args.interpolate):
                        for metric, value in evaluate(model).items():
                            records.append(
                                {
                                    "task": task,
                                    "method": method,
                                    "alpha": alpha,
                                    "backbone": model.backbone.identifier,
                                    "shots": shots,
                                    "seed": seed,
                                    "metric": metric,
                                    "value": float(value),
                                }
                            )
                    logger.info("Scored %s at %d shots, seed %d", method, shots, seed)
                session["elapsed_seconds"] = round(
                    session["elapsed_seconds"] + time.perf_counter() - started, 1
                )
                elapsed = sum(entry["elapsed_seconds"] for entry in sessions)
                environment = {**session, "elapsed_seconds": elapsed, "sessions": sessions}
                payload = {"config": config, "environment": environment, "records": records}
                few_shot.write_results(args.output, payload)

    print(f"\n{args.evaluate_on} (mean ± std over {len(args.seeds)} seeds), {elapsed / 60:.1f} min")
    for task in args.tasks:
        summary = FewShotEvaluator.aggregate(
            [
                {**r, "method": few_shot.row_name(r["method"], r["alpha"])}
                for r in records
                if r["task"] == task
            ]
        )
        metrics = list(dict.fromkeys(row["metric"] for row in summary))
        print(f"\n{task}")
        print(f"{'metric':26s}{'method':20s}" + "".join(f"{f'{n} shots':>16s}" for n in args.shots))
        for metric in metrics:
            for name in few_shot.row_names(args.methods, args.interpolate):
                cells = [
                    f"{row['mean']:.3f} ± {row['std']:.3f}"
                    + (f" ({row['num_undefined']} undefined)" if row["num_undefined"] else "")
                    for row in summary
                    if row["method"] == name and row["metric"] == metric
                ]
                print(f"{metric:26s}{name:20s}" + "".join(f"{cell:>16s}" for cell in cells))


if __name__ == "__main__":
    main()
