"""benchmarks/embedding_quality.py: task samples, embedding metrics, flags, and resuming."""

import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from embedfit import ImageDataset
from embedfit.exceptions import ConfigurationError
from embedfit.testing import BAR_CLASSES, make_bar_images

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"
CLASSES = list(BAR_CLASSES)


@pytest.fixture(scope="module")
def quality():
    sys.path.insert(0, str(BENCHMARKS))  # The script imports few_shot from its own folder.
    try:
        spec = importlib.util.spec_from_file_location(
            "embedding_quality_benchmark", BENCHMARKS / "embedding_quality.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(BENCHMARKS))
    return module


def bars(per_class, seed):
    return make_bar_images(per_class, num_classes=len(CLASSES), image_size=16, seed=seed)


@pytest.fixture(scope="module")
def pool():
    return ImageDataset.from_lists(*bars(12, seed=0))


def counts(dataset):
    return Counter(dataset.labels)


# Class partition and task samples.


def test_the_partition_is_disjoint_complete_and_seeded(quality):
    seen, unseen = quality.partition_classes(CLASSES, 2, seed=3)
    assert len(unseen) == 2 and len(seen) == 2
    assert set(seen) | set(unseen) == set(CLASSES) and not set(seen) & set(unseen)
    assert seen == [name for name in CLASSES if name in seen]  # Class order is kept.
    assert quality.partition_classes(CLASSES, 2, seed=3) == (seen, unseen)
    assert len({tuple(quality.partition_classes(CLASSES, 2, seed=s)[1]) for s in range(20)}) > 1


def test_the_partition_leaves_two_seen_classes(quality):
    with pytest.raises(SystemExit, match="at least 2 seen classes"):
        quality.partition_classes(CLASSES, 3, seed=0)


def test_the_partition_holds_out_two_classes_or_more(quality):
    # With one unseen class, retrieval and centroid accuracy among them are 1 by construction.
    with pytest.raises(SystemExit, match="at least 2 unseen classes"):
        quality.partition_classes(CLASSES, 1, seed=0)


def test_the_open_set_sample_trains_on_seen_classes_only(quality, pool):
    seen, train, support, heldout = quality.open_set_sample(
        pool, CLASSES, unseen=2, shots=3, seed=1
    )
    unseen = [name for name in CLASSES if name not in seen]
    assert counts(train) == dict.fromkeys(seen, 3)
    assert counts(support) == dict.fromkeys(unseen, 3)
    assert counts(heldout) == dict.fromkeys(seen, 3)
    again = quality.open_set_sample(pool, CLASSES, unseen=2, shots=3, seed=1)
    assert again[0] == seen
    for first, second in zip((train, support, heldout), again[1:], strict=True):
        assert first.images == second.images
    pool_ids = {id(image) for image in pool.images}
    used = [id(image) for image in train.images + support.images + heldout.images]
    assert set(used) <= pool_ids and len(set(used)) == len(used)  # All three are disjoint.


def test_the_heldout_reference_is_seeded_and_spans_the_pool(quality, pool):
    draws = {
        tuple(map(id, quality.open_set_sample(pool, CLASSES, unseen=2, shots=3, seed=s)[3].images))
        for s in range(5)
    }
    assert len(draws) > 1


def test_the_heldout_reference_needs_twice_the_shots(quality, pool):
    with pytest.raises(SystemExit, match="held-out"):
        quality.open_set_sample(pool, CLASSES, unseen=2, shots=7, seed=0)


def test_the_unseen_classes_do_not_depend_on_the_shots(quality, pool):
    assert (
        quality.open_set_sample(pool, CLASSES, unseen=2, shots=2, seed=4)[0]
        == quality.open_set_sample(pool, CLASSES, unseen=2, shots=5, seed=4)[0]
    )


def test_the_rare_positive_sample_spreads_negatives_evenly(quality, pool):
    positive, train = quality.rare_positive_sample(pool, CLASSES, shots=2, negatives=7, seed=5)
    assert counts(train) == {"positive": 2, "negative": 7}
    source = {id(image): label for image, label in zip(pool.images, pool.labels, strict=True)}
    labels = [source[id(image)] for image in train.images]
    assert len(set(map(id, train.images))) == len(train)
    assert [labels[i] == positive for i in range(len(train))] == [
        label == "positive" for label in train.labels
    ]
    negatives = Counter(label for label in labels if label != positive)
    assert set(negatives) == set(CLASSES) - {positive}
    assert sorted(negatives.values()) == [2, 2, 3]
    again = quality.rare_positive_sample(pool, CLASSES, shots=2, negatives=7, seed=5)
    assert again[0] == positive and again[1].images == train.images


def test_the_negatives_do_not_depend_on_the_shots(quality, pool):
    def negatives(shots):
        _, train = quality.rare_positive_sample(pool, CLASSES, shots=shots, negatives=6, seed=2)
        return [
            id(i)
            for i, label in zip(train.images, train.labels, strict=True)
            if label == "negative"
        ]

    assert negatives(2) == negatives(4)


def test_too_many_negatives_is_an_error(quality, pool):
    with pytest.raises(SystemExit, match="negatives"):
        quality.rare_positive_sample(pool, CLASSES, shots=2, negatives=40, seed=0)


# Metrics on hand-made embeddings.


def test_leave_one_out_retrieval_on_separated_classes(quality):
    embeddings = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=np.float32)
    assert quality.retrieval_quality(embeddings, ["a", "a", "b", "b"]) == (1.0, 1.0)


def test_leave_one_out_retrieval_drops_queries_without_a_relevant_item(quality):
    embeddings = np.array([[1, 0], [0, 1], [0.9, 0.1]], dtype=np.float32)
    # Each "a" finds the "b" first and the other "a" second: AP 1/2, precision at 1 is 0. The
    # single "b" has nothing relevant and is not scored.
    assert quality.retrieval_quality(embeddings, ["a", "a", "b"]) == (0.5, 0.0)


def test_retrieval_without_any_scorable_query_is_nan(quality):
    embeddings = np.array([[1, 0], [0, 1]], dtype=np.float32)
    assert all(math.isnan(value) for value in quality.retrieval_quality(embeddings, ["a", "b"]))


def test_max_similarity_is_the_nearest_reference_cosine(quality):
    reference = np.array([[2, 0], [0, 3]], dtype=np.float32)
    queries = np.array([[1, 0], [1, 1], [-1, 0]], dtype=np.float32)
    expected = [1.0, 1 / math.sqrt(2), 0.0]
    assert quality.max_similarity(queries, reference) == pytest.approx(expected, abs=1e-6)


def test_centroid_accuracy(quality):
    support = np.array([[1, 0], [1, 0.2], [0, 1], [0.2, 1]], dtype=np.float32)
    queries = np.array([[1, 0.1], [0.1, 1], [1, 0.05]], dtype=np.float32)
    accuracy = quality.centroid_accuracy(support, ["a", "a", "b", "b"], queries, ["a", "b", "b"])
    assert accuracy == pytest.approx(2 / 3)


def test_ranking_metrics_of_a_perfect_and_an_inverted_score(quality):
    is_positive = np.array([True, False, False, False, True])
    perfect = quality.ranking_metrics(is_positive, np.array([0.9, 0.1, 0.2, 0.3, 0.8]))
    assert perfect == {
        "roc_auc": 1.0,
        "pr_auc": 1.0,
        "recall_at_fpr_0.01": 1.0,
        "recall_at_fpr_0.05": 1.0,
    }
    inverted = quality.ranking_metrics(is_positive, np.array([0.1, 0.9, 0.8, 0.7, 0.2]))
    assert inverted["roc_auc"] == 0.0 and inverted["recall_at_fpr_0.01"] == 0.0


# Both evaluations on a stub model whose embeddings separate the classes perfectly.


class StubModel:
    """Embeds the image "<class><n>" as the one-hot vector of its class, plus a small offset."""

    def __init__(self, classes, labels=None):
        self.classes = list(classes)
        self.labels = labels
        self.label_mapping = self

    def index_of(self, label):
        return self.labels.index(label)

    def encode(self, images):
        rows = []
        for image in images:
            row = np.full(len(self.classes), 0.01)
            row[self.classes.index(image[0])] = 1.0
            row[-1] += int(image[1:]) * 1e-3  # Distinct images, same direction.
            rows.append(row)
        return np.asarray(rows, dtype=np.float32)

    def predict(self, *, embeddings):
        return np.asarray(self.labels, dtype=object)[embeddings[:, : len(self.labels)].argmax(1)]

    def predict_proba(self, *, embeddings):
        positive = embeddings[:, self.classes.index("p")]
        return np.stack([1 - positive, positive], axis=1)


def dataset(names):
    return ImageDataset.from_lists(names, [name[0] for name in names])


def test_open_set_metrics_of_separated_embeddings_are_perfect(quality):
    seen = ["a", "b"]
    train, heldout = dataset(["a1", "a2", "b1", "b2"]), dataset(["a3", "a4", "b3", "b4"])
    support = dataset(["c1", "c2", "d1", "d2"])
    evaluation = dataset([f"{name}{n}" for name in "abcd" for n in range(5, 8)])
    evaluate = quality.open_set_evaluation(train, support, heldout, evaluation, seen)
    metrics = evaluate(StubModel("abcd", labels=seen))
    assert set(metrics) == OPEN_SET_METRICS
    assert all(value == pytest.approx(1.0) for value in metrics.values()), metrics


def test_rare_positive_metrics_of_separated_embeddings_are_perfect(quality):
    train = ImageDataset.from_lists(["p1", "p2", "n1", "m1"], ["positive"] * 2 + ["negative"] * 2)
    evaluation = dataset([f"{name}{n}" for name in "pnm" for n in range(3, 9)])
    evaluate = quality.rare_positive_evaluation(train, evaluation, "p")
    metrics = evaluate(StubModel("pnm", labels=["negative", "positive"]))
    assert metrics.pop("prevalence") == pytest.approx(1 / 3)
    assert set(metrics) == RARE_POSITIVE_METRICS - {"prevalence"}
    assert all(value == pytest.approx(1.0) for value in metrics.values()), metrics


def test_repeated_evaluation_images_are_dropped_after_the_first(quality):
    images, labels = bars(3, seed=0)
    repeated = list(
        zip(
            [*images, images[1], images[4], images[1]],
            [*labels, labels[1], labels[4], labels[1]],
            strict=True,
        )
    )
    kept, dropped = quality.few_shot.drop_repeats(repeated)
    assert dropped == 3
    assert kept == list(zip(images, labels, strict=True))


# The command line.

REQUIRED = ["--dataset", "eurosat", "--evaluate-on", "validation", "--output", "results.json"]


@pytest.mark.parametrize("option", ["--unseen-classes", "--negatives"])
@pytest.mark.parametrize("value", ["0", "-1", "x"])
def test_task_counts_must_be_positive_ints(quality, option, value, capsys):
    with pytest.raises(SystemExit):
        quality.parse_arguments([*REQUIRED, option, value])
    assert f"{option}: expected a positive int" in capsys.readouterr().err


def test_an_unknown_task_is_rejected(quality, capsys):
    with pytest.raises(SystemExit):
        quality.parse_arguments([*REQUIRED, "--tasks", "closed_set"])
    assert "--tasks: invalid choice" in capsys.readouterr().err


def test_a_task_is_named_once(quality, capsys):
    with pytest.raises(SystemExit):
        quality.parse_arguments([*REQUIRED, "--tasks", "open_set", "open_set"])
    assert "--tasks names a task twice" in capsys.readouterr().err


def test_one_unseen_class_is_rejected(quality, capsys):
    with pytest.raises(SystemExit):
        quality.parse_arguments([*REQUIRED, "--unseen-classes", "1"])
    assert "--unseen-classes must be at least 2" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("task", "option"),
    [("rare_positive", ["--unseen-classes", "2"]), ("open_set", ["--negatives", "50"])],
)
def test_task_options_need_their_task(quality, task, option, capsys):
    with pytest.raises(SystemExit):
        quality.parse_arguments([*REQUIRED, "--tasks", task, *option])
    assert "applies to the" in capsys.readouterr().err


def test_the_training_options_are_checked_as_in_few_shot(quality, capsys):
    with pytest.raises(SystemExit):
        quality.parse_arguments([*REQUIRED, "--methods", "frozen", "--max-steps", "5"])
    assert "applies to trained methods only" in capsys.readouterr().err


# Tiny runs on synthetic images with the dummy backbone.


def synthetic_loader(root, *, pool, evaluation, evaluate_on, seed):
    pool_images, pool_labels = bars(pool, seed=10 + seed)
    eval_images, eval_labels = bars(evaluation, seed=20 + seed)
    splits = {
        "pool": list(zip(pool_images, pool_labels, strict=True)),
        evaluate_on: list(zip(eval_images, eval_labels, strict=True)),
    }
    return splits, list(CLASSES)


@pytest.fixture
def run(quality, monkeypatch, tmp_path):
    monkeypatch.setitem(quality.few_shot.LOADERS, "eurosat", synthetic_loader)
    for name in ("HF_HOME", "TORCH_HOME", "TMPDIR"):  # The script only sets unset ones.
        monkeypatch.setenv(name, str(tmp_path))
    threads = quality.few_shot.torch.get_num_threads()

    def call(*options, output="results.json"):
        path = tmp_path / output
        quality.main(
            [
                "--dataset", "eurosat", "--evaluate-on", "validation", "--output", str(path),
                "--data-root", str(tmp_path / "data"), "--backbone", "dummy/tiny",
                "--image-size", "16", "--pool-per-class", "10", "--eval-per-class", "4",
                "--shots", "2", "--seeds", "0", "--threads", "1", *options,
            ]
        )  # fmt: skip
        return json.loads(path.read_text())

    yield call
    quality.few_shot.torch.set_num_threads(threads)


OPEN_SET_METRICS = {
    "novelty_heldout_roc_auc",
    "novelty_heldout_pr_auc",
    "seen_accuracy",
    "seen_map",
    "seen_precision_at_1",
    "unseen_map",
    "unseen_precision_at_1",
    "unseen_centroid_accuracy",
    "novelty_roc_auc",
    "novelty_pr_auc",
}
RANKING = {"roc_auc", "pr_auc", "recall_at_fpr_0.01", "recall_at_fpr_0.05"}
RARE_POSITIVE_METRICS = RANKING | {f"knn_{name}" for name in RANKING} | {"prevalence"}
METHODS = ["frozen", "end_to_end", "supcon", "pair"]


def test_a_tiny_run_of_both_tasks_writes_tidy_records(run, capsys):
    results = run(
        *("--negatives", "6", "--unseen-classes", "2", "--batch-size", "8", "--max-steps", "2"),
        *("--default-learning-rate", "1e-2"),
    )
    records = results["records"]
    keys = {"task", "method", "alpha", "backbone", "shots", "seed", "metric", "value"}
    assert all(set(record) == keys for record in records)
    assert {r["alpha"] for r in records} == {None}
    for task, metrics in (("open_set", OPEN_SET_METRICS), ("rare_positive", RARE_POSITIVE_METRICS)):
        found = {(r["method"], r["metric"]) for r in records if r["task"] == task}
        assert found == {(method, metric) for method in METHODS for metric in metrics}
    assert all(0 <= r["value"] <= 1 for r in records if not math.isnan(r["value"]))
    prevalence = {r["value"] for r in records if r["metric"] == "prevalence"}
    assert prevalence == {0.25}  # One class of four, four evaluation images each.
    assert results["config"]["tasks"] == ["open_set", "rare_positive"]
    assert results["config"]["evaluation_size"] == 16
    assert len(results["environment"]["sessions"]) == 1
    printed = capsys.readouterr().out
    assert "open_set" in printed and "rare_positive" in printed and "unseen_map" in printed


def test_a_rerun_resumes_without_training(run, quality, monkeypatch):
    first = run("--methods", "frozen", "--tasks", "rare_positive", "--negatives", "6")

    def refuse(*args, **kwargs):
        raise AssertionError("a finished group was trained again")

    monkeypatch.setattr(quality.Trainer, "train", refuse)
    second = run("--methods", "frozen", "--tasks", "rare_positive", "--negatives", "6")
    assert second == first  # Nothing left to run, so nothing is written.


def test_a_rerun_with_another_task_option_is_refused(run):
    run("--methods", "frozen", "--tasks", "rare_positive", "--negatives", "6")
    with pytest.raises(SystemExit, match="different configuration"):
        run("--methods", "frozen", "--tasks", "rare_positive", "--negatives", "5")


def test_too_many_unseen_classes_fail_before_training(run, quality, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("trained before the check")

    monkeypatch.setattr(quality.Trainer, "train", refuse)
    with pytest.raises(SystemExit, match="at least 2 seen classes"):
        run("--methods", "frozen", "--tasks", "open_set", "--unseen-classes", "3")


def test_a_rerun_trains_only_the_missing_groups(run, quality, monkeypatch, tmp_path):
    options = ("--methods", "frozen", "--negatives", "6", "--unseen-classes", "2")
    first = run(*options)
    partial = {**first, "records": [r for r in first["records"] if r["task"] == "open_set"]}
    (tmp_path / "results.json").write_text(json.dumps(partial))
    trained = []
    train = quality.Trainer.train

    def counting(self, *args, **kwargs):
        trained.append(self.model.labels)
        return train(self, *args, **kwargs)

    monkeypatch.setattr(quality.Trainer, "train", counting)
    second = run(*options)
    assert len(trained) == 1 and list(trained[0]) == ["negative", "positive"]
    assert second["records"] == first["records"]
    assert len(second["environment"]["sessions"]) == 2


def test_supcon_needs_two_positives_and_fails_before_training(run, quality, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("trained before the check")

    monkeypatch.setattr(quality.Trainer, "train", refuse)
    options = ("--tasks", "rare_positive", "--negatives", "6", "--shots", "1")
    with pytest.raises(ConfigurationError, match="at least 2 samples"):
        run("--methods", "frozen", "supcon", *options)


def test_pair_trains_on_a_single_positive(run):
    options = ("--tasks", "rare_positive", "--negatives", "6", "--shots", "1")
    records = run("--methods", "pair", *options)["records"]
    assert {r["metric"] for r in records} == RARE_POSITIVE_METRICS


def test_repeated_evaluation_images_are_dropped_and_counted(run, quality, monkeypatch):
    def repeating_loader(root, **kwargs):
        splits, classes = synthetic_loader(root, **kwargs)
        evaluated = splits[kwargs["evaluate_on"]]
        evaluated += [evaluated[0], evaluated[5]]
        return splits, classes

    monkeypatch.setitem(quality.few_shot.LOADERS, "eurosat", repeating_loader)
    config = run("--methods", "frozen", "--tasks", "rare_positive", "--negatives", "6")["config"]
    assert config["evaluation_repeats_dropped"] == 2
    assert config["evaluation_size"] == 16
    resumed = run("--methods", "frozen", "--tasks", "rare_positive", "--negatives", "6")
    assert resumed["config"] == config


@pytest.mark.parametrize("task", [None, "closed_set"])
def test_a_record_without_a_known_task_is_not_a_results_file(run, tmp_path, task):
    options = ("--methods", "frozen", "--tasks", "rare_positive", "--negatives", "6")
    results = run(*options)
    for record in results["records"]:
        if task is None:
            del record["task"]
        else:
            record["task"] = task
    (tmp_path / "results.json").write_text(json.dumps(results))
    with pytest.raises(SystemExit, match="not a results file"):
        run(*options)


def test_the_summary_counts_undefined_runs(run, quality, monkeypatch, capsys):
    monkeypatch.setattr(quality, "retrieval_quality", lambda *args: (math.nan, math.nan))
    run("--methods", "frozen", "--tasks", "open_set", "--unseen-classes", "2")
    printed = capsys.readouterr().out.splitlines()
    lines = [line for line in printed if "seen_map" in line]
    assert len(lines) == 2 and all("(1 undefined)" in line for line in lines)
    accuracy = [line for line in printed if "accuracy" in line]
    assert accuracy and all("undefined" not in line for line in accuracy)


def test_interpolation_at_zero_is_the_frozen_baseline(run, capsys):
    results = run(
        *("--methods", "frozen", "supcon", "--interpolate", "0", "0.5", "--max-steps", "2"),
        *("--negatives", "6", "--unseen-classes", "2", "--batch-size", "8"),
        *("--default-learning-rate", "1e-2"),
    )

    def scores(method, alpha):
        return {
            (r["task"], r["metric"]): r["value"]
            for r in results["records"]
            if (r["method"], r["alpha"]) == (method, alpha)
        }

    frozen = scores("frozen", None)
    assert len(frozen) == len(OPEN_SET_METRICS) + len(RARE_POSITIVE_METRICS)
    interpolated = scores("supcon", 0.0)
    assert interpolated.keys() == frozen.keys()
    for key, value in frozen.items():
        assert value == interpolated[key] or (math.isnan(value) and math.isnan(interpolated[key]))
    assert scores("supcon", 0.5).keys() == frozen.keys()
    trained = scores("supcon", None)
    assert any(trained[key] != value for key, value in frozen.items() if not math.isnan(value))
    assert results["config"]["interpolate"] == [0.0, 0.5]
    printed = capsys.readouterr().out
    assert "supcon α=0.5" in printed
