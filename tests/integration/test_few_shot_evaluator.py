"""FewShotEvaluator (acceptance criterion I)."""

import math

import pytest

from embedfit import EmbedFitModel, FewShotEvaluator, ImageDataset, TrainingArguments
from embedfit.exceptions import ConfigurationError
from embedfit.testing import BAR_CLASSES, make_bar_images

LABELS = list(BAR_CLASSES[:3])
KEYS = {"method", "backbone", "shots", "seed", "metric", "value"}


def factory():
    return EmbedFitModel.from_pretrained("dummy/tiny", labels=LABELS)


@pytest.fixture(scope="module")
def pool():
    return ImageDataset.from_lists(*make_bar_images(16, num_classes=3, seed=0))


@pytest.fixture(scope="module")
def test_set():
    return ImageDataset.from_lists(*make_bar_images(8, num_classes=3, seed=1))


@pytest.fixture(scope="module")
def records(pool, test_set):
    fast = {"num_epochs": 6, "batch_size": 12, "learning_rate": 1e-2, "num_workers": 0}
    arguments = {"supcon": TrainingArguments(training_mode="supcon", **fast)}
    evaluator = FewShotEvaluator(shots=[4, 8], seeds=[1, 2])
    return evaluator.run(
        pool,
        test_dataset=test_set,
        methods=["frozen", "supcon"],
        model_factory=factory,
        training_arguments=arguments,
    )


def test_records_are_tidy_and_complete(records):
    assert all(set(record) == KEYS for record in records)
    combinations = {(r["method"], r["shots"], r["seed"]) for r in records}
    assert combinations == {(m, s, d) for m in ("frozen", "supcon") for s in (4, 8) for d in (1, 2)}
    metrics = {r["metric"] for r in records}
    assert {"accuracy", "balanced_accuracy", "macro_f1"} <= metrics
    assert len(records) == len(combinations) * len(metrics)
    assert {r["backbone"] for r in records} == {"dummy/tiny"}
    assert all(isinstance(r["value"], float) and 0 <= r["value"] <= 1 for r in records)


def test_representation_training_helps_on_this_task(records):
    accuracy = {
        (r["method"], r["shots"], r["seed"]): r["value"]
        for r in records
        if r["metric"] == "accuracy"
    }
    supcon = [accuracy["supcon", 8, seed] for seed in (1, 2)]
    frozen = [accuracy["frozen", 8, seed] for seed in (1, 2)]
    assert sum(supcon) / 2 > sum(frozen) / 2 + 0.1


def test_aggregate_hand_computed():
    rows = FewShotEvaluator.aggregate(
        [
            {
                "method": "m",
                "backbone": "b",
                "shots": 4,
                "seed": s,
                "metric": "accuracy",
                "value": v,
            }
            for s, v in ((1, 0.5), (2, 0.7), (3, 0.9), (4, float("nan")))
        ]
        + [
            {
                "method": "m",
                "backbone": "b",
                "shots": 8,
                "seed": 1,
                "metric": "accuracy",
                "value": 1.0,
            }
        ]
    )
    assert rows[0] == {
        "method": "m",
        "backbone": "b",
        "shots": 4,
        "metric": "accuracy",
        "mean": pytest.approx(0.7),
        "std": pytest.approx(0.2),
        "num_runs": 3,
        "num_undefined": 1,
    }
    assert (rows[1]["mean"], rows[1]["std"], rows[1]["num_runs"]) == (1.0, 0.0, 1)
    undefined = FewShotEvaluator.aggregate(
        [
            {
                "method": "m",
                "backbone": "b",
                "shots": 4,
                "seed": 1,
                "metric": "pr_auc",
                "value": float("nan"),
            }
        ]
    )
    assert math.isnan(undefined[0]["mean"]) and undefined[0]["num_undefined"] == 1
    with pytest.raises(ConfigurationError, match="keys method, backbone"):
        FewShotEvaluator.aggregate([{"method": "m"}])


def test_aggregate_of_a_real_run(records):
    rows = FewShotEvaluator.aggregate(records)
    assert len(rows) == len(records) // 2  # Two seeds per group.
    assert all(row["num_runs"] == 2 and row["std"] >= 0 for row in rows)


def test_every_method_trains_on_the_same_few_shot_sample_and_never_on_test_data(
    pool, test_set, monkeypatch
):
    import embedfit.benchmark as benchmark

    seen = []
    original = benchmark.Trainer

    class SpyTrainer(original):
        def __init__(self, model, args, train_dataset, *a, **k):
            seen.append((args.training_mode, args.seed, tuple(id(s) for s in train_dataset)))
            super().__init__(model, args, train_dataset, *a, **k)

    monkeypatch.setattr(benchmark, "Trainer", SpyTrainer)
    FewShotEvaluator(shots=[2], seeds=[5, 6]).run(
        pool,
        test_dataset=test_set,
        methods=["frozen", "end_to_end"],
        model_factory=factory,
        training_arguments={
            "end_to_end": TrainingArguments(
                training_mode="end_to_end", num_epochs=1, batch_size=6, num_workers=0
            )
        },
    )
    by_seed = {}
    for _, seed, samples in seen:
        by_seed.setdefault(seed, set()).add(samples)
        assert len(samples) == 6  # 2 shots x 3 classes.
        assert not set(samples) & {id(s) for s in test_set}
    assert all(len(samples) == 1 for samples in by_seed.values())  # Same draw for each method.
    assert by_seed[5] != by_seed[6]
    assert sorted({seed for _, seed, _ in seen}) == [5, 6]  # The run's seed reaches training.


@pytest.mark.parametrize(
    "shots, seeds, match",
    [
        ([], [1], "shots"),
        ([4, 4], [1], "duplicates"),
        ([0], [1], "shots"),
        ([4], [], "seeds"),
        ([4], [-1], "seeds"),
        ([4.0], [1], "shots"),
        ("48", [1], "shots"),
    ],
)
def test_invalid_evaluator_arguments(shots, seeds, match):
    with pytest.raises(ConfigurationError, match=match):
        FewShotEvaluator(shots=shots, seeds=seeds)


def test_invalid_run_arguments(pool, test_set):
    evaluator = FewShotEvaluator(shots=[4], seeds=[1])

    def run(**overrides):
        options = {
            "dataset": pool,
            "test_dataset": test_set,
            "methods": ["frozen"],
            "model_factory": factory,
            **overrides,
        }
        return evaluator.run(**options)

    with pytest.raises(ConfigurationError, match="distinct training modes"):
        run(methods=["frozen", "magic"])
    with pytest.raises(ConfigurationError, match="distinct training modes"):
        run(methods=["frozen", "frozen"])
    with pytest.raises(ConfigurationError, match="non-empty sequence"):
        run(methods=[])
    with pytest.raises(ConfigurationError, match="test_dataset must be a non-empty ImageDataset"):
        run(test_dataset=ImageDataset([]))
    with pytest.raises(ConfigurationError, match="has training_mode 'supcon'"):
        run(training_arguments={"frozen": TrainingArguments(training_mode="supcon")})
    shared = factory()
    with pytest.raises(ConfigurationError, match="new EmbedFitModel on every call"):
        FewShotEvaluator(shots=[4], seeds=[1, 2]).run(
            pool, test_dataset=test_set, methods=["frozen"], model_factory=lambda: shared
        )
    with pytest.raises(ConfigurationError, match="fewer than 64 samples"):
        FewShotEvaluator(shots=[4, 64], seeds=[1]).run(
            pool, test_dataset=test_set, methods=["frozen"], model_factory=factory
        )
