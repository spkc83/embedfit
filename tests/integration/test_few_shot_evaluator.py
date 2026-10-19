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
    complete = {"method": "m", "backbone": "b", "shots": 4, "seed": 1, "metric": "accuracy"}
    with pytest.raises(ConfigurationError, match="numeric value"):
        FewShotEvaluator.aggregate([complete | {"value": "high"}])
    with pytest.raises(ConfigurationError, match="keys method, backbone"):
        FewShotEvaluator.aggregate(
            [{k: v for k, v in complete.items() if k != "seed"} | {"value": 1}]
        )
    assert FewShotEvaluator.aggregate([]) == []


def test_aggregate_of_a_real_run(records):
    rows = FewShotEvaluator.aggregate(records)
    assert len(rows) == len(records) // 2  # Two seeds per group.
    assert all(row["num_runs"] == 2 and row["num_undefined"] == 0 for row in rows)


def test_every_method_trains_on_the_same_few_shot_sample_and_never_on_test_data(
    pool, test_set, monkeypatch
):
    import embedfit.benchmark as benchmark

    seen = []
    original = benchmark.Trainer

    class SpyTrainer(original):
        def __init__(self, model, args, train_dataset, eval_dataset=None, **k):
            # No validation data is used inside the benchmark, least of all the test set.
            assert eval_dataset is None
            seen.append((args.training_mode, args.seed, tuple(id(s) for s in train_dataset)))
            super().__init__(model, args, train_dataset, eval_dataset, **k)

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
    with pytest.raises(ConfigurationError, match="new EmbedFitModel with a new backbone"):
        run(model_factory=lambda: shared)
    with pytest.raises(ConfigurationError, match="new EmbedFitModel with a new backbone"):
        # A new wrapper around a shared backbone would leak trained weights between runs.
        run(model_factory=lambda: EmbedFitModel(shared.backbone, labels=LABELS))
    with pytest.raises(ConfigurationError, match="must return an EmbedFitModel"):
        run(model_factory=lambda: "model")
    import copy

    with pytest.raises(ConfigurationError, match="new EmbedFitModel with a new backbone"):
        # A shallow copy is a new object that still shares every parameter.
        run(model_factory=lambda: EmbedFitModel(copy.copy(shared.backbone), labels=LABELS))

    def shares_a_submodule():
        model = factory()
        model.backbone.features = shared.backbone.features
        return model

    with pytest.raises(ConfigurationError, match="new EmbedFitModel with a new backbone"):
        run(model_factory=shares_a_submodule)
    with pytest.raises(ConfigurationError, match="must be callable"):
        run(model_factory="factory")
    with pytest.raises(ConfigurationError, match="must be a TrainingArguments, got str"):
        run(training_arguments={"frozen": "fast"})
    with pytest.raises(ConfigurationError, match="must map method names"):
        run(training_arguments=["frozen"])
    with pytest.raises(ConfigurationError, match="fewer than 64 samples"):
        FewShotEvaluator(shots=[4, 64], seeds=[1]).run(
            pool, test_dataset=test_set, methods=["frozen"], model_factory=factory
        )


def test_a_method_that_cannot_train_fails_before_any_other_method_trains(
    pool, test_set, monkeypatch
):
    import embedfit.benchmark as benchmark

    trained = []
    original_train = benchmark.Trainer.train
    monkeypatch.setattr(
        benchmark.Trainer, "train", lambda self: trained.append(1) or original_train(self)
    )
    evaluator = FewShotEvaluator(shots=[1, 4], seeds=[1])  # SupCon needs 2 samples per class.
    with pytest.raises(ConfigurationError, match="smallest class has 1"):
        evaluator.run(
            pool, test_dataset=test_set, methods=["frozen", "supcon"], model_factory=factory
        )
    assert trained == []


def test_checkpoint_dir_is_refused_before_any_training(pool, test_set, tmp_path):
    calls = []

    def counting_factory():
        calls.append(1)
        return factory()

    arguments = {
        "supcon": TrainingArguments(
            training_mode="supcon", num_workers=0, checkpoint_dir=str(tmp_path / "ckpt")
        )
    }
    with pytest.raises(ConfigurationError, match="checkpoint_dir"):
        FewShotEvaluator(shots=[4], seeds=[1]).run(
            pool,
            test_dataset=test_set,
            methods=["frozen", "supcon"],
            model_factory=counting_factory,
            training_arguments=arguments,
        )
    assert not calls and not (tmp_path / "ckpt").exists()
