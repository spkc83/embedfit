"""Registered samplers and metrics take effect in Trainer and FewShotEvaluator (SPEC §26)."""

import json
import logging
from functools import partial

import numpy as np
import pytest
import torch

import embedfit.evaluation.registry as metric_registry
import embedfit.sampling.registry as sampler_registry
from embedfit import EmbedFitModel, FewShotEvaluator, ImageDataset, Trainer, TrainingArguments
from embedfit.evaluation import recall_at_fpr, register_metric
from embedfit.exceptions import ConfigurationError
from embedfit.sampling import register_sampler
from embedfit.testing import BAR_CLASSES, make_bar_images

LABELS = list(BAR_CLASSES[:3])


@pytest.fixture(autouse=True)
def isolated_registries(monkeypatch):
    monkeypatch.setattr(sampler_registry, "_SAMPLERS", dict(sampler_registry._SAMPLERS))
    monkeypatch.setattr(metric_registry, "_METRICS", dict(metric_registry._METRICS))


def dataset(per_class, seed):
    return ImageDataset.from_lists(*make_bar_images(per_class, num_classes=3, seed=seed))


def make_model(**kwargs):
    return EmbedFitModel.from_pretrained("dummy/tiny", labels=LABELS, embedding_dim=16, **kwargs)


def args(**kwargs):
    options = {
        "training_mode": "supcon",
        "num_epochs": 2,
        "batch_size": 6,
        "learning_rate": 1e-2,
        "num_workers": 0,
        **kwargs,
    }
    return TrainingArguments(**options)


class Recorder:
    """Class-pair batches that record every epoch and iteration they serve."""

    created = []

    def __init__(self, targets, *, batch_size, seed, bad_epoch=None, overcount=0):
        self.targets, self.seed, self.epoch = np.asarray(targets), seed, 0
        self.bad_epoch, self.overcount = bad_epoch, overcount
        self.served = []
        Recorder.created.append(self)

    def _batches(self):
        if self.epoch == self.bad_epoch:
            # One sample of each class: no positive pair for a contrastive loss.
            return [[int(np.flatnonzero(self.targets == c)[0]) for c in np.unique(self.targets)]]
        return [np.flatnonzero(self.targets == c)[:2].tolist() for c in np.unique(self.targets)]

    def __iter__(self):
        self.served.append(self.epoch)
        return iter(self._batches())

    def __len__(self):
        return len(self._batches()) + self.overcount

    def set_epoch(self, epoch):
        self.epoch = epoch


@pytest.fixture
def recorder():
    Recorder.created = []
    register_sampler("recorder", Recorder)
    return Recorder


def test_a_registered_sampler_serves_training_and_validation_batches(recorder):
    trainer = Trainer(
        make_model(), args(sampler="recorder"), dataset(4, 0), eval_dataset=dataset(4, 1)
    )
    Recorder.created = []  # Instances built while validating the setup are not used to train.
    output = trainer.train()
    train_sampler, validation_sampler = Recorder.created
    assert train_sampler.served == [0, 1]
    assert validation_sampler.served == [0, 0]  # Validation always uses epoch 0's batches.
    assert validation_sampler.seed != train_sampler.seed
    assert output.representation_steps == 6  # 3 class-pair batches x 2 epochs


def test_the_setup_check_iterates_every_epoch_before_training(recorder):
    Trainer(make_model(), args(sampler="recorder", num_epochs=3), dataset(4, 0))
    assert sorted(epoch for r in Recorder.created for epoch in r.served) == [0, 1, 2]


def test_a_batch_without_a_positive_pair_fails_before_training(recorder):
    register_sampler("late_failure", partial(Recorder, bad_epoch=2))
    with pytest.raises(ConfigurationError, match=r"epoch 3.*no two samples of the same class"):
        Trainer(make_model(), args(sampler="late_failure", num_epochs=3), dataset(4, 0))


def test_a_length_that_does_not_match_the_batches_fails_before_training(recorder):
    register_sampler("overcount", partial(Recorder, overcount=1))
    with pytest.raises(ConfigurationError, match="reports 4 batches but yielded 3"):
        Trainer(make_model(), args(sampler="overcount"), dataset(4, 0))


@pytest.mark.parametrize(
    "batch, match",
    [([], "empty batch"), ([0, 12], "outside"), ([-1, 0], "outside"), ([0.5], "integer")],
)
def test_malformed_batches_fail_before_training(batch, match):
    class Broken(Recorder):
        def _batches(self):
            return [batch]

    register_sampler("broken", Broken)
    for mode in ("supcon", "end_to_end"):
        with pytest.raises(ConfigurationError, match=match):
            Trainer(make_model(), args(training_mode=mode, sampler="broken"), dataset(4, 0))


@pytest.mark.parametrize("empty_on", ["train", "eval"])
def test_a_sampler_without_batches_fails_before_training(recorder, empty_on):
    class Empty(Recorder):
        def _batches(self):
            on_validation = self.seed != 42
            return [] if on_validation == (empty_on == "eval") else super()._batches()

    register_sampler("empty", Empty)
    for mode in ("supcon", "end_to_end"):
        with pytest.raises(ConfigurationError, match="yielded no batches in epoch 1"):
            Trainer(
                make_model(),
                args(training_mode=mode, sampler="empty"),
                dataset(4, 0),
                eval_dataset=dataset(4, 1),
            )


def test_a_length_that_varies_between_epochs_fails_before_training(recorder):
    class Growing(Recorder):
        def _batches(self):
            return super()._batches()[: 2 + self.epoch]

    register_sampler("growing", Growing)
    with pytest.raises(ConfigurationError, match="same number of batches in every epoch"):
        Trainer(make_model(), args(sampler="growing"), dataset(4, 0))


def test_validation_batches_are_checked_too(recorder):
    class FailsOnValidation(Recorder):
        def _batches(self):
            # Validation batches are drawn with their own seed, never the training seed.
            self.bad_epoch = None if self.seed == 42 else 0
            return super()._batches()

    register_sampler("fails_on_validation", FailsOnValidation)
    model, train = make_model(), dataset(4, 0)
    Trainer(model, args(sampler="fails_on_validation"), train)
    with pytest.raises(ConfigurationError, match="no two samples of the same class"):
        Trainer(model, args(sampler="fails_on_validation"), train, eval_dataset=dataset(4, 1))


def test_a_sampler_registered_over_balanced_is_checked(recorder):
    register_sampler("balanced", partial(Recorder, bad_epoch=1), overwrite=True)
    with pytest.raises(ConfigurationError, match="epoch 2.*no two samples of the same class"):
        Trainer(make_model(), args(num_epochs=2), dataset(4, 0))


def test_random_batches_for_supcon_are_rejected_when_one_lacks_a_positive_pair():
    # 3 classes, 2 per class, batches of 4 drawn at random: some batch holds no pair.
    with pytest.raises(ConfigurationError, match="no two samples of the same class"):
        Trainer(make_model(), args(sampler="random", batch_size=4, num_epochs=20), dataset(2, 0))


def test_end_to_end_default_batches_are_those_of_a_reseeded_shuffling_data_loader():
    from torch.utils.data import DataLoader, RandomSampler

    from embedfit import representation

    # 15 samples in batches of 4: the last batch is partial.
    objective = representation._objective(
        make_model(),
        dataset(5, 0),
        args(training_mode="end_to_end", batch_size=4),
        (),
        seed=3,
        classifier=torch.nn.Linear(16, 3),
    )
    for epoch in (0, 1, 2):
        objective.set_epoch(epoch)
        generator = torch.Generator().manual_seed(3 * 100_003 + epoch)
        shuffled = RandomSampler(range(15), generator=generator)
        expected = [
            batch.tolist() for batch in DataLoader(range(15), batch_size=4, sampler=shuffled)
        ]
        assert [list(batch) for batch in objective.loader.batch_sampler] == expected


def test_the_default_supcon_sampler_is_not_iterated_at_setup(monkeypatch):
    from embedfit.sampling import BalancedClassBatchSampler

    def no_iteration(self):
        raise AssertionError("the setup check iterated the built-in balanced sampler")

    monkeypatch.setattr(BalancedClassBatchSampler, "__iter__", no_iteration)
    Trainer(make_model(), args(num_epochs=3), dataset(4, 0), eval_dataset=dataset(4, 1))


def test_end_to_end_with_class_balanced_batches(recorder):
    output = Trainer(
        make_model(), args(training_mode="end_to_end", sampler="balanced"), dataset(4, 0)
    ).train()
    assert output.representation_steps == 4  # ceil(12 / 6) batches x 2 epochs


def test_the_sampler_is_recorded_with_the_saved_model(tmp_path, recorder):
    trainer = Trainer(make_model(), args(sampler="recorder", num_epochs=1), dataset(4, 0))
    trainer.train()
    directory = trainer.save_model(tmp_path / "model")
    recorded = json.loads((directory / "training_metadata.json").read_text())
    assert recorded["training_arguments"]["sampler"] == "recorder"
    assert "output_dir" not in recorded["training_arguments"]


@pytest.fixture(scope="module")
def evaluated():
    model = make_model(positive_label=LABELS[0])
    trainer = Trainer(model, TrainingArguments(training_mode="frozen"), dataset(6, 0))
    trainer.train()
    return trainer


def test_requested_metrics_are_added_to_the_defaults(evaluated):
    register_metric("recall_at_fpr_10pct", partial(recall_at_fpr, target_fpr=0.1))
    test = dataset(6, 1)
    defaults = evaluated.evaluate(test)
    results = evaluated.evaluate(test, metrics=["recall_at_fpr_10pct"])
    assert set(results) == set(defaults) | {"recall_at_fpr_10pct"}
    output = evaluated.predict(test)
    scores = output.probabilities[:, 0]
    assert results["recall_at_fpr_10pct"] == recall_at_fpr(
        test.labels, scores, positive_label=LABELS[0], target_fpr=0.1
    )


def test_undefined_requested_metrics_are_nan_with_a_warning(evaluated, caplog):
    register_metric("recall_at_fpr_10pct", partial(recall_at_fpr, target_fpr=0.1))
    test = dataset(6, 1)
    no_positives = ImageDataset([sample for sample in test if sample.label != LABELS[0]])
    with caplog.at_level(logging.WARNING, logger="embedfit"):
        results = evaluated.evaluate(no_positives, metrics=["recall_at_fpr_10pct"])
    assert np.isnan(results["recall_at_fpr_10pct"])
    assert "recall_at_fpr_10pct reported as NaN" in caplog.text


def test_invalid_metric_requests_fail_before_predicting(evaluated, monkeypatch):
    register_metric("custom", partial(recall_at_fpr, target_fpr=0.1))

    def no_prediction(*_, **__):
        raise AssertionError("predicted before checking the metric names")

    monkeypatch.setattr(evaluated, "predict", no_prediction)
    test = dataset(2, 1)
    for metrics, match in (
        (["magic"], "Unknown metric 'magic'"),
        (["custom", "custom"], "duplicates"),
        (["roc_auc"], "computed by default"),
        ("custom", "sequence of metric names"),
        (None, "sequence of metric names"),
        ([["custom"]], "metric names must be str"),
        ([None], "metric names must be str"),
    ):
        with pytest.raises(ConfigurationError, match=match):
            evaluated.evaluate(test, metrics=metrics)


def test_requested_metrics_need_a_positive_label():
    register_metric("custom", partial(recall_at_fpr, target_fpr=0.1))
    trainer = Trainer(make_model(), TrainingArguments(training_mode="frozen"), dataset(6, 0))
    trainer.train()
    with pytest.raises(ConfigurationError, match="positive_label"):
        trainer.evaluate(dataset(2, 1), metrics=["custom"])


def test_few_shot_records_include_requested_metrics():
    register_metric("recall_at_fpr_10pct", partial(recall_at_fpr, target_fpr=0.1))
    records = FewShotEvaluator(shots=[2], seeds=[0]).run(
        dataset(4, 0),
        test_dataset=dataset(4, 1),
        methods=["frozen"],
        model_factory=lambda: make_model(positive_label=LABELS[0]),
        metrics=["recall_at_fpr_10pct"],
    )
    assert "recall_at_fpr_10pct" in {record["metric"] for record in records}


def test_few_shot_records_metrics_given_as_a_generator_for_every_method():
    register_metric("recall_at_fpr_10pct", partial(recall_at_fpr, target_fpr=0.1))
    records = FewShotEvaluator(shots=[2], seeds=[0]).run(
        dataset(4, 0),
        test_dataset=dataset(4, 1),
        methods=["frozen", "supcon"],
        model_factory=lambda: make_model(positive_label=LABELS[0]),
        training_arguments={"supcon": args(num_epochs=1, batch_size=4)},
        metrics=(name for name in ["recall_at_fpr_10pct"]),
    )
    recorded = {record["method"] for record in records if record["metric"] == "recall_at_fpr_10pct"}
    assert recorded == {"frozen", "supcon"}


def test_few_shot_rejects_bad_metrics_before_any_training(monkeypatch):
    import embedfit.benchmark as benchmark

    def no_training(*_, **__):
        raise AssertionError("trained before checking the metric names")

    monkeypatch.setattr(benchmark.Trainer, "train", no_training)
    register_metric("custom", partial(recall_at_fpr, target_fpr=0.1))
    evaluator = FewShotEvaluator(shots=[2], seeds=[0])
    run = partial(evaluator.run, dataset(4, 0), test_dataset=dataset(4, 1), methods=["frozen"])
    with pytest.raises(ConfigurationError, match="Unknown metric 'magic'"):
        run(model_factory=lambda: make_model(positive_label=LABELS[0]), metrics=["magic"])
    with pytest.raises(ConfigurationError, match="positive_label"):
        run(model_factory=make_model, metrics=["custom"])
