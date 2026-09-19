"""Frozen baseline end to end: dataset → Trainer → head fit → predict (acceptance B, C, L)."""

import logging

import numpy as np
import pytest
import torch

from embedfit import EmbedFitModel, ImageDataset, Trainer, TrainingArguments
from embedfit.exceptions import ConfigurationError, HeadNotFittedError
from embedfit.testing import BAR_CLASSES, make_bar_images

LABELS = list(BAR_CLASSES[:3])


@pytest.fixture
def datasets():
    train = ImageDataset.from_lists(*make_bar_images(12, num_classes=3, seed=0))
    test = ImageDataset.from_lists(*make_bar_images(10, num_classes=3, seed=1))
    return train, test


def make_model(**kwargs) -> EmbedFitModel:
    return EmbedFitModel.from_pretrained("dummy/tiny", labels=LABELS, embedding_dim=32, **kwargs)


def test_frozen_training_fits_a_head_without_touching_the_backbone(datasets):
    train, test = datasets
    model = make_model()
    before = [p.detach().clone() for p in model.backbone.parameters()]

    trainer = Trainer(model, TrainingArguments(training_mode="frozen"), train, eval_dataset=test)
    output = trainer.train()

    assert (output.training_mode, output.num_train_samples, output.representation_steps) == (
        "frozen",
        len(train),
        0,
    )
    assert model.backbone.is_frozen and model.head.is_fitted
    for parameter, original in zip(model.backbone.parameters(), before, strict=True):
        assert not parameter.requires_grad and parameter.grad is None
        assert torch.equal(parameter, original)

    result = trainer.predict()
    assert result.label_order == tuple(LABELS)
    assert result.probabilities.shape == (len(test), 3)
    assert np.allclose(result.probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert result.labels == test.labels
    # An untrained backbone gives weak features, so this baseline only has to beat chance
    # (1/3); seeds are fixed, so the value (0.60) is deterministic. Representation training
    # is what is expected to raise it.
    assert (result.predictions == np.array(test.labels)).mean() > 0.5


def test_model_level_api_matches_trainer(datasets):
    train, test = datasets
    model = make_model()
    model.fit_head(train.images, train.labels)
    from_images = model.predict_proba(test.images)
    from_embeddings = model.predict_proba(embeddings=model.encode(test.images))
    assert np.allclose(from_images, from_embeddings, atol=1e-6)
    assert model.predict(test.images).tolist() == [LABELS[i] for i in from_images.argmax(axis=1)]
    assert model.predict(test.images[0]).shape == (1,)


def test_probability_columns_follow_declared_label_order_not_sort_order(datasets):
    train, test = datasets
    forward = make_model()
    forward.fit_head(train.images, train.labels)
    reordered = EmbedFitModel.from_pretrained("dummy/tiny", labels=LABELS[::-1], embedding_dim=32)
    reordered.fit_head(train.images, train.labels)
    assert np.allclose(
        forward.predict_proba(test.images),
        reordered.predict_proba(test.images)[:, ::-1],
        atol=1e-4,
    )
    assert forward.predict(test.images).tolist() == reordered.predict(test.images).tolist()


def test_risk_score_requires_and_uses_the_explicit_positive_label(datasets):
    train, test = datasets
    implicit = make_model()
    implicit.fit_head(train.images, train.labels)
    with pytest.raises(ConfigurationError, match="explicit positive class"):
        implicit.risk_score(test.images)

    model = make_model(positive_label="horizontal")
    model.fit_head(train.images, train.labels)
    scores = model.risk_score(test.images)
    assert np.allclose(scores, model.predict_proba(test.images)[:, LABELS.index("horizontal")])
    is_positive = np.array(test.labels) == "horizontal"
    assert scores[is_positive].mean() > scores[~is_positive].mean()


def test_empty_input_predicts_nothing(datasets):
    train, _ = datasets
    model = make_model()
    model.fit_head(train.images, train.labels)
    assert model.predict([]).shape == (0,)
    assert model.predict_proba([]).shape == (0, 3)


def test_prediction_before_fitting(datasets):
    with pytest.raises(HeadNotFittedError, match="fit_head or Trainer.train"):
        make_model().predict(datasets[1].images)


def test_fit_head_input_validation(datasets):
    train, _ = datasets
    model = make_model()
    with pytest.raises(ConfigurationError, match="exactly one of images or embeddings"):
        model.fit_head(train.images, train.labels, embeddings=np.zeros((36, 32), np.float32))
    with pytest.raises(ConfigurationError, match="exactly one"):
        model.fit_head(labels=train.labels)
    with pytest.raises(ConfigurationError, match="requires labels"):
        model.fit_head(train.images)
    with pytest.raises(ConfigurationError, match="dimension 8, expected 32"):
        model.fit_head(labels=train.labels, embeddings=np.zeros((36, 8), np.float32))
    with pytest.raises(ConfigurationError, match="not one of the model's labels"):
        model.fit_head(train.images[:2], ["vertical", "zigzag"])
    with pytest.raises(ConfigurationError, match="no training samples"):
        model.fit_head(train.images[:12], train.labels[:12])  # Only the first class.
    unlabeled = EmbedFitModel.from_pretrained("dummy/tiny")
    with pytest.raises(ConfigurationError, match="never inferred"):
        unlabeled.fit_head(train.images, train.labels)


def test_trainer_rejects_bad_configuration_before_training(datasets):
    train, _ = datasets
    with pytest.raises(ConfigurationError, match=r"'triplet' is not available.*'end_to_end'"):
        Trainer(make_model(), TrainingArguments(training_mode="triplet"), train)
    frozen = TrainingArguments(training_mode="frozen")
    with pytest.raises(ConfigurationError, match="class labels"):
        Trainer(EmbedFitModel.from_pretrained("dummy/tiny"), frozen, train)
    with pytest.raises(ConfigurationError, match="train_dataset must be an ImageDataset"):
        Trainer(make_model(), frozen, train.images)
    with pytest.raises(ConfigurationError, match="empty"):
        Trainer(make_model(), frozen, ImageDataset([]))
    with pytest.raises(ConfigurationError, match="not one of the model's labels"):
        Trainer(make_model(), frozen, ImageDataset.from_lists(train.images[:1], ["zigzag"]))
    with pytest.raises(ConfigurationError, match="contradicts training_mode='frozen'"):
        Trainer(
            make_model(),
            TrainingArguments(training_mode="frozen", freeze_backbone_before_head=False),
            train,
        )
    with pytest.raises(ConfigurationError, match="needs a dataset"):
        Trainer(make_model(), frozen, train).predict()


def test_repeated_runs_give_identical_probabilities(datasets):
    train, test = datasets
    runs = []
    for _ in range(2):
        trainer = Trainer(make_model(), TrainingArguments(training_mode="frozen", seed=7), train)
        trainer.train()
        runs.append(trainer.predict(test).probabilities)
    assert np.array_equal(runs[0], runs[1])


def test_arguments_reach_seeding_encoding_and_device_placement(datasets, monkeypatch):
    # The frozen pipeline is deterministic whatever the seed, so output comparisons cannot
    # show that arguments are honoured; observe the calls instead.
    import embedfit.trainer as trainer_module

    train, test = datasets
    calls = {"batch_sizes": []}
    monkeypatch.setattr(
        trainer_module,
        "set_seed",
        lambda seed, *, deterministic: calls.update(seed=seed, deterministic=deterministic),
    )
    model = make_model()
    encode = model.encode

    def spy_encode(images, **kwargs):
        calls["batch_sizes"].append(kwargs["batch_size"])
        return encode(images, **kwargs)

    monkeypatch.setattr(model, "encode", spy_encode)
    monkeypatch.setattr(model, "to", lambda device: calls.update(device=device) or model)

    args = TrainingArguments(
        training_mode="frozen", seed=1234, deterministic=True, eval_batch_size=5, device="cpu"
    )
    trainer = Trainer(model, args, train)
    trainer.train()
    trainer.predict(test)
    assert (calls["seed"], calls["deterministic"]) == (1234, True)
    assert calls["batch_sizes"] == [5, 5]
    assert calls["device"] == torch.device("cpu")


def test_configured_head_instance_survives_training(datasets):
    from embedfit.heads import LogisticRegressionHead

    train, _ = datasets
    head = LogisticRegressionHead(class_weight="balanced", C=0.01)
    model = make_model(head=head)
    assert model.head_name is None
    Trainer(model, TrainingArguments(training_mode="frozen"), train).train()
    assert model.head is head and head.is_fitted and head.class_weight == "balanced"


def test_head_name_in_arguments_replaces_a_named_head_only(datasets, monkeypatch):
    from embedfit.heads import LogisticRegressionHead, registry

    train, _ = datasets
    monkeypatch.setattr(registry, "_HEADS", dict(registry._HEADS))
    registry.register_head("strong_l2", lambda: LogisticRegressionHead(C=0.01))

    model = make_model()
    original = model.head
    Trainer(model, TrainingArguments(training_mode="frozen", head="strong_l2"), train).train()
    assert model.head is not original and model.head.C == 0.01 and model.head_name == "strong_l2"

    with pytest.raises(ConfigurationError, match="created with a head instance"):
        Trainer(
            make_model(head=LogisticRegressionHead()),
            TrainingArguments(training_mode="frozen", head="strong_l2"),
            train,
        )


def test_logs_name_phases_and_counts_but_no_sample_content(caplog, logged_messages):
    caplog.set_level(logging.DEBUG, logger="embedfit")
    images, labels = make_bar_images(4, num_classes=2, seed=0)
    secret = {"account_number": "000123456789"}
    train = ImageDataset(
        [
            {
                "image": np.full((32, 32, 3), 173, np.uint8) if i == 0 else image,
                "label": label,
                "sample_id": f"sid-{i}-77",
                "amount": 4321.5,
                "metadata": secret,
            }
            for i, (image, label) in enumerate(zip(images, labels, strict=True))
        ]
    )
    model = EmbedFitModel.from_pretrained("dummy/tiny", labels=list(BAR_CLASSES[:2]))
    Trainer(model, TrainingArguments(training_mode="frozen"), train).train()

    messages = logged_messages()
    assert "Phase 1/2" in messages and "Phase 2/2" in messages
    assert f"{len(train)} samples" in messages
    for forbidden in ("000123456789", "account_number", "4321", "sid-", "173", "vertical"):
        assert forbidden not in messages


def test_evaluate_matches_metrics_computed_from_predictions(datasets):
    from embedfit.evaluation import brier_score, classification_metrics, pr_auc, roc_auc

    train, test = datasets
    model = make_model(positive_label="horizontal")
    trainer = Trainer(model, TrainingArguments(training_mode="frozen"), train, eval_dataset=test)
    trainer.train()
    results = trainer.evaluate()

    output = trainer.predict()
    scores = output.probabilities[:, LABELS.index("horizontal")]
    expected = classification_metrics(
        test.labels, output.predictions, labels=LABELS, positive_label="horizontal"
    )
    for name, metric in (("roc_auc", roc_auc), ("pr_auc", pr_auc), ("brier_score", brier_score)):
        expected[name] = metric(test.labels, scores, positive_label="horizontal")
    assert results == pytest.approx(expected)
    assert results["roc_auc"] > 0.5 and 0 <= results["brier_score"] <= 1
    assert trainer.evaluate(train)["accuracy"] >= results["accuracy"]  # Explicit dataset wins.


def test_evaluate_without_positive_label_reports_only_label_metrics(datasets):
    train, test = datasets
    trainer = Trainer(make_model(), TrainingArguments(training_mode="frozen"), train)
    trainer.train()
    results = trainer.evaluate(test)
    assert set(results) == {
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    }


def test_evaluate_reports_undefined_metrics_as_nan_with_a_warning(datasets, caplog):
    train, test = datasets
    model = make_model(positive_label="horizontal")
    trainer = Trainer(model, TrainingArguments(training_mode="frozen"), train)
    trainer.train()
    no_positives = ImageDataset([sample for sample in test if sample.label != "horizontal"])
    with caplog.at_level(logging.WARNING, logger="embedfit"):
        results = trainer.evaluate(no_positives)
    assert np.isnan(results["roc_auc"]) and np.isnan(results["pr_auc"])
    assert not np.isnan(results["brier_score"]) and not np.isnan(results["accuracy"])
    assert "reported as NaN" in caplog.text


def test_evaluate_needs_data(datasets):
    train, _ = datasets
    trainer = Trainer(make_model(), TrainingArguments(training_mode="frozen"), train)
    trainer.train()
    with pytest.raises(ConfigurationError, match="needs a dataset"):
        trainer.evaluate()
    with pytest.raises(ConfigurationError, match="empty dataset"):
        trainer.evaluate(ImageDataset([]))
