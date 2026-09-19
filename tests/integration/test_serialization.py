"""Save → reload round trips (acceptance criterion F, TEST_STRATEGY "Serialization")."""

import json

import numpy as np
import pytest

from embedfit import EmbedFitModel, ImageDataset, Trainer, TrainingArguments
from embedfit.backbones import load_backbone
from embedfit.exceptions import BackboneProviderError, ConfigurationError
from embedfit.heads import ClassificationHead, LogisticRegressionHead, load_head
from embedfit.testing import BAR_CLASSES, make_bar_images

# Deliberately not in sorted order, so that a loader that sorts labels would be caught.
LABELS = ["vertical", "diagonal", "horizontal"]


@pytest.fixture
def data():
    return make_bar_images(8, num_classes=3, seed=0), make_bar_images(5, num_classes=3, seed=1)


@pytest.fixture
def fitted(data):
    (train_images, train_labels), _ = data
    model = EmbedFitModel.from_pretrained(
        "dummy/tiny",
        labels=LABELS,
        positive_label="diagonal",
        embedding_dim=24,
        seed=3,
        head=LogisticRegressionHead(C=0.5, class_weight="balanced"),
    )
    model.fit_head(train_images, train_labels)
    return model


def test_round_trip_preserves_everything_that_affects_predictions(fitted, data, tmp_path):
    _, (images, _) = data
    fitted.save_pretrained(tmp_path / "model")
    loaded = EmbedFitModel.from_pretrained(tmp_path / "model")

    assert np.allclose(loaded.encode(images), fitted.encode(images), atol=1e-6)
    assert np.allclose(loaded.predict_proba(images), fitted.predict_proba(images), atol=1e-6)
    assert loaded.predict(images).tolist() == fitted.predict(images).tolist()
    assert np.allclose(loaded.risk_score(images), fitted.risk_score(images), atol=1e-6)
    assert loaded.labels == tuple(LABELS) and loaded.positive_label == "diagonal"
    assert loaded.normalize_embeddings is True
    assert (
        loaded.backbone.identifier == "dummy/tiny" and loaded.backbone.pooling == "provider_default"
    )
    assert loaded.embedding_dim == 24
    assert (loaded.head.C, loaded.head.class_weight) == (0.5, "balanced")
    assert not loaded.training


def test_normalization_policy_and_str_path_round_trip(data, tmp_path):
    (train_images, train_labels), (images, _) = data
    model = EmbedFitModel.from_pretrained("dummy/tiny", labels=LABELS, normalize_embeddings=False)
    model.fit_head(train_images, train_labels)
    model.save_pretrained(str(tmp_path))
    loaded = EmbedFitModel.from_pretrained(str(tmp_path))
    assert loaded.normalize_embeddings is False
    assert not np.allclose(np.linalg.norm(loaded.encode(images), axis=1), 1.0)
    assert np.allclose(loaded.predict_proba(images), model.predict_proba(images), atol=1e-6)


def test_bundle_layout_and_config_contents(fitted, tmp_path):
    fitted.save_pretrained(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "backbone",
        "embedfit_config.json",
        "head",
        "labels.json",
        "training_metadata.json",
    ]
    config = json.loads((tmp_path / "embedfit_config.json").read_text())
    assert config["framework"] == "embedfit" and config["version"] and config["format_version"] == 1
    assert config["backbone"]["provider"] == "dummy"
    assert config["backbone"]["identifier"] == "dummy/tiny"
    assert config["backbone"]["pooling"] == "provider_default"
    assert config["backbone"]["embedding_dim"] == 24
    assert len(config["backbone"]["preprocessing_fingerprint"]) == 64
    assert config["normalize_embeddings"] is True
    assert config["head"] == {"type": "logistic_regression", "fitted": True}
    assert json.loads((tmp_path / "labels.json").read_text()) == {
        "labels": LABELS,
        "positive_label": "diagonal",
    }


def test_non_string_labels_keep_their_types(data, tmp_path):
    (train_images, train_labels), (images, _) = data
    as_int = [LABELS.index(label) * 10 for label in train_labels]
    model = EmbedFitModel.from_pretrained("dummy/tiny", labels=[0, 20, 10], positive_label=20)
    model.fit_head(train_images, as_int)
    model.save_pretrained(tmp_path)
    loaded = EmbedFitModel.from_pretrained(tmp_path)
    assert loaded.labels == (0, 20, 10) and type(loaded.positive_label) is int
    assert loaded.predict(images).tolist() == model.predict(images).tolist()


def test_unfitted_head_and_label_free_model_round_trip(tmp_path):
    EmbedFitModel.from_pretrained("dummy/tiny").save_pretrained(tmp_path)
    loaded = EmbedFitModel.from_pretrained(tmp_path)
    assert loaded.labels is None and not loaded.head.is_fitted


def test_bundle_contains_no_pickles_and_no_training_data(data, tmp_path):
    (train_images, train_labels), _ = data
    secret = {"account_number": "000123456789"}
    train = ImageDataset(
        [
            {"image": image, "label": label, "sample_id": f"sid-{i}-77", "metadata": secret}
            for i, (image, label) in enumerate(zip(train_images, train_labels, strict=True))
        ]
    )
    model = EmbedFitModel.from_pretrained("dummy/tiny", labels=list(BAR_CLASSES[:3]))
    trainer = Trainer(model, TrainingArguments(training_mode="frozen"), train)
    trainer.train()
    trainer.save_model(tmp_path)

    for path in (p for p in tmp_path.rglob("*") if p.is_file()):
        content = path.read_bytes()
        for forbidden in (b"000123456789", b"account_number", b"sid-"):
            assert forbidden not in content, path
        if path.suffix == ".npz":
            np.load(path, allow_pickle=False).close()  # Raises if any array is pickled.
        if path.suffix == ".json":
            json.loads(content)
    import torch

    # weights_only=True refuses anything but tensors and plain containers.
    torch.load(tmp_path / "backbone" / "weights.pt", weights_only=True)


def test_trainer_save_model_records_aggregate_metadata(data, tmp_path):
    (train_images, train_labels), (images, _) = data
    model = EmbedFitModel.from_pretrained("dummy/tiny", labels=list(BAR_CLASSES[:3]))
    args = TrainingArguments(training_mode="frozen", output_dir=str(tmp_path / "out"), seed=5)
    trainer = Trainer(model, args, ImageDataset.from_lists(train_images, train_labels))
    with pytest.raises(ConfigurationError, match="call train"):
        trainer.save_model()
    trainer.train()
    assert trainer.save_model() == tmp_path / "out"  # Defaults to args.output_dir.

    metadata = json.loads((tmp_path / "out" / "training_metadata.json").read_text())
    assert metadata["training_arguments"]["seed"] == 5
    assert metadata["train_output"]["num_train_samples"] == len(train_images)
    loaded = EmbedFitModel.from_pretrained(tmp_path / "out")
    assert np.allclose(loaded.predict_proba(images), model.predict_proba(images), atol=1e-6)


def test_saved_values_cannot_be_overridden_at_load_time(fitted, tmp_path):
    fitted.save_pretrained(tmp_path)
    for kwargs in (
        {"labels": ["a", "b"]},
        {"positive_label": "diagonal"},
        {"normalize_embeddings": False},
        {"head": LogisticRegressionHead()},
        {"embedding_dim": 8},
    ):
        with pytest.raises(ConfigurationError, match="cannot be set when loading a saved model"):
            EmbedFitModel.from_pretrained(tmp_path, **kwargs)
    assert EmbedFitModel.from_pretrained(tmp_path, device="cpu").labels == tuple(LABELS)


def test_a_directory_is_a_saved_model_only_if_it_holds_the_config(tmp_path, monkeypatch):
    # A folder that happens to be called like an identifier must not hijack the identifier.
    (tmp_path / "dummy" / "tiny").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    assert EmbedFitModel.from_pretrained("dummy/tiny").backbone.identifier == "dummy/tiny"
    with pytest.raises(BackboneProviderError):
        EmbedFitModel.from_pretrained(str(tmp_path / "dummy"))
    with pytest.raises(ConfigurationError, match="not a saved model directory"):
        EmbedFitModel.from_pretrained(tmp_path / "dummy")


def test_tampered_or_incomplete_bundles_are_rejected(fitted, tmp_path):
    fitted.save_pretrained(tmp_path)

    backbone_config = tmp_path / "backbone" / "config.json"
    original = backbone_config.read_text()
    backbone_config.write_text(original.replace('"image_size": 32', '"image_size": 64'))
    with pytest.raises(ConfigurationError, match="changed after the model was saved"):
        EmbedFitModel.from_pretrained(tmp_path)
    backbone_config.write_text(original)

    config_path = tmp_path / "embedfit_config.json"
    config = json.loads(config_path.read_text())
    config_path.write_text(json.dumps(config | {"format_version": 99}))
    with pytest.raises(ConfigurationError, match="format version 99"):
        EmbedFitModel.from_pretrained(tmp_path)
    config_path.write_text(json.dumps({k: v for k, v in config.items() if k != "head"}))
    with pytest.raises(ConfigurationError, match="not a complete model bundle"):
        EmbedFitModel.from_pretrained(tmp_path)
    config_path.write_text(json.dumps(config))

    (tmp_path / "labels.json").unlink()
    with pytest.raises(ConfigurationError, match="labels.json' is missing"):
        EmbedFitModel.from_pretrained(tmp_path)


def test_labels_that_json_cannot_represent_are_rejected_at_save_time(tmp_path):
    model = EmbedFitModel.from_pretrained("dummy/tiny", labels=[("a", 1), ("b", 2)])
    with pytest.raises(ConfigurationError, match="Labels of type tuple cannot be saved"):
        model.save_pretrained(tmp_path)
    assert not (tmp_path / "embedfit_config.json").exists()  # Nothing half-written.


def test_custom_head_without_persistence_fails_clearly(tmp_path):
    class Constant(ClassificationHead):
        is_fitted = True

        def fit(self, features, targets, *, num_classes):
            pass

        def predict_proba(self, features):
            return np.tile([0.5, 0.5], (len(features), 1))

    model = EmbedFitModel.from_pretrained("dummy/tiny", labels=["a", "b"], head=Constant())
    with pytest.raises(
        ConfigurationError, match="Constant cannot be saved: it has no registry_name"
    ):
        model.save_pretrained(tmp_path)
    with pytest.raises(ConfigurationError, match="does not support saving"):
        Constant().save(tmp_path)
    with pytest.raises(ConfigurationError, match="does not support loading"):
        Constant.load(tmp_path)


def test_training_metadata_must_be_json(tmp_path):
    model = EmbedFitModel.from_pretrained("dummy/tiny")
    with pytest.raises(ConfigurationError, match="JSON-serialisable"):
        model.save_pretrained(tmp_path, training_metadata={"array": np.zeros(2)})


class TestHeadPersistence:
    @pytest.mark.parametrize("num_classes", [2, 3])
    def test_fitted_head_predicts_identically_after_reload(self, num_classes, tmp_path):
        rng = np.random.default_rng(0)
        features = rng.normal(size=(40, 5)).astype(np.float32)
        targets = np.arange(40) % num_classes
        head = LogisticRegressionHead(C=2.0, max_iter=500, class_weight="balanced", seed=9)
        head.fit(features, targets, num_classes=num_classes)
        head.save(tmp_path)
        loaded = load_head(tmp_path)
        assert isinstance(loaded, LogisticRegressionHead) and loaded.is_fitted
        assert (loaded.C, loaded.max_iter, loaded.class_weight, loaded.seed) == (
            2.0,
            500,
            "balanced",
            9,
        )
        assert np.array_equal(loaded.predict_proba(features), head.predict_proba(features))
        with pytest.raises(ConfigurationError, match="dimension 4, expected 5"):
            loaded.predict_proba(features[:, :4])

    def test_inconsistent_parameters_are_rejected(self, tmp_path):
        head = LogisticRegressionHead()
        head.fit(np.eye(4, dtype=np.float32), np.array([0, 1, 0, 1]), num_classes=2)
        head.save(tmp_path)
        config = json.loads((tmp_path / "config.json").read_text())
        (tmp_path / "config.json").write_text(json.dumps(config | {"feature_dim": 7}))
        with pytest.raises(ConfigurationError, match="do not match 2 classes and 7 features"):
            load_head(tmp_path)

    def test_unknown_or_missing_head_type(self, tmp_path):
        with pytest.raises(ConfigurationError, match="is missing or is not a JSON object"):
            load_head(tmp_path)
        (tmp_path / "config.json").write_text('{"type": "svm"}')
        with pytest.raises(ConfigurationError, match="not a registered head class"):
            load_head(tmp_path)


class TestBackboneLoading:
    def test_dispatches_on_the_saved_provider(self, dummy_backbone, tmp_path):
        dummy_backbone.save_pretrained(tmp_path)
        assert load_backbone(tmp_path).identifier == "dummy/tiny"

    def test_unknown_provider_and_missing_config(self, tmp_path):
        with pytest.raises(BackboneProviderError, match="is missing or is not a JSON object"):
            load_backbone(tmp_path)
        (tmp_path / "config.json").write_text('{"provider": "mystery"}')
        with pytest.raises(BackboneProviderError, match="no registered loader.*dummy"):
            load_backbone(tmp_path)

    def test_torchvision_model_round_trip(self, tmp_path):
        pytest.importorskip("torchvision")
        images, labels = make_bar_images(3, num_classes=2, seed=0)
        model = EmbedFitModel.from_pretrained(
            "torchvision/resnet18",
            labels=list(BAR_CLASSES[:2]),
            weights=None,
            image_size=64,
            mean=[0.4, 0.5, 0.6],
            std=[0.2, 0.25, 0.3],
        )
        model.fit_head(images, labels)
        model.save_pretrained(tmp_path)
        loaded = EmbedFitModel.from_pretrained(tmp_path)
        assert loaded.backbone.identifier == "torchvision/resnet18"
        assert (loaded.backbone.image_size, loaded.backbone.mean) == (64, (0.4, 0.5, 0.6))
        assert np.allclose(loaded.encode(images), model.encode(images), atol=1e-5)
        assert np.allclose(loaded.predict_proba(images), model.predict_proba(images), atol=1e-5)


def test_supcon_trained_model_survives_a_round_trip(tmp_path):
    """dataset → trainer → representation training → head fit → predict → save → reload."""
    labels = list(BAR_CLASSES[:3])
    train = ImageDataset.from_lists(*make_bar_images(8, num_classes=3, seed=0))
    images, _ = make_bar_images(5, num_classes=3, seed=1)
    model = EmbedFitModel.from_pretrained("dummy/tiny", labels=labels, positive_label="diagonal")
    untrained = model.encode(images)
    args = TrainingArguments(
        training_mode="supcon",
        num_epochs=3,
        batch_size=12,
        learning_rate=1e-2,
        num_workers=0,
        output_dir=str(tmp_path / "run"),
    )
    trainer = Trainer(model, args, train)
    trainer.train()
    saved_to = trainer.save_model()

    loaded = EmbedFitModel.from_pretrained(saved_to)
    assert not np.allclose(loaded.encode(images), untrained, atol=1e-3)  # Adapted weights kept.
    assert np.allclose(loaded.encode(images), model.encode(images), atol=1e-6)
    assert np.allclose(loaded.predict_proba(images), model.predict_proba(images), atol=1e-6)
    assert loaded.predict(images).tolist() == model.predict(images).tolist()

    metadata = json.loads((saved_to / "training_metadata.json").read_text())
    output = metadata["train_output"]
    assert output["training_mode"] == "supcon" and output["representation_steps"] == 6
    assert len(output["train_losses"]) == 3 and output["best_epoch"] == 3
    assert metadata["training_arguments"]["learning_rate"] == 1e-2
