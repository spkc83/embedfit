"""Representation checkpoints on disk (ADR 0016)."""

import json

import numpy as np
import pytest
import torch

from embedfit import EmbedFitModel, ImageDataset, Trainer, TrainingArguments
from embedfit.exceptions import ConfigurationError, HeadNotFittedError
from embedfit.heads import LogisticRegressionHead
from embedfit.representation import train_representation
from embedfit.testing import BAR_CLASSES, make_bar_images

LABELS = list(BAR_CLASSES[:3])


def dataset(per_class, seed):
    return ImageDataset.from_lists(*make_bar_images(per_class, num_classes=3, seed=seed))


def make_model(**kwargs):
    return EmbedFitModel.from_pretrained("dummy/tiny", labels=LABELS, embedding_dim=32, **kwargs)


def args(directory, mode="supcon", **kwargs):
    options = {
        "training_mode": mode,
        "num_epochs": 3,
        "batch_size": 6,
        "learning_rate": 1e-2,
        "num_workers": 0,
        "checkpoint_dir": str(directory),
        "output_dir": str(directory.parent / "model"),
        **kwargs,
    }
    return TrainingArguments(**options)


def index(directory):
    return json.loads((directory / "checkpoints.json").read_text())


def test_without_validation_only_the_last_epoch_is_kept(tmp_path):
    directory = tmp_path / "checkpoints"
    output = Trainer(make_model(), args(directory), dataset(6, 0)).train()
    assert sorted(p.name for p in directory.iterdir()) == ["checkpoints.json", "epoch-0003"]
    record = index(directory)
    assert (record["best_epoch"], record["last_epoch"]) == (3, 3) == (output.best_epoch, 3)
    assert [e["train_loss"] for e in record["epochs"]] == list(output.train_losses)


@pytest.fixture(scope="module")
def validated(tmp_path_factory):
    directory = tmp_path_factory.mktemp("validated") / "checkpoints"
    train = dataset(6, 0)
    model = make_model()
    trainer = Trainer(
        model, args(directory, num_epochs=4, learning_rate=5e-2), train, dataset(4, 3)
    )
    return directory, train, model, trainer.train()


def test_best_and_last_epochs_are_kept(validated):
    directory, _, _, output = validated
    record = index(directory)
    assert record["best_epoch"] == output.best_epoch and record["last_epoch"] == 4
    expected = {f"epoch-{epoch:04d}" for epoch in (output.best_epoch, 4)}
    assert {p.name for p in directory.iterdir() if p.is_dir()} == expected
    assert [e["eval_loss"] for e in record["epochs"]] == list(output.eval_losses)
    assert not [p for p in directory.iterdir() if p.name.startswith(".")]


def test_a_checkpoint_is_a_model_bundle_with_an_unfitted_head(validated):
    directory, train, model, output = validated
    loaded = EmbedFitModel.from_pretrained(directory / f"epoch-{output.best_epoch:04d}")
    assert loaded.labels == model.labels and not loaded.head.is_fitted
    with pytest.raises(HeadNotFittedError):
        loaded.predict(train.images[:1])
    loaded.fit_head(train.images, train.labels)
    np.testing.assert_allclose(
        loaded.predict_proba(train.images), model.predict_proba(train.images), atol=1e-5
    )
    metadata = json.loads(
        (directory / f"epoch-{output.best_epoch:04d}" / "training_metadata.json").read_text()
    )
    assert metadata["checkpoint"]["epoch"] == output.best_epoch
    assert not {"output_dir", "checkpoint_dir"} & metadata["training_arguments"].keys()


def test_end_to_end_checkpoints_hold_an_unfitted_linear_head(tmp_path):
    directory = tmp_path / "checkpoints"
    Trainer(make_model(), args(directory, mode="end_to_end", num_epochs=1), dataset(6, 0)).train()
    config = json.loads((directory / "epoch-0001" / "embedfit_config.json").read_text())
    assert config["head"] == {"type": "linear", "fitted": False, "configured_by_name": True}


def test_checkpoints_contain_no_pickles_and_no_training_data(tmp_path):
    images, labels = make_bar_images(6, num_classes=3, seed=0)
    secret = {"account_number": "000123456789"}
    train = ImageDataset(
        [
            {"image": image, "label": label, "sample_id": f"sid-{i}-77", "metadata": secret}
            for i, (image, label) in enumerate(zip(images, labels, strict=True))
        ]
    )
    directory = tmp_path / "checkpoints"
    trainer = Trainer(make_model(), args(directory, mode="pair", num_epochs=2), train)
    trainer.train()
    trainer.save_model()
    for path in (p for p in tmp_path.rglob("*") if p.is_file()):
        content = path.read_bytes()
        for forbidden in (b"000123456789", b"account_number", b"sid-", str(tmp_path).encode()):
            assert forbidden not in content, path
        if path.suffix == ".json":
            json.loads(content)
    torch.load(directory / "epoch-0002" / "backbone" / "weights.pt", weights_only=True)


def test_a_failed_save_leaves_earlier_checkpoints_intact(tmp_path, monkeypatch):
    directory = tmp_path / "checkpoints"
    model = make_model()
    original = type(model.backbone).save_pretrained
    calls = []

    def failing(self, path):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("disk full")
        original(self, path)

    monkeypatch.setattr(type(model.backbone), "save_pretrained", failing)
    with pytest.raises(OSError, match="disk full"):
        Trainer(model, args(directory), dataset(6, 0)).train()
    assert sorted(p.name for p in directory.iterdir()) == ["checkpoints.json", "epoch-0001"]
    assert index(directory)["last_epoch"] == 1
    EmbedFitModel.from_pretrained(directory / "epoch-0001")


def test_a_non_empty_directory_is_refused_without_naming_it(tmp_path):
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    (directory / "other.txt").write_text("x")
    with pytest.raises(ConfigurationError, match="checkpoint_dir is not empty") as error:
        Trainer(make_model(), args(directory), dataset(6, 0))
    assert str(tmp_path) not in str(error.value)


def test_a_file_is_refused_as_not_a_directory_without_naming_it(tmp_path):
    directory = tmp_path / "checkpoints"
    directory.write_text("x")
    with pytest.raises(ConfigurationError, match="checkpoint_dir is not a directory") as error:
        Trainer(make_model(), args(directory), dataset(6, 0))
    assert str(tmp_path) not in str(error.value)


def test_a_second_run_into_the_same_directory_is_refused_without_side_effects(tmp_path):
    directory = tmp_path / "checkpoints"
    model = make_model()
    trainer = Trainer(model, args(directory, num_epochs=1, unfreeze_last_blocks=1), dataset(6, 0))
    trainer.train()
    assert model.backbone.is_frozen
    with pytest.raises(ConfigurationError, match="checkpoint_dir is not empty"):
        trainer.train()
    assert model.backbone.is_frozen
    assert not any(p.requires_grad for p in model.backbone.parameters())


def test_direct_training_writes_checkpoints_too(tmp_path):
    directory = tmp_path / "checkpoints"
    model = make_model()
    model.unfreeze_backbone()
    train_representation(model, dataset(6, 0), None, args(directory, num_epochs=1))
    assert index(directory)["last_epoch"] == 1


def test_unsaveable_setups_fail_when_the_trainer_is_created(tmp_path):
    directory = tmp_path / "checkpoints"
    tuples = [("a", 1), ("b", 2), ("c", 3)]
    images, labels = make_bar_images(6, num_classes=3, seed=0)
    as_tuples = [tuples[LABELS.index(label)] for label in labels]
    model = EmbedFitModel.from_pretrained("dummy/tiny", labels=tuples)
    with pytest.raises(ConfigurationError, match="cannot be saved"):
        Trainer(model, args(directory), ImageDataset.from_lists(images, as_tuples))

    fitted = LogisticRegressionHead()
    train = dataset(6, 0)
    instance_model = make_model(head=fitted)
    instance_model.fit_head(train.images, train.labels)
    with pytest.raises(ConfigurationError, match="fitted head"):
        Trainer(instance_model, args(directory), train)
    assert not directory.exists()


def test_an_unfitted_head_instance_is_copied_into_checkpoints(tmp_path):
    directory = tmp_path / "checkpoints"
    head = LogisticRegressionHead(class_weight="balanced")
    model = make_model(head=head)
    Trainer(model, args(directory, num_epochs=1), dataset(6, 0)).train()
    loaded = EmbedFitModel.from_pretrained(directory / "epoch-0001")
    assert isinstance(loaded.head, LogisticRegressionHead) and not loaded.head.is_fitted
    assert loaded.head.class_weight == "balanced"


def test_save_model_refuses_the_checkpoint_directory(tmp_path):
    directory = tmp_path / "checkpoints"
    trainer = Trainer(make_model(), args(directory, num_epochs=1), dataset(6, 0))
    trainer.train()
    for target in (directory, directory / "model", tmp_path):
        with pytest.raises(ConfigurationError, match="checkpoint_dir"):
            trainer.save_model(target)


def test_direct_training_into_a_used_directory_leaves_the_backbone_frozen(tmp_path):
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    (directory / "other.txt").write_text("x")
    model = make_model()
    model.freeze_backbone()
    with pytest.raises(ConfigurationError, match="checkpoint_dir is not empty"):
        train_representation(
            model, dataset(6, 0), None, args(directory, unfreeze_last_blocks=1), validate=False
        )
    assert model.backbone.is_frozen


def test_checkpoints_record_the_sampler_augmentation_and_partial_unfreezing(tmp_path):
    directory = tmp_path / "checkpoints"
    model = make_model()
    last = set(model.backbone.parameter_blocks()[-1])
    before = {n: v.detach().clone() for n, v in model.backbone.state_dict().items()}
    options = {
        "sampler": "balanced",
        "augmentation": "natural",
        "unfreeze_last_blocks": 1,
        "num_epochs": 1,
    }
    Trainer(model, args(directory, mode="end_to_end", **options), dataset(6, 0)).train()
    changed = {n for n, v in model.backbone.state_dict().items() if not torch.equal(v, before[n])}
    assert changed == last
    metadata = json.loads((directory / "epoch-0001" / "training_metadata.json").read_text())
    recorded = metadata["training_arguments"]
    assert recorded["sampler"] == "balanced" and recorded["augmentation"] == "natural"
    assert recorded["unfreeze_last_blocks"] == 1
