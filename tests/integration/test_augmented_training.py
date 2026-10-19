"""Task augmentation during representation training (SPEC §13, ADR 0014)."""

import io
import json

import numpy as np
import pytest
import torch

from embedfit import EmbedFitModel, ImageDataset, Trainer, TrainingArguments, representation
from embedfit.augmentation import register_augmentation, registry
from embedfit.exceptions import ConfigurationError
from embedfit.sampling import PairSampler, register_sampler
from embedfit.sampling import registry as sampler_registry
from embedfit.testing import BAR_CLASSES, make_bar_images

LABELS = list(BAR_CLASSES[:3])
TRAINED_MODES = ["supcon", "pair", "end_to_end"]


def dataset(per_class, seed):
    return ImageDataset.from_lists(*make_bar_images(per_class, num_classes=3, seed=seed))


def make_model():
    return EmbedFitModel.from_pretrained("dummy/tiny", labels=LABELS, embedding_dim=16)


def make_args(mode, **kwargs):
    options = {
        "training_mode": mode,
        "num_epochs": 2,
        "batch_size": 12,
        "learning_rate": 1e-2,
        "num_workers": 0,
        "positives_per_example": 1,
        "negatives_per_example": 1,
        **kwargs,
    }
    return TrainingArguments(**options)


class Recorder:
    """An identity pipeline that counts its calls and records one random draw per call."""

    def __init__(self):
        self.calls = 0
        self.draws = []

    def __call__(self, image):
        self.calls += 1
        self.draws.append(torch.rand(()).item())
        return image


@pytest.fixture
def recorder(monkeypatch):
    monkeypatch.setattr(registry, "_AUGMENTATIONS", dict(registry._AUGMENTATIONS))
    pipeline = Recorder()
    register_augmentation("recorder", lambda: pipeline)
    return pipeline


@pytest.fixture
def draw_keys(monkeypatch):
    """The ``(epoch, draw)`` key of every augmented training image, in order; no augmentation."""
    keys = []

    def recording(pipeline, image, *, seed, epoch, draw):
        keys.append((epoch, draw))
        return image

    monkeypatch.setattr(representation, "augment_image", recording)
    return keys


def train(mode, train_data, eval_data=None, **kwargs):
    model = make_model()
    trainer = Trainer(model, make_args(mode, **kwargs), train_data, eval_dataset=eval_data)
    return model, trainer, trainer.train()


def training_draws(mode, args, data):
    """Images that representation training draws from the training set."""
    if mode == "supcon":
        return len(data) * args.num_epochs  # 36 samples: three full 12-image batches an epoch.
    if mode == "end_to_end":
        return len(data) * args.num_epochs
    sampler = PairSampler(
        data,
        positives_per_example=args.positives_per_example,
        negatives_per_example=args.negatives_per_example,
        seed=args.seed,
    )
    return 2 * sum(len(sampler.sample(epoch)) for epoch in range(args.num_epochs))


@pytest.mark.parametrize("mode", TRAINED_MODES)
def test_only_training_draws_are_augmented(mode, recorder):
    train_data, eval_data, test_data = dataset(12, 0), dataset(4, 1), dataset(4, 2)
    model = make_model()
    args = make_args(mode, augmentation="recorder")
    trainer = Trainer(model, args, train_data, eval_dataset=eval_data)
    assert recorder.calls == 1  # The set-up probe on a synthetic image.
    trainer.train()
    # Validation batches, the end_to_end linear probe, and head fitting add no calls.
    assert recorder.calls == 1 + training_draws(mode, args, train_data)
    calls = recorder.calls
    trainer.predict(test_data)
    trainer.evaluate(test_data)
    model.encode(test_data.images)
    assert recorder.calls == calls


def test_hard_negative_mining_encodes_clean_images(recorder, monkeypatch):
    model = make_model()
    encode = model.encode
    calls_during_encode = []

    def spy(*args, **kwargs):
        before = recorder.calls
        result = encode(*args, **kwargs)
        calls_during_encode.append(recorder.calls - before)
        return result

    monkeypatch.setattr(model, "encode", spy)
    args = make_args("pair", augmentation="recorder", hard_negative_fraction=1.0)
    Trainer(model, args, dataset(12, 0)).train()
    assert len(calls_during_encode) >= args.num_epochs  # Mining re-encodes every epoch.
    assert calls_during_encode == [0] * len(calls_during_encode)


def test_both_sides_of_every_pair_in_every_epoch_get_their_own_view(recorder):
    train("pair", dataset(12, 0), augmentation="recorder")
    assert len(set(recorder.draws[1:])) == len(recorder.draws) - 1


def test_pair_draw_keys_count_both_sides_from_zero_every_epoch(draw_keys):
    data = dataset(12, 0)
    args = make_args("pair", augmentation="natural")
    Trainer(make_model(), args, data, eval_dataset=dataset(4, 1)).train()
    sampler = PairSampler(data, positives_per_example=1, negatives_per_example=1, seed=args.seed)
    assert draw_keys == [
        (epoch, draw)
        for epoch in range(args.num_epochs)
        for draw in range(2 * len(sampler.sample(epoch)))
    ]


@pytest.mark.parametrize("mode", TRAINED_MODES)
def test_an_identity_pipeline_trains_exactly_like_none(mode, monkeypatch):
    monkeypatch.setattr(registry, "_AUGMENTATIONS", dict(registry._AUGMENTATIONS))
    register_augmentation("identity", lambda: lambda image: image)
    train_data, eval_data, test_data = dataset(12, 0), dataset(4, 1), dataset(4, 2)
    plain, _, plain_output = train(mode, train_data, eval_data, augmentation="none")
    same, _, same_output = train(mode, train_data, eval_data, augmentation="identity")
    assert plain_output.train_losses == same_output.train_losses
    assert plain_output.eval_losses == same_output.eval_losses
    assert np.array_equal(plain.encode(test_data.images), same.encode(test_data.images))


@pytest.mark.parametrize("mode", TRAINED_MODES)
@pytest.mark.parametrize("profile", ["natural", "document"])
def test_every_trained_mode_runs_with_each_profile(mode, profile, tmp_path):
    train_data, test_data = dataset(12, 0), dataset(4, 2)
    _, trainer, output = train(mode, train_data, augmentation=profile)
    assert np.isfinite(output.train_losses).all()
    assert len(trainer.predict(test_data).predictions) == len(test_data)
    directory = trainer.save_model(tmp_path / "model")
    metadata = json.loads((directory / "training_metadata.json").read_text())
    assert metadata["training_arguments"]["augmentation"] == profile


# Worker processes are forked from the threaded test process; the warning is about the test
# runner, not about the code under test.
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
@pytest.mark.parametrize("mode", ["supcon", "pair"])
def test_augmented_training_is_reproducible_and_independent_of_workers(mode):
    train_data, test_data = dataset(12, 0), dataset(4, 2)
    runs = [
        train(mode, train_data, augmentation="natural", num_workers=workers)
        for workers in (0, 0, 2)
    ]
    losses = [output.train_losses for _, _, output in runs]
    assert losses[0] == losses[1] == losses[2]
    embeddings = [model.encode(test_data.images) for model, _, _ in runs]
    assert np.array_equal(embeddings[0], embeddings[2])
    _, _, plain = train(mode, train_data, augmentation="none")
    assert plain.train_losses != losses[0]  # The augmentation did change the batches.


def test_a_sample_drawn_twice_gets_a_different_view_each_time(recorder):
    images, labels = make_bar_images(12, num_classes=3, seed=0)
    keep = [i for i, label in enumerate(labels) if label != LABELS[0]] + [0, 1]
    data = ImageDataset.from_lists([images[i] for i in keep], [labels[i] for i in keep])
    train("supcon", data, augmentation="recorder")
    # The rare class has two samples, so they recur in most batches.
    assert len(recorder.draws) - 1 > len(data) * 2
    assert len(set(recorder.draws[1:])) == len(recorder.draws) - 1


def test_a_malformed_pipeline_fails_before_training(monkeypatch):
    monkeypatch.setattr(registry, "_AUGMENTATIONS", dict(registry._AUGMENTATIONS))
    register_augmentation("gray", lambda: lambda image: image[:1])
    model = make_model()
    before = [p.detach().clone() for p in model.backbone.parameters()]
    with pytest.raises(ConfigurationError, match=r"\[3, H, W\]"):
        Trainer(model, make_args("supcon", augmentation="gray"), dataset(12, 0))
    assert all(torch.equal(p, b) for p, b in zip(model.backbone.parameters(), before, strict=True))


def test_a_pipeline_unregistered_after_the_arguments_were_made_is_rejected(monkeypatch):
    monkeypatch.setattr(registry, "_AUGMENTATIONS", dict(registry._AUGMENTATIONS))
    register_augmentation("temporary", lambda: lambda image: image)
    args = make_args("supcon", augmentation="temporary")
    del registry._AUGMENTATIONS["temporary"]
    with pytest.raises(ConfigurationError, match="Unknown augmentation 'temporary'"):
        Trainer(make_model(), args, dataset(12, 0))


class Repeating:
    """A custom sampler whose batches hold every chosen position twice."""

    def __init__(self, targets, *, batch_size, seed):
        self._size = len(targets)
        self._half = batch_size // 2
        self._seed = seed
        self._epoch = 0

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __len__(self):
        return 2

    def __iter__(self):
        order = np.random.default_rng([self._seed, self._epoch]).permutation(self._size).tolist()
        for start in (0, self._half):
            yield [p for p in order[start : start + self._half] for _ in range(2)]


@pytest.fixture
def repeating(monkeypatch):
    monkeypatch.setattr(sampler_registry, "_SAMPLERS", dict(sampler_registry._SAMPLERS))
    register_sampler("repeating", Repeating)
    return "repeating"


@pytest.mark.parametrize("mode", ["supcon", "end_to_end"])
def test_a_custom_sampler_that_repeats_positions_gets_a_view_per_draw(mode, repeating, recorder):
    args = make_args(mode, sampler=repeating, augmentation="recorder")
    Trainer(make_model(), args, dataset(12, 0)).train()
    draws_per_epoch = 2 * args.batch_size  # Two batches of six positions, each drawn twice.
    assert recorder.calls == 1 + draws_per_epoch * args.num_epochs
    assert len(set(recorder.draws[1:])) == len(recorder.draws) - 1


@pytest.mark.parametrize("mode", ["supcon", "end_to_end"])
def test_an_identity_pipeline_trains_like_none_under_a_custom_sampler(mode, repeating, monkeypatch):
    monkeypatch.setattr(registry, "_AUGMENTATIONS", dict(registry._AUGMENTATIONS))
    register_augmentation("identity", lambda: lambda image: image)
    train_data, eval_data, test_data = dataset(12, 0), dataset(4, 1), dataset(4, 2)
    plain, _, plain_output = train(
        mode, train_data, eval_data, sampler=repeating, augmentation="none"
    )
    same, _, same_output = train(
        mode, train_data, eval_data, sampler=repeating, augmentation="identity"
    )
    assert plain_output.train_losses == same_output.train_losses
    assert plain_output.eval_losses == same_output.eval_losses
    assert np.array_equal(plain.encode(test_data.images), same.encode(test_data.images))


@pytest.mark.parametrize("mode", ["supcon", "end_to_end"])
def test_draws_are_counted_from_zero_every_epoch_after_the_setup_check(mode, repeating, draw_keys):
    args = make_args(mode, sampler=repeating, augmentation="natural")
    Trainer(make_model(), args, dataset(12, 0), eval_dataset=dataset(4, 1)).train()
    draws = 2 * args.batch_size
    assert draw_keys == [(epoch, draw) for epoch in range(args.num_epochs) for draw in range(draws)]


def test_encoded_bytes_train_exactly_like_the_decoded_images():
    images, labels = make_bar_images(6, num_classes=3, seed=0)
    encoded = []
    for image in images:
        buffer = io.BytesIO()
        image.save(buffer, "PNG")  # Lossless, so both datasets hold the same pixels.
        encoded.append(buffer.getvalue())
    test_data = dataset(4, 2)
    runs = [
        train("supcon", ImageDataset.from_lists(source, labels), augmentation="natural")
        for source in (images, encoded)
    ]
    assert runs[0][2].train_losses == runs[1][2].train_losses
    assert np.array_equal(*(model.encode(test_data.images) for model, _, _ in runs))
