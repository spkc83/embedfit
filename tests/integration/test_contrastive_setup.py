"""Step budgets, the SupCon projection head, and SupCon views (ADRs 0017 and 0018)."""

import json

import numpy as np
import pytest
import torch

from embedfit import EmbedFitModel, ImageDataset, Trainer, TrainingArguments, representation
from embedfit.augmentation import register_augmentation, registry
from embedfit.augmentation.pipeline import augment_image
from embedfit.data.image import to_rgb_tensor
from embedfit.exceptions import ConfigurationError
from embedfit.losses.supcon import SupConLoss
from embedfit.representation import train_representation
from embedfit.sampling import register_sampler
from embedfit.sampling import registry as sampler_registry
from embedfit.serialization import load_model
from embedfit.testing import BAR_CLASSES, make_bar_images

LABELS = list(BAR_CLASSES[:3])
EMBEDDING_DIM = 16


def dataset(per_class, seed):
    return ImageDataset.from_lists(*make_bar_images(per_class, num_classes=3, seed=seed))


def make_model():
    return EmbedFitModel.from_pretrained("dummy/tiny", labels=LABELS, embedding_dim=EMBEDDING_DIM)


def make_args(mode="supcon", **kwargs):
    options = {
        "training_mode": mode,
        "num_epochs": 1,
        "batch_size": 12,
        "learning_rate": 1e-2,
        "num_workers": 0,
        "positives_per_example": 1,
        "negatives_per_example": 1,
        **kwargs,
    }
    return TrainingArguments(**options)


def train(mode="supcon", train_data=None, eval_data=None, **kwargs):
    model = make_model()
    data = train_data if train_data is not None else dataset(12, 0)
    trainer = Trainer(model, make_args(mode, **kwargs), data, eval_dataset=eval_data)
    return model, trainer, trainer.train()


@pytest.fixture
def supcon_calls(monkeypatch):
    """Shape of the embeddings of every SupCon call, and whether gradients were on."""
    calls = []
    forward = SupConLoss.forward

    def spy(self, embeddings, targets):
        calls.append((tuple(embeddings.shape), torch.is_grad_enabled()))
        return forward(self, embeddings, targets)

    monkeypatch.setattr(SupConLoss, "forward", spy)
    return calls


# --- max_steps ------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["supcon", "pair", "end_to_end"])
def test_max_steps_is_the_budget_and_num_epochs_is_ignored(mode):
    _, _, output = train(mode, eval_data=dataset(4, 1), num_epochs=1, max_steps=7)
    assert output.representation_steps == 7
    assert len(output.train_losses) == len(output.eval_losses) >= 2
    assert np.isfinite(output.train_losses).all()


def test_a_final_partial_epoch_counts_as_an_epoch(supcon_calls):
    # 36 samples in batches of 12: 3 steps an epoch, so 7 steps end one step into epoch 3.
    _, _, output = train(eval_data=dataset(4, 1), max_steps=7)
    assert output.representation_steps == 7
    assert len(output.train_losses) == len(output.eval_losses) == 3
    assert output.best_epoch == int(np.argmin(output.eval_losses)) + 1
    assert sum(grad for _, grad in supcon_calls) == 7


def test_steps_count_optimizer_steps_under_gradient_accumulation(supcon_calls):
    # 3 batches in groups of 2 and 1: 2 steps an epoch. Step 5 is the first group of epoch 3.
    _, _, output = train(max_steps=5, gradient_accumulation_steps=2)
    assert output.representation_steps == 5
    assert len(output.train_losses) == 3
    assert sum(grad for _, grad in supcon_calls) == 3 + 3 + 2


def test_the_schedule_is_sized_on_max_steps(monkeypatch):
    sizes = []
    schedule = representation._schedule

    def spy(total_steps, warmup_ratio):
        sizes.append(total_steps)
        return schedule(total_steps, warmup_ratio)

    monkeypatch.setattr(representation, "_schedule", spy)
    train(num_epochs=4, max_steps=5)
    assert sizes == [5]


def test_without_max_steps_the_budget_is_still_num_epochs():
    _, _, output = train(num_epochs=2)
    assert output.representation_steps == 6
    assert len(output.train_losses) == 2


def test_a_partial_final_epoch_is_checkpointed(tmp_path):
    directory = tmp_path / "checkpoints"
    train(
        max_steps=7,
        checkpoint_dir=str(directory),
        output_dir=str(tmp_path / "model"),
    )
    index = json.loads((directory / "checkpoints.json").read_text())
    assert (index["best_epoch"], index["last_epoch"], len(index["epochs"])) == (3, 3, 3)
    metadata = json.loads((directory / "epoch-0003" / "training_metadata.json").read_text())
    assert metadata["checkpoint"]["steps"] == 7
    assert metadata["training_arguments"]["max_steps"] == 7


class LateFailure:
    """Class-pair batches, three an epoch, with no positive pair from epoch 3 on."""

    def __init__(self, targets, *, batch_size, seed):
        self.targets, self.epoch = np.asarray(targets), 0

    def _batches(self):
        if self.epoch >= 2:
            return [[int(np.flatnonzero(self.targets == c)[0]) for c in np.unique(self.targets)]]
        return [np.flatnonzero(self.targets == c)[:2].tolist() for c in np.unique(self.targets)]

    def __iter__(self):
        return iter(self._batches())

    def __len__(self):
        return len(self._batches()) if self.epoch < 2 else 3

    def set_epoch(self, epoch):
        self.epoch = epoch


@pytest.fixture
def late_failure(monkeypatch):
    monkeypatch.setattr(sampler_registry, "_SAMPLERS", dict(sampler_registry._SAMPLERS))
    register_sampler("late_failure", LateFailure)
    return "late_failure"


def test_the_sampler_check_covers_the_epochs_max_steps_will_run(late_failure):
    data = dataset(4, 0)
    Trainer(make_model(), make_args(sampler=late_failure, max_steps=6), data)  # 2 epochs.
    with pytest.raises(ConfigurationError, match=r"epoch 3.*no two samples of the same class"):
        Trainer(make_model(), make_args(sampler=late_failure, max_steps=7), data)


class NoBatches(LateFailure):
    """A sampler that reports and yields no batches."""

    def _batches(self):
        return []

    def __len__(self):
        return 0


def test_a_sampler_without_batches_fails_before_max_steps_are_planned(monkeypatch):
    monkeypatch.setattr(sampler_registry, "_SAMPLERS", dict(sampler_registry._SAMPLERS))
    register_sampler("no_batches", NoBatches)
    with pytest.raises(ConfigurationError, match="no optimizer step in epoch 1"):
        Trainer(make_model(), make_args(sampler="no_batches", max_steps=5), dataset(4, 0))


class Singles(LateFailure):
    """Three batches an epoch, each of one image per class: no two samples of a class."""

    def _batches(self):
        single = [int(np.flatnonzero(self.targets == c)[0]) for c in np.unique(self.targets)]
        return [single] * 3

    def __len__(self):
        return 3


@pytest.fixture
def singles(monkeypatch):
    monkeypatch.setattr(sampler_registry, "_SAMPLERS", dict(sampler_registry._SAMPLERS))
    register_sampler("singles", Singles)
    return "singles"


def test_one_view_needs_two_samples_of_a_class_per_batch(singles):
    with pytest.raises(ConfigurationError, match="no two samples of the same class"):
        Trainer(make_model(), make_args(sampler=singles, augmentation="natural"), dataset(4, 0))


def test_several_views_are_positives_of_each_other_in_the_sampler_check(singles):
    args = make_args(sampler=singles, max_steps=4, contrastive_views=2, augmentation="natural")
    output = Trainer(make_model(), args, dataset(4, 0)).train()
    assert output.representation_steps == 4


def test_validation_batches_have_one_view_and_still_need_two_samples_of_a_class(singles):
    args = make_args(sampler=singles, contrastive_views=2, augmentation="natural")
    with pytest.raises(ConfigurationError, match="no two samples of the same class"):
        Trainer(make_model(), args, dataset(4, 0), eval_dataset=dataset(4, 1))


def test_pair_setup_validation_draws_no_pairs_ahead(monkeypatch):
    drawn = []
    pairs_for = representation._PairDataset.pairs_for

    def spy(self, epoch):
        drawn.append(epoch)
        return pairs_for(self, epoch)

    monkeypatch.setattr(representation._PairDataset, "pairs_for", spy)
    Trainer(make_model(), make_args("pair", max_steps=50), dataset(4, 0))
    assert drawn == [0]  # Only the dataset's first epoch, as without max_steps.


# --- projection_dim -------------------------------------------------------------------------


def test_supcon_is_computed_on_the_projection(supcon_calls):
    model, trainer, output = train(eval_data=dataset(4, 1), num_epochs=2, projection_dim=8)
    assert supcon_calls and all(shape[-1] == 8 for shape, _ in supcon_calls)
    assert any(not grad for _, grad in supcon_calls)  # Validation used the projection too.
    assert np.isfinite(output.eval_losses).all()
    # The head is fitted on the backbone's embeddings.
    assert model.encode(dataset(2, 2).images).shape == (6, EMBEDDING_DIM)
    assert len(trainer.predict(dataset(2, 2)).predictions) == 6


def test_the_projection_is_optimised_with_the_backbone(monkeypatch):
    sizes = []
    adamw = torch.optim.AdamW

    class Recording(adamw):
        def __init__(self, params, **kwargs):
            params = list(params)
            sizes.append(len(params))
            super().__init__(params, **kwargs)

    monkeypatch.setattr(torch.optim, "AdamW", Recording)
    backbone = sum(1 for _ in make_model().backbone.parameters())
    train(projection_dim=8)
    train()
    assert sizes == [backbone + 4, backbone]  # Two linear layers, each a weight and a bias.


def test_the_projection_is_seeded_from_the_arguments_not_the_global_generator():
    data, test_data = dataset(12, 0), dataset(4, 2)
    embeddings = []
    for global_seed in (1, 2):
        torch.manual_seed(global_seed)
        model = make_model()
        train_representation(model, data, None, make_args(num_epochs=2, projection_dim=8))
        model.eval()
        embeddings.append(model.encode(test_data.images))
    assert np.array_equal(*embeddings)


def test_the_projection_does_not_hide_a_frozen_backbone():
    model = make_model()
    model.freeze_backbone()
    with pytest.raises(ConfigurationError, match="Every backbone parameter is frozen"):
        train_representation(model, dataset(12, 0), None, make_args(projection_dim=8))


def test_the_end_to_end_classifier_does_not_hide_a_frozen_backbone():
    model = make_model()
    model.freeze_backbone()
    with pytest.raises(ConfigurationError, match="Every backbone parameter is frozen"):
        train_representation(model, dataset(12, 0), None, make_args("end_to_end"))


def test_the_projection_seed_leaves_accelerator_generators_alone(monkeypatch):
    def forbidden(seed):
        raise AssertionError("an accelerator generator was seeded")

    model = make_model()
    monkeypatch.setattr(torch.cuda, "manual_seed_all", forbidden)
    monkeypatch.setattr(torch.cuda, "manual_seed", forbidden)
    representation._projection_head(model, make_args(projection_dim=8))


def test_the_best_epoch_state_holds_the_projection(monkeypatch):
    snapshots = []
    state_dict = torch.nn.ModuleList.state_dict

    def spy(self, *args, **kwargs):
        state = state_dict(self, *args, **kwargs)
        snapshots.append(set(state))
        return state

    monkeypatch.setattr(torch.nn.ModuleList, "state_dict", spy)
    train(eval_data=dataset(4, 1), projection_dim=8)
    # The ModuleList holds the backbone at 0 and the projection head at 1.
    assert snapshots and all({"1.0.weight", "1.2.weight"} <= keys for keys in snapshots)


def test_the_projection_is_not_saved_with_the_model(tmp_path):
    model, trainer, _ = train(projection_dim=8)
    directory = trainer.save_model(tmp_path / "model")
    metadata = json.loads((directory / "training_metadata.json").read_text())
    assert metadata["training_arguments"]["projection_dim"] == 8
    loaded = load_model(directory)
    assert set(loaded.state_dict()) == set(model.state_dict())
    images = dataset(2, 2).images
    assert np.array_equal(loaded.encode(images), model.encode(images))


# --- contrastive_views ----------------------------------------------------------------------


def test_each_view_is_a_separate_augmentation_of_the_drawn_image(supcon_calls):
    train(eval_data=dataset(4, 1), contrastive_views=2, augmentation="natural", projection_dim=8)
    training = [shape for shape, grad in supcon_calls if grad]
    validation = [shape for shape, grad in supcon_calls if not grad]
    assert training == [(12, 2, 8)] * 3
    assert validation and all(len(shape) == 2 for shape in validation)


def test_views_get_distinct_draw_keys(monkeypatch):
    keys = []

    def recording(pipeline, image, *, seed, epoch, draw):
        keys.append((epoch, draw))
        return image

    monkeypatch.setattr(representation, "augment_image", recording)
    data = dataset(12, 0)
    train(train_data=data, num_epochs=2, contrastive_views=3, augmentation="natural")
    assert sorted(keys) == [(epoch, draw) for epoch in range(2) for draw in range(3 * len(data))]


def test_a_view_is_the_draw_keyed_by_its_position():
    pipeline = registry.create_augmentation("natural")
    model, data = make_model(), dataset(2, 0)
    targets = model.label_mapping.encode(data.labels)
    views = representation._AugmentedPixelDataset(model, data, targets, pipeline, 5, views=2)
    pixels, _ = views[(1, 3, 4)]
    image = to_rgb_tensor(data.images[4])
    for view in range(2):
        expected = augment_image(pipeline, image, seed=5, epoch=1, draw=3 * 2 + view)
        assert torch.equal(pixels[view], model.backbone.preprocess(expected))
    assert not torch.equal(pixels[0], pixels[1])
    single = representation._AugmentedPixelDataset(model, data, targets, pipeline, 5)
    expected = augment_image(pipeline, image, seed=5, epoch=1, draw=3)
    assert torch.equal(single[(1, 3, 4)][0], model.backbone.preprocess(expected))


def test_one_view_trains_exactly_like_before(monkeypatch):
    monkeypatch.setattr(registry, "_AUGMENTATIONS", dict(registry._AUGMENTATIONS))
    register_augmentation("identity", lambda: lambda image: image)
    _, _, plain = train(num_epochs=2, augmentation="identity")
    _, _, single = train(num_epochs=2, augmentation="identity", contrastive_views=1)
    assert plain.train_losses == single.train_losses


# Worker processes are forked from the threaded test process; the warning is about the test
# runner, not about the code under test.
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_views_are_reproducible_and_independent_of_workers():
    test_data = dataset(4, 2)
    runs = [
        train(contrastive_views=2, augmentation="natural", num_workers=workers)
        for workers in (0, 2)
    ]
    assert runs[0][2].train_losses == runs[1][2].train_losses
    assert np.array_equal(*(model.encode(test_data.images) for model, _, _ in runs))
