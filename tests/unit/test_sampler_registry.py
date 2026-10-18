import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, RandomSampler

from embedfit.exceptions import ConfigurationError
from embedfit.sampling import (
    BalancedClassBatchSampler,
    BatchSampler,
    available_samplers,
    batch_shape,
    create_sampler,
    register_sampler,
    registry,
)

TARGETS = np.array([0] * 10 + [1] * 6 + [2] * 4)


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    monkeypatch.setattr(registry, "_SAMPLERS", dict(registry._SAMPLERS))


class FixedBatches:
    def __init__(self, targets, *, batch_size, seed):
        self.batches = [list(range(start, start + 2)) for start in range(0, len(targets), 2)]
        self.epochs = []

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

    def set_epoch(self, epoch):
        self.epochs.append(epoch)


def test_builtin_samplers_are_registered():
    assert available_samplers() == ["balanced", "random"]


def test_balanced_matches_a_directly_built_sampler():
    built = create_sampler("balanced", TARGETS, batch_size=8, seed=3)
    classes, per_class = batch_shape(8, 3)
    direct = BalancedClassBatchSampler(
        TARGETS, classes_per_batch=classes, samples_per_class=per_class, seed=3
    )
    assert isinstance(built, BatchSampler)
    for epoch in (0, 1):
        built.set_epoch(epoch)
        direct.set_epoch(epoch)
        assert list(built) == list(direct)


def test_random_matches_a_data_loader_with_a_reseeded_random_sampler():
    built = create_sampler("random", TARGETS, batch_size=6, seed=5)
    assert isinstance(built, BatchSampler) and len(built) == 4
    generator = torch.Generator()
    reference = DataLoader(
        range(len(TARGETS)),
        batch_size=6,
        sampler=RandomSampler(range(len(TARGETS)), generator=generator),
    )
    for epoch in (0, 1, 0):
        built.set_epoch(epoch)
        generator.manual_seed(5 * 100_003 + epoch)
        assert list(built) == [batch.tolist() for batch in reference]


def test_random_repeats_an_epoch_and_varies_between_epochs():
    built = create_sampler("random", TARGETS, batch_size=6, seed=5)
    first = list(built)
    assert list(built) == first
    built.set_epoch(1)
    assert list(built) != first


def test_a_registered_sampler_is_created_with_targets_batch_size_and_seed():
    received = {}

    def factory(targets, *, batch_size, seed):
        received.update(targets=targets, batch_size=batch_size, seed=seed)
        return FixedBatches(targets, batch_size=batch_size, seed=seed)

    register_sampler("fixed", factory)
    assert "fixed" in available_samplers()
    sampler = create_sampler("fixed", TARGETS, batch_size=4, seed=9)
    assert isinstance(sampler, FixedBatches)
    assert received["batch_size"] == 4 and received["seed"] == 9
    assert np.array_equal(received["targets"], TARGETS)


def test_registration_rules():
    register_sampler("fixed", FixedBatches)
    with pytest.raises(ConfigurationError, match="already registered"):
        register_sampler("fixed", FixedBatches)
    register_sampler("fixed", FixedBatches, overwrite=True)
    for name in ("", " fixed", 3):
        with pytest.raises(ConfigurationError, match="Invalid sampler name"):
            register_sampler(name, FixedBatches)
    with pytest.raises(ConfigurationError, match="callable"):
        register_sampler("broken", "not a factory")


def test_unknown_name_lists_the_available_samplers():
    with pytest.raises(ConfigurationError, match=r"Unknown sampler 'magic'.*'balanced'"):
        create_sampler("magic", TARGETS, batch_size=4, seed=0)


def test_a_factory_that_rejects_the_call_is_a_configuration_error():
    register_sampler("no_seed", lambda targets, *, batch_size: FixedBatches(targets, batch_size=1))
    with pytest.raises(ConfigurationError, match="Sampler 'no_seed' does not accept"):
        create_sampler("no_seed", TARGETS, batch_size=4, seed=0)


def test_a_type_error_inside_the_factory_keeps_its_message():
    def buggy(targets, *, batch_size, seed):
        return len(None)

    register_sampler("buggy", buggy)
    with pytest.raises(ConfigurationError, match="does not accept.*NoneType"):
        create_sampler("buggy", TARGETS, batch_size=4, seed=0)


def test_batch_shape_is_still_importable_from_representation():
    from embedfit import representation

    assert representation.batch_shape is batch_shape


def test_a_result_without_the_protocol_is_rejected():
    register_sampler("plain_list", lambda targets, *, batch_size, seed: [[0, 1]])
    with pytest.raises(ConfigurationError, match="__iter__, __len__, and set_epoch"):
        create_sampler("plain_list", TARGETS, batch_size=4, seed=0)
