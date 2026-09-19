from collections import Counter

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from embedfit.exceptions import ConfigurationError
from embedfit.sampling import BalancedClassBatchSampler

# 40 of class 0, 8 of class 1, 3 of class 2: a rare-event layout.
TARGETS = np.array([0] * 40 + [1] * 8 + [2] * 3)


def make(**kwargs):
    options = {"classes_per_batch": 3, "samples_per_class": 4, "seed": 0, **kwargs}
    return BalancedClassBatchSampler(TARGETS, **options)


def test_every_batch_has_the_requested_class_layout_without_duplicates():
    sampler = make()
    batches = list(sampler)
    assert len(batches) == len(sampler) == 5  # ceil(51 / 12)
    for batch in batches:
        assert len(set(batch)) == len(batch)
        # Class 2 has only 3 samples: it contributes all of them, never a duplicate.
        assert Counter(TARGETS[batch].tolist()) == {0: 4, 1: 4, 2: 3}


def test_every_sample_of_a_class_is_used_before_any_repeats():
    sampler = make(classes_per_batch=1, samples_per_class=4, batches_per_epoch=30)
    seen = {0: [], 1: [], 2: []}
    for batch in sampler:
        seen[int(TARGETS[batch[0]])].extend(batch)
    for target, size in ((0, 40), (1, 8), (2, 3)):
        draws = seen[target]
        for start in range(0, len(draws) - size + 1, size):
            assert len(set(draws[start : start + size])) == size


def test_classes_take_turns_when_only_some_fit_in_a_batch():
    sampler = make(classes_per_batch=2, samples_per_class=2, batches_per_epoch=30)
    visits = Counter()
    for batch in sampler:
        classes = set(TARGETS[batch].tolist())
        assert len(classes) == 2
        visits.update(classes)
    assert max(visits.values()) - min(visits.values()) <= 1  # 60 slots over 3 classes.


def test_batches_depend_only_on_seed_and_epoch():
    np.random.seed(7)
    expected_next = np.random.rand()
    np.random.seed(7)
    first = list(make())
    assert np.random.rand() == expected_next  # Global RNG untouched.
    assert first == list(make())
    assert first != list(make(seed=1))
    sampler = make()
    sampler.set_epoch(1)
    second_epoch = list(sampler)
    assert second_epoch != first
    sampler.set_epoch(0)
    assert list(sampler) == first


def test_works_as_a_dataloader_batch_sampler():
    dataset = TensorDataset(torch.arange(len(TARGETS)), torch.from_numpy(TARGETS))
    for positions, targets in DataLoader(dataset, batch_sampler=make()):
        assert torch.equal(targets, torch.from_numpy(TARGETS)[positions])
        assert (torch.bincount(targets) >= 2).all()  # Every sample has a positive.


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"samples_per_class": 1}, "samples_per_class must be >= 2"),
        ({"samples_per_class": 0}, "positive int"),
        ({"classes_per_batch": 4}, r"exceeds the number of classes \(3\)"),
        ({"classes_per_batch": 1.5}, "positive int"),
        ({"seed": -1}, "seed"),
        ({"batches_per_epoch": 0}, "batches_per_epoch"),
    ],
)
def test_invalid_options(kwargs, match):
    with pytest.raises(ConfigurationError, match=match):
        make(**kwargs)


@pytest.mark.parametrize(
    "targets, match",
    [
        ([0, 0, 1], "smallest class has 1"),
        ([], "non-empty 1-D array"),
        ([[0, 0], [1, 1]], "non-empty 1-D array"),
        ([0.0, 0.0, 1.0, 1.0], "class indices"),
    ],
)
def test_invalid_targets(targets, match):
    with pytest.raises(ConfigurationError, match=match):
        BalancedClassBatchSampler(targets, classes_per_batch=2, samples_per_class=2, seed=0)


def test_draw_refuses_more_items_than_exist_instead_of_looping():
    with pytest.raises(ConfigurationError, match="Cannot draw 3 distinct items from 2"):
        BalancedClassBatchSampler._draw([], [1, 2], 3, np.random.default_rng(0))
