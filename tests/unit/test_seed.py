import random

import numpy as np
import pytest
import torch

from embedfit.exceptions import ConfigurationError
from embedfit.utils import set_seed


def _draw() -> tuple[float, float, float]:
    return random.random(), float(np.random.rand()), float(torch.rand(1))


def test_same_seed_reproduces_all_generators():
    set_seed(123)
    first = _draw()
    set_seed(123)
    assert _draw() == first


def test_different_seeds_differ():
    set_seed(1)
    first = _draw()
    set_seed(2)
    assert _draw() != first


@pytest.mark.parametrize("bad", [-1, 2**32, 1.5, "7", None, True])
def test_invalid_seed_rejected(bad):
    with pytest.raises(ConfigurationError, match="seed"):
        set_seed(bad)


def test_deterministic_mode_sets_torch_flags():
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_benchmark = torch.backends.cudnn.benchmark
    try:
        set_seed(0, deterministic=True)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.benchmark is False
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)
        torch.backends.cudnn.benchmark = previous_benchmark
