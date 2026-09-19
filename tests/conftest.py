"""Shared fixtures. Everything here is offline, CPU-only, and deterministic."""

import random

import numpy as np
import pytest
import torch
from PIL import Image

from embedfit.backbones import DummyBackbone
from embedfit.testing import make_bar_images

# Tiny tensors run far faster on one thread, especially on a busy machine, and single-threaded
# arithmetic keeps tiny training runs bit-for-bit reproducible.
torch.set_num_threads(1)


@pytest.fixture(autouse=True)
def _restore_global_rng_state():
    """Keep tests order-independent: undo any global seeding a test performs."""
    python_state, numpy_state, torch_state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)


@pytest.fixture
def bar_dataset() -> tuple[list[Image.Image], list[str]]:
    """Three classes of 32x32 RGB bar images, eight per class."""
    return make_bar_images(8, num_classes=3, seed=0)


@pytest.fixture
def dummy_backbone() -> DummyBackbone:
    return DummyBackbone(embedding_dim=16, image_size=32, seed=0)
