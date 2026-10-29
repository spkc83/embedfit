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
def _no_network(request, monkeypatch):
    """Fail any test that opens an internet connection unless it is marked ``network``.

    ADR 0004 makes the default suite offline; this turns that from a convention into a check.
    Unix-domain sockets stay usable, because data-loader worker processes rely on them.
    """
    if request.node.get_closest_marker("network"):
        return
    import socket

    real_connect = socket.socket.connect

    def guarded_connect(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError(
                "This test tried to open a network connection. Mark it with "
                "@pytest.mark.network or build the model without downloading a checkpoint."
            )
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


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
def spawned_workers(monkeypatch):
    """Start the data-loader workers of representation training by spawning, not forking.

    Workers forked from the multi-threaded test process can deadlock; spawned ones start
    clean. This changes how workers start, not what they compute.
    """
    from embedfit import representation

    loader = representation.DataLoader

    def spawning(*arguments, **options):
        if options.get("num_workers"):
            options["multiprocessing_context"] = "spawn"
        return loader(*arguments, **options)

    monkeypatch.setattr(representation, "DataLoader", spawning)


@pytest.fixture
def bar_dataset() -> tuple[list[Image.Image], list[str]]:
    """Three classes of 32x32 RGB bar images, eight per class."""
    return make_bar_images(8, num_classes=3, seed=0)


@pytest.fixture
def dummy_backbone() -> DummyBackbone:
    return DummyBackbone(embedding_dim=16, image_size=32, seed=0)


@pytest.fixture
def logged_messages(caplog):
    """Return the formatted log messages captured so far, without file names or line numbers.

    ``caplog.text`` includes source locations such as ``trainer.py:173``, which would make a
    search for a forbidden value like ``173`` fail for reasons unrelated to privacy.
    """
    return lambda: "\n".join(record.getMessage() for record in caplog.records)
