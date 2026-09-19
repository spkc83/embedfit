import json

import numpy as np
import pytest
import torch
from torch import nn

from embedfit.exceptions import ConfigurationError, HeadNotFittedError
from embedfit.heads import LinearHead, create_head, load_head


@pytest.fixture
def blobs():
    rng = np.random.default_rng(0)
    centers = np.array([[4.0, 0.0], [0.0, 4.0], [-4.0, -4.0]])
    targets = np.repeat(np.arange(3), 20)
    return (centers[targets] + rng.normal(0, 0.7, (60, 2))).astype(np.float32), targets


def test_learns_separable_classes_with_columns_in_class_index_order(blobs):
    features, targets = blobs
    head = LinearHead()
    assert not head.is_fitted and head.num_classes is None
    head.fit(features, targets, num_classes=3)
    assert head.is_fitted and head.num_classes == 3
    assert (head.predict(features) == targets).all()
    probabilities = head.predict_proba(np.array([[4, 0], [0, 4], [-4, -4]], dtype=np.float32))
    assert probabilities.argmax(axis=1).tolist() == [0, 1, 2]
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_fit_is_deterministic_and_leaves_the_global_rng_alone(blobs):
    features, targets = blobs
    torch.manual_seed(3)
    expected_next = torch.rand(1)
    torch.manual_seed(3)
    first, second = LinearHead(seed=5), LinearHead(seed=5)
    first.fit(features, targets, num_classes=3)
    assert torch.equal(torch.rand(1), expected_next)
    second.fit(features, targets, num_classes=3)
    assert np.array_equal(first.predict_proba(features), second.predict_proba(features))


def test_weight_decay_shrinks_the_weights(blobs):
    features, targets = blobs
    light, heavy = LinearHead(weight_decay=0.0), LinearHead(weight_decay=1.0)
    light.fit(features, targets, num_classes=3)
    heavy.fit(features, targets, num_classes=3)
    assert heavy._layer.weight.norm() < 0.5 * light._layer.weight.norm()


def test_from_module_copies_a_trained_layer(blobs):
    layer = nn.Linear(2, 3)
    head = LinearHead.from_module(layer, seed=7)
    expected = torch.softmax(layer(torch.from_numpy(blobs[0])), dim=1).detach().numpy()
    assert np.allclose(head.predict_proba(blobs[0]), expected, atol=1e-6)
    with torch.no_grad():
        layer.weight.zero_()  # The head holds its own copy.
    assert np.allclose(head.predict_proba(blobs[0]), expected, atol=1e-6)
    assert head.seed == 7 and not any(p.requires_grad for p in head._layer.parameters())
    with pytest.raises(ConfigurationError, match="nn.Linear with a bias"):
        LinearHead.from_module(nn.Linear(2, 3, bias=False))


def test_save_and_load_round_trip_without_pickle(blobs, tmp_path):
    features, targets = blobs
    head = LinearHead(weight_decay=0.01, max_iter=50, seed=9)
    head.fit(features, targets, num_classes=3)
    head.save(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["config.json", "parameters.npz"]
    np.load(tmp_path / "parameters.npz", allow_pickle=False).close()
    loaded = load_head(tmp_path)
    assert isinstance(loaded, LinearHead)
    assert (loaded.weight_decay, loaded.max_iter, loaded.seed) == (0.01, 50, 9)
    assert np.array_equal(loaded.predict_proba(features), head.predict_proba(features))

    LinearHead().save(tmp_path)  # An unfitted head removes stale parameters.
    assert not (tmp_path / "parameters.npz").exists() and not load_head(tmp_path).is_fitted


def test_corrupt_files_are_configuration_errors(blobs, tmp_path):
    head = LinearHead()
    head.fit(*blobs, num_classes=3)
    head.save(tmp_path)
    np.savez(tmp_path / "parameters.npz", weight=np.zeros((3, 2)), bias=np.zeros(5))
    with pytest.raises(ConfigurationError, match="not a .num_classes, D. weight"):
        LinearHead.load(tmp_path)
    np.savez(tmp_path / "parameters.npz", weight=np.array([object()], dtype=object))
    with pytest.raises(ConfigurationError, match="valid saved LinearHead"):
        LinearHead.load(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    (tmp_path / "config.json").write_text(json.dumps({"type": config["type"]}))
    with pytest.raises(ConfigurationError, match="valid saved LinearHead"):
        LinearHead.load(tmp_path)


def test_validation(blobs):
    features, targets = blobs
    with pytest.raises(HeadNotFittedError):
        LinearHead().predict_proba(features)
    with pytest.raises(ConfigurationError, match="no training samples"):
        LinearHead().fit(features, targets, num_classes=4)
    head = LinearHead()
    head.fit(features, targets, num_classes=3)
    with pytest.raises(ConfigurationError, match="dimension 5, expected 2"):
        head.predict_proba(np.zeros((1, 5), dtype=np.float32))
    for kwargs in ({"weight_decay": -1}, {"max_iter": 0}, {"seed": -1}, {"weight_decay": True}):
        with pytest.raises(ConfigurationError):
            LinearHead(**kwargs)
    assert isinstance(create_head("linear", max_iter=10), LinearHead)
