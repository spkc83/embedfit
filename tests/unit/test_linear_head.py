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


def test_to_module_returns_an_independent_trainable_copy(blobs):
    features, targets = blobs
    head = LinearHead()
    with pytest.raises(HeadNotFittedError):
        head.to_module()
    head.fit(features, targets, num_classes=3)
    layer = head.to_module()
    assert isinstance(layer, nn.Linear) and layer.weight.dtype == torch.float32
    assert all(p.requires_grad for p in layer.parameters())
    expected = head.predict_proba(features)
    with torch.no_grad():
        probabilities = torch.softmax(layer(torch.from_numpy(features)), dim=1).double().numpy()
        assert np.allclose(probabilities, expected, atol=1e-6)
        layer.weight.zero_()  # Training the copy must not change the head.
    assert np.array_equal(head.predict_proba(features), expected)
    assert np.allclose(LinearHead.from_module(head.to_module()).predict_proba(features), expected)


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
    with pytest.raises(ConfigurationError, match=r"not a finite float \[num_classes, D\] weight"):
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


def test_fit_after_from_module_starts_from_the_given_weights_and_reaches_the_same_optimum(blobs):
    features, targets = blobs
    cold = LinearHead(max_iter=500)
    cold.fit(features, targets, num_classes=3)
    warm = LinearHead.from_module(nn.Linear(2, 3), max_iter=500)
    warm.fit(features, targets, num_classes=3)
    # The objective is convex: a different start changes the path, not the solution.
    assert np.allclose(warm.predict_proba(features), cold.predict_proba(features), atol=1e-3)
    with pytest.raises(TypeError):
        LinearHead.from_module(nn.Linear(2, 3), bogus=1)
    with pytest.raises(ConfigurationError, match="max_iter"):
        LinearHead.from_module(nn.Linear(2, 3), max_iter=0)


@pytest.mark.parametrize(
    "weight, bias",
    [
        (np.zeros((3, 2), dtype=">f4"), np.zeros(3, dtype=">f4")),
        (np.array([["a", "b"]] * 3), np.zeros(3)),
        (np.zeros((3, 2), dtype=np.complex64), np.zeros(3)),
        (np.zeros((3, 2), dtype=bool), np.zeros(3)),
        (np.full((3, 2), np.nan), np.zeros(3)),
        (np.zeros((3, 0)), np.zeros(3)),
        (np.zeros((1, 2)), np.zeros(1)),
    ],
)
def test_malformed_parameters_are_configuration_errors(blobs, tmp_path, weight, bias):
    head = LinearHead()
    head.fit(*blobs, num_classes=3)
    head.save(tmp_path)
    np.savez(tmp_path / "parameters.npz", weight=weight, bias=bias)
    if weight.dtype.kind == "f" and weight.dtype.byteorder == ">":
        loaded = LinearHead.load(tmp_path)  # A valid float array in another byte order is fine.
        assert loaded.num_classes == 3
        return
    with pytest.raises(ConfigurationError):
        LinearHead.load(tmp_path)


def test_truncated_archive_and_bad_flags_are_configuration_errors(blobs, tmp_path):
    import json

    head = LinearHead()
    head.fit(*blobs, num_classes=3)
    head.save(tmp_path)
    archive = tmp_path / "parameters.npz"
    archive.write_bytes(archive.read_bytes()[: archive.stat().st_size // 2])
    with pytest.raises(ConfigurationError, match="valid saved LinearHead"):
        LinearHead.load(tmp_path)
    head.save(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    (tmp_path / "config.json").write_text(json.dumps(config | {"fitted": "yes"}))
    with pytest.raises(ConfigurationError, match="valid saved LinearHead"):
        LinearHead.load(tmp_path)
