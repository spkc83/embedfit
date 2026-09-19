import numpy as np
import pytest

from embedfit.exceptions import ConfigurationError
from embedfit.heads import (
    ClassificationHead,
    HeadNotFittedError,
    LogisticRegressionHead,
    available_heads,
    create_head,
    register_head,
    registry,
)


@pytest.fixture
def blobs():
    rng = np.random.default_rng(0)
    centers = np.array([[4.0, 0.0], [0.0, 4.0], [-4.0, -4.0]])
    targets = np.repeat(np.arange(3), 20)
    return (centers[targets] + rng.normal(0, 0.5, (60, 2))).astype(np.float32), targets


def test_learns_separable_classes(blobs):
    features, targets = blobs
    head = LogisticRegressionHead()
    assert not head.is_fitted
    head.fit(features, targets, num_classes=3)
    assert head.is_fitted
    assert (head.predict(features) == targets).all()


def test_probability_columns_follow_class_index_order(blobs):
    features, targets = blobs
    head = LogisticRegressionHead()
    head.fit(features, targets, num_classes=3)
    probabilities = head.predict_proba(np.array([[4, 0], [0, 4], [-4, -4]], dtype=np.float32))
    assert probabilities.shape == (3, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert probabilities.argmax(axis=1).tolist() == [0, 1, 2]


def test_same_seed_is_deterministic(blobs):
    features, targets = blobs
    first, second = LogisticRegressionHead(seed=3), LogisticRegressionHead(seed=3)
    first.fit(features, targets, num_classes=3)
    second.fit(features, targets, num_classes=3)
    assert np.array_equal(first.predict_proba(features), second.predict_proba(features))


def test_balanced_class_weight_raises_minority_recall():
    rng = np.random.default_rng(1)
    features = np.concatenate([rng.normal(0, 1, (200, 2)), rng.normal(1.5, 1, (10, 2))])
    targets = np.array([0] * 200 + [1] * 10)
    plain, balanced = LogisticRegressionHead(), LogisticRegressionHead(class_weight="balanced")
    plain.fit(features.astype(np.float32), targets, num_classes=2)
    balanced.fit(features.astype(np.float32), targets, num_classes=2)
    minority = features[200:].astype(np.float32)
    assert balanced.predict(minority).sum() > plain.predict(minority).sum()


def test_predict_before_fit(blobs):
    with pytest.raises(HeadNotFittedError):
        LogisticRegressionHead().predict_proba(blobs[0])


def test_missing_class_is_rejected_so_columns_stay_meaningful(blobs):
    features, targets = blobs
    with pytest.raises(ConfigurationError, match=r"Class indices \[3\] have no training samples"):
        LogisticRegressionHead().fit(features, targets, num_classes=4)


@pytest.mark.parametrize(
    "features, targets, match",
    [
        (np.zeros((4, 2), dtype=np.float32), np.array([0, 1, 0]), "same length"),
        (np.zeros((4, 2), dtype=np.float32), np.array([0, 1, 2, 0]), r"lie in \[0, 2\)"),
        (np.zeros((4, 2), dtype=np.float32), np.array([0.0, 1.0, 0.0, 1.0]), "integer"),
        (np.zeros(4, dtype=np.float32), np.array([0, 1, 0, 1]), "2-D float array"),
        (np.zeros((4, 2), dtype=np.int64), np.array([0, 1, 0, 1]), "2-D float array"),
        (np.full((4, 2), np.nan, dtype=np.float32), np.array([0, 1, 0, 1]), "NaN or infinite"),
    ],
)
def test_invalid_fit_inputs(features, targets, match):
    with pytest.raises(ConfigurationError, match=match):
        LogisticRegressionHead().fit(features, targets, num_classes=2)


def test_predict_rejects_wrong_feature_width(blobs):
    features, targets = blobs
    head = LogisticRegressionHead()
    head.fit(features, targets, num_classes=3)
    with pytest.raises(ConfigurationError, match="dimension 5, expected 2"):
        head.predict_proba(np.zeros((1, 5), dtype=np.float32))


def test_empty_batch_gives_zero_rows(blobs):
    features, targets = blobs
    head = LogisticRegressionHead()
    head.fit(features, targets, num_classes=3)
    assert head.predict_proba(np.empty((0, 2), dtype=np.float32)).shape == (0, 3)
    assert head.predict(np.empty((0, 2), dtype=np.float32)).shape == (0,)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"C": 0},
        {"C": -1.0},
        {"max_iter": 0},
        {"max_iter": 1.5},
        {"class_weight": "x"},
        {"seed": -1},
        {"seed": "x"},
        {"seed": True},
    ],
)
def test_invalid_options(kwargs):
    with pytest.raises(ConfigurationError):
        LogisticRegressionHead(**kwargs)


def test_registry(monkeypatch):
    monkeypatch.setattr(registry, "_HEADS", dict(registry._HEADS))
    assert "logistic_regression" in available_heads()
    assert isinstance(create_head("logistic_regression", C=0.5), LogisticRegressionHead)
    with pytest.raises(ConfigurationError, match="Unknown head 'svm'.*logistic_regression"):
        create_head("svm")
    with pytest.raises(ConfigurationError, match="already registered"):
        register_head("logistic_regression", LogisticRegressionHead)

    class Constant(ClassificationHead):
        is_fitted = True

        def fit(self, features, targets, *, num_classes):
            pass

        def predict_proba(self, features):
            return np.tile([0.2, 0.8], (len(features), 1))

    register_head("no_options", lambda: Constant())
    with pytest.raises(ConfigurationError, match=r"does not accept the options \['seed'\]"):
        create_head("no_options", seed=1)

    register_head("constant", Constant)
    assert create_head("constant").predict(np.zeros((3, 2))).tolist() == [1, 1, 1]
