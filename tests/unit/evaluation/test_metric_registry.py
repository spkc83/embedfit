from functools import partial

import pytest

from embedfit.evaluation import (
    available_metrics,
    brier_score,
    classification_metrics,
    get_metric,
    pr_auc,
    recall_at_fpr,
    register_metric,
    registry,
    roc_auc,
)
from embedfit.exceptions import ConfigurationError

Y_TRUE = ["rare", "common", "rare", "common", "common"]
SCORES = [0.9, 0.8, 0.7, 0.2, 0.1]


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    monkeypatch.setattr(registry, "_METRICS", dict(registry._METRICS))


def test_builtin_metrics_are_the_library_functions():
    assert available_metrics() == ["brier_score", "pr_auc", "roc_auc"]
    assert get_metric("roc_auc") is roc_auc
    assert get_metric("pr_auc") is pr_auc
    assert get_metric("brier_score") is brier_score


def test_a_partial_of_a_library_metric_registers_without_glue():
    metric = partial(recall_at_fpr, target_fpr=0.5)
    register_metric("recall_at_fpr_50pct", metric)
    assert "recall_at_fpr_50pct" in available_metrics()
    assert get_metric("recall_at_fpr_50pct")(Y_TRUE, SCORES, positive_label="rare") == 1.0


def test_registration_rules():
    register_metric("custom", roc_auc)
    with pytest.raises(ConfigurationError, match="already registered"):
        register_metric("custom", pr_auc)
    register_metric("custom", pr_auc, overwrite=True)
    assert get_metric("custom") is pr_auc
    for name in ("", " custom", None):
        with pytest.raises(ConfigurationError, match="Invalid metric name"):
            register_metric(name, roc_auc)
    with pytest.raises(ConfigurationError, match="callable"):
        register_metric("broken", 0.5)


@pytest.mark.parametrize("name", ["roc_auc", "pr_auc", "brier_score"])
def test_builtin_names_are_reserved_even_with_overwrite(name):
    with pytest.raises(ConfigurationError, match="reserved"):
        register_metric(name, recall_at_fpr, overwrite=True)
    assert get_metric(name) is {"roc_auc": roc_auc, "pr_auc": pr_auc}.get(name, brier_score)


def test_names_returned_by_classification_metrics_are_reserved():
    names = classification_metrics(
        Y_TRUE, Y_TRUE, labels=["rare", "common"], positive_label="rare"
    ).keys()
    for name in names:
        with pytest.raises(ConfigurationError, match="reserved"):
            register_metric(name, roc_auc)


def test_unknown_name_lists_the_available_metrics():
    with pytest.raises(ConfigurationError, match=r"Unknown metric 'magic'.*'roc_auc'"):
        get_metric("magic")
