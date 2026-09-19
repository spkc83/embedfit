import numpy as np
import pytest

from embedfit.data import LabelMapping
from embedfit.exceptions import ConfigurationError


def test_order_is_preserved_and_round_trips():
    mapping = LabelMapping(["risk", "legitimate", "other"])
    assert mapping.labels == ("risk", "legitimate", "other")
    assert len(mapping) == 3
    indices = mapping.encode(["other", "risk", "risk"])
    assert indices.dtype == np.int64 and indices.tolist() == [2, 0, 0]
    assert mapping.decode(indices).tolist() == ["other", "risk", "risk"]


def test_numpy_scalars_and_arrays_are_the_same_labels():
    mapping = LabelMapping(np.array(["b", "a"]))
    assert mapping.labels == ("b", "a") and type(mapping.labels[0]) is str
    assert mapping.encode(np.array(["a", "b"])).tolist() == [1, 0]
    assert LabelMapping([0, 1]).encode(np.array([1, 0, 1])).tolist() == [1, 0, 1]


def test_decode_dtype_follows_label_types():
    assert LabelMapping([0, 1]).decode([1, 0]).dtype.kind == "i"
    assert LabelMapping(["a", "b"]).decode([1]).dtype.kind == "U"
    mixed = LabelMapping(["a", 1]).decode([0, 1])
    assert mixed.dtype == object and mixed.tolist() == ["a", 1]


def test_membership_compares_type_as_well_as_value():
    mapping = LabelMapping([0, 1])
    assert 1 in mapping
    assert True not in mapping and 1.0 not in mapping and [1] not in mapping
    with pytest.raises(ConfigurationError, match="value and type"):
        mapping.index_of(True)


def test_unknown_label_is_rejected():
    with pytest.raises(ConfigurationError, match="not one of the model's labels"):
        LabelMapping(["a", "b"]).encode(["a", "c"])


@pytest.mark.parametrize(
    "labels, match",
    [
        (["only"], "at least 2"),
        (["a", "a"], "unique"),
        ("ab", "sequence"),
        ([["a"], ["b"]], "hashable"),
        (np.array([[0, 1]]), "sequence"),
    ],
)
def test_invalid_labels(labels, match):
    with pytest.raises(ConfigurationError, match=match):
        LabelMapping(labels)


def test_tuple_labels_stay_single_elements():
    decoded = LabelMapping([("a", "b"), ("c",)]).decode([1, 0, 1])
    assert decoded.shape == (3,) and decoded.dtype == object
    assert decoded.tolist() == [("c",), ("a", "b"), ("c",)]


def test_nan_label_is_rejected():
    with pytest.raises(ConfigurationError, match="NaN"):
        LabelMapping([float("nan"), 1.0])
