from datetime import datetime

import pytest
import torch.utils.data

from embedfit import ImageDataset
from embedfit.data import ImageSample
from embedfit.exceptions import ConfigurationError


def test_mappings_and_samples_are_normalised(tmp_path):
    records = [
        {
            "image": tmp_path / "a.png",
            "label": "risk",
            "sample_id": "s1",
            "amount": 12.5,
            "group_id": "g1",
            "timestamp": datetime(2024, 1, 1),
            "metadata": {"k": "v"},
        },
        ImageSample(image=tmp_path / "b.png", label="legitimate"),
    ]
    dataset = ImageDataset(records)
    assert len(dataset) == 2
    assert dataset.labels == ["risk", "legitimate"]
    assert dataset.images == [tmp_path / "a.png", tmp_path / "b.png"]  # Paths stay undecoded.
    assert dataset[0].amount == 12.5 and dataset[0].metadata == {"k": "v"}
    assert [sample.label for sample in dataset] == dataset.labels


def test_from_lists(bar_dataset):
    images, labels = bar_dataset
    dataset = ImageDataset.from_lists(images, labels)
    assert len(dataset) == len(images) and dataset.labels == labels
    with pytest.raises(ConfigurationError, match="same length"):
        ImageDataset.from_lists(images, labels[:-1])


def test_map_style_torch_dataset_is_accepted(bar_dataset):
    images, labels = bar_dataset

    class Source(torch.utils.data.Dataset):
        def __len__(self):
            return 3

        def __getitem__(self, position):
            return {"image": images[position], "label": labels[position]}

    assert ImageDataset(Source()).labels == labels[:3]


@pytest.mark.parametrize(
    "records, match",
    [
        ([{"image": "a.png"}], r"Record 0 is missing required keys: \['label'\]"),
        ([{"image": "a.png", "label": 1, "acct": "12345678"}], r"unknown keys \['acct'\]"),
        ([("a.png", 1)], "Record 0 must be an ImageSample or a mapping, got tuple"),
        (iter([]), "must be a sequence"),
    ],
)
def test_invalid_records(records, match):
    with pytest.raises(ConfigurationError, match=match):
        ImageDataset(records)


def test_errors_and_reprs_never_show_record_values():
    with pytest.raises(ConfigurationError) as excinfo:
        ImageDataset([{"image": "a.png", "label": 1, "acct": "12345678"}])
    assert "12345678" not in str(excinfo.value)

    sample = ImageSample("check-0099.png", "risk", amount=9876.5, metadata={"acct": "12345678"})
    for text in (repr(sample), repr(ImageDataset([sample]))):
        assert not any(secret in text for secret in ("0099", "risk", "9876", "12345678"))
    assert "metadata" in repr(sample)


def test_samples_compare_and_hash_by_identity_even_with_array_images():
    import numpy as np

    first = ImageSample(np.zeros((4, 4, 3), np.uint8), "risk", metadata={"k": "v"})
    second = ImageSample(np.zeros((4, 4, 3), np.uint8), "risk", metadata={"k": "v"})
    assert first == first and first != second
    assert len({first, second}) == 2 and first in [second, first]


def test_metadata_is_a_read_only_copy():
    source = {"k": "v"}
    sample = ImageSample("a.png", "x", metadata=source)
    source["k"] = "changed"
    assert sample.metadata["k"] == "v"
    with pytest.raises(TypeError):
        sample.metadata["k"] = "mutated"


def test_indexing_and_slicing(bar_dataset):
    dataset = ImageDataset.from_lists(*bar_dataset)
    assert dataset[-1].label == bar_dataset[1][-1]
    subset = dataset[2:5]
    assert isinstance(subset, ImageDataset) and subset.labels == bar_dataset[1][2:5]


def test_mapping_and_unsized_inputs_are_rejected():
    with pytest.raises(ConfigurationError, match="must be a sequence"):
        ImageDataset({"image": "a.png", "label": 1})
    with pytest.raises(ConfigurationError, match="must be sequences"):
        ImageDataset.from_lists((i for i in range(2)), ["a", "b"])
