import io
import logging
import sys
from datetime import datetime

import numpy as np
import pytest
import torch.utils.data
from PIL import Image

from embedfit import EmbedFitModel, ImageDataset, Trainer, TrainingArguments
from embedfit.data import ImageSample
from embedfit.exceptions import ConfigurationError, ImageInputError


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


class TestSkipCorruptImages:
    @pytest.fixture
    def records(self, tmp_path):
        image = Image.new("RGB", (8, 8))
        image.save(tmp_path / "ok.png")
        (tmp_path / "acct-99887766.png").write_bytes(b"not an image")
        encoded = io.BytesIO()
        image.save(encoded, "PNG")
        return [
            {"image": tmp_path / "ok.png", "label": "a"},
            {"image": tmp_path / "missing.png", "label": "b"},
            {"image": tmp_path / "acct-99887766.png", "label": "secret-label"},
            {"image": encoded.getvalue(), "label": "c"},
            {"image": b"not an image", "label": "d"},
            {"image": np.zeros((8, 8, 3), np.float32) + 2.0, "label": "e"},  # Invalid, not a file.
            ImageSample(str(tmp_path / "ok.png"), "f"),
        ]

    def test_unreadable_files_and_bytes_are_dropped_and_counted(self, records):
        dataset = ImageDataset(records, on_corrupt_image="skip")
        assert dataset.labels == ["a", "c", "e", "f"]
        assert dataset.num_skipped == 3
        assert repr(dataset) == "ImageDataset(num_samples=4, num_skipped=3)"
        # Derived datasets are built from clean samples and drop nothing.
        assert dataset[1:].num_skipped == 0 and dataset.subset([0]).num_skipped == 0

    def test_raise_is_the_default_and_reads_no_file(self, records):
        dataset = ImageDataset(records)
        assert len(dataset) == len(records) and dataset.num_skipped == 0
        assert repr(dataset) == f"ImageDataset(num_samples={len(records)})"
        assert ImageDataset(records, on_corrupt_image="raise").num_skipped == 0

    def test_skip_report_is_a_count_only(self, records, caplog, logged_messages):
        with caplog.at_level(logging.INFO):
            ImageDataset(records, on_corrupt_image="skip")
        assert logged_messages() == "Skipped 3 of 7 records whose image could not be read"
        assert "99887766" not in caplog.text and "secret-label" not in caplog.text

    def test_nothing_is_logged_when_nothing_is_skipped(self, caplog):
        with caplog.at_level(logging.INFO):
            ImageDataset(
                [{"image": np.zeros((4, 4), np.uint8), "label": 1}], on_corrupt_image="skip"
            )
        assert not caplog.records

    def test_from_lists_passes_the_policy_through(self, tmp_path):
        dataset = ImageDataset.from_lists(
            [tmp_path / "missing.png", np.zeros((4, 4), np.uint8)], [0, 1], on_corrupt_image="skip"
        )
        assert dataset.labels == [1] and dataset.num_skipped == 1

    def test_directories_still_raise(self, tmp_path):
        with pytest.raises(ImageInputError, match="could not be opened"):
            ImageDataset([{"image": tmp_path, "label": 0}], on_corrupt_image="skip")

    @pytest.mark.parametrize("policy", ["ignore", None, True])
    def test_unknown_policy_is_rejected(self, policy):
        with pytest.raises(ConfigurationError, match="on_corrupt_image"):
            ImageDataset([], on_corrupt_image=policy)


@pytest.mark.parametrize(
    "method, package, extra",
    [("from_pandas", "pandas", "pandas"), ("from_hf_dataset", "datasets", "datasets")],
)
def test_a_missing_extra_names_the_package_to_install(monkeypatch, method, package, extra):
    monkeypatch.setitem(sys.modules, package, None)  # Makes `import package` fail.
    with pytest.raises(ConfigurationError, match=rf"pip install 'embedfit\[{extra}\]'"):
        getattr(ImageDataset, method)(object())


def test_skipping_a_whole_class_is_caught_by_the_trainer(tmp_path, bar_dataset):
    images, labels = bar_dataset
    records = [
        {"image": tmp_path / "missing.png" if label == labels[0] else image, "label": label}
        for image, label in zip(images, labels, strict=True)
    ]
    train = ImageDataset(records, on_corrupt_image="skip")
    assert labels[0] not in train.labels
    model = EmbedFitModel.from_pretrained("dummy/tiny", labels=sorted(set(labels)))
    with pytest.raises(ConfigurationError, match="have no samples in train_dataset"):
        Trainer(model, TrainingArguments(training_mode="frozen"), train)
