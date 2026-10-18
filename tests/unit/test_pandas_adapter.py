import io
import logging
from datetime import datetime

import numpy as np
import pytest
import torch
from PIL import Image

from embedfit import EmbedFitModel, ImageDataset, Trainer, TrainingArguments
from embedfit.data import to_rgb_tensor
from embedfit.exceptions import ConfigurationError, ImageInputError
from embedfit.testing import BAR_CLASSES, make_bar_images

pd = pytest.importorskip("pandas")


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def test_default_columns_and_ignored_extra_columns():
    images, labels = make_bar_images(2, num_classes=2, seed=0)
    frame = pd.DataFrame({"image": images, "label": labels, "acct": ["1", "2", "3", "4"]})
    dataset = ImageDataset.from_pandas(frame)
    assert dataset.labels == labels and dataset.images == images
    assert all(sample.metadata is None and sample.sample_id is None for sample in dataset)


def test_every_field_can_be_mapped_and_missing_values_become_none():
    frame = pd.DataFrame(
        {
            "img": [np.zeros((4, 4, 3), np.uint8)] * 3,
            "y": np.array([0, 1, 1]),
            "id": ["s1", "s2", None],
            "account": ["g1", np.nan, "g2"],
            "seen": pd.to_datetime(["2024-01-01", None, "2024-03-01"]),
            "value": [12.5, np.nan, 3.0],
            "branch": ["north", "south", None],
            "score": [0.5, 0.25, 0.125],
        },
        index=["acct-1", "acct-2", "acct-3"],
    )
    dataset = ImageDataset.from_pandas(
        frame,
        columns={
            "image": "img",
            "label": "y",
            "sample_id": "id",
            "group_id": "account",
            "timestamp": "seen",
            "amount": "value",
        },
        metadata_columns=["branch", "score", "account"],
    )
    first, second, third = dataset
    assert dataset.labels == [0, 1, 1] and all(type(label) is int for label in dataset.labels)
    assert (first.sample_id, first.group_id, first.amount) == ("s1", "g1", 12.5)
    assert type(first.amount) is float
    assert first.timestamp == datetime(2024, 1, 1) and isinstance(first.timestamp, datetime)
    assert dict(first.metadata) == {"branch": "north", "score": 0.5, "account": "g1"}
    assert type(first.metadata["score"]) is float
    assert (second.group_id, second.timestamp, second.amount) == (None, None, None)
    assert dict(second.metadata)["account"] is None
    assert third.sample_id is None and dict(third.metadata)["branch"] is None


def test_image_column_kinds(tmp_path):
    image = Image.fromarray(np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3))
    image.save(tmp_path / "image.png")
    encoded = _png(image)
    kinds = [
        tmp_path / "image.png",
        str(tmp_path / "image.png"),
        image,
        np.asarray(image),
        torch.from_numpy(np.array(image)).permute(2, 0, 1).contiguous(),
        encoded,
        {"bytes": encoded, "path": "relative/name.png"},  # Bytes win over the path.
        {"bytes": None, "path": str(tmp_path / "image.png")},  # Absolute path, no bytes.
    ]
    frame = pd.DataFrame({"image": pd.Series(kinds, dtype=object), "label": range(len(kinds))})
    dataset = ImageDataset.from_pandas(frame)
    assert dataset.images[5] == encoded and dataset.images[6] == encoded
    assert dataset.images[7] == tmp_path / "image.png"
    expected = to_rgb_tensor(image)
    for stored in dataset.images:
        assert torch.equal(to_rgb_tensor(stored), expected)


def test_relative_path_without_bytes_is_unreadable(tmp_path):
    frame = pd.DataFrame(
        {
            "image": [{"bytes": None, "path": "image.png"}, np.zeros((4, 4), np.uint8)],
            "label": [0, 1],
        }
    )
    with pytest.raises(ImageInputError, match="Record 0: .*relative path"):
        ImageDataset.from_pandas(frame)
    dataset = ImageDataset.from_pandas(frame, on_corrupt_image="skip")
    assert dataset.labels == [1] and dataset.num_skipped == 1
    # An invalid policy is reported before any record is read.
    with pytest.raises(ConfigurationError, match="on_corrupt_image"):
        ImageDataset.from_pandas(frame, on_corrupt_image="drop")


def test_skip_policy_is_passed_through(tmp_path):
    (tmp_path / "broken.png").write_bytes(b"not an image")
    frame = pd.DataFrame(
        {
            "image": [tmp_path / "broken.png", b"garbage", np.zeros((4, 4), np.uint8)],
            "label": [0, 1, 2],
        }
    )
    assert len(ImageDataset.from_pandas(frame)) == 3
    dataset = ImageDataset.from_pandas(frame, on_corrupt_image="skip")
    assert dataset.labels == [2] and dataset.num_skipped == 2


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"columns": {"image": "missing"}}, r"column 'missing' .*not in the DataFrame"),
        ({"columns": {"imgae": "image"}}, r"Unknown field 'imgae'"),
        ({"columns": {"metadata": "acct"}}, r"metadata_columns"),
        ({"columns": {"sample_id": "label"}}, r"column 'label' is mapped to more than one field"),
        ({"metadata_columns": ["nope"]}, r"column 'nope' .*not in the DataFrame"),
        ({"metadata_columns": "acct"}, r"metadata_columns must be a sequence of column names"),
        ({"on_corrupt_image": "drop"}, r"on_corrupt_image"),
    ],
)
def test_invalid_column_mappings(kwargs, match):
    frame = pd.DataFrame({"image": ["a.png"], "label": [0], "acct": ["12345678"]})
    with pytest.raises(ConfigurationError, match=match):
        ImageDataset.from_pandas(frame, **kwargs)


def test_missing_label_names_the_row_position_not_the_index():
    frame = pd.DataFrame(
        {"image": ["a.png", "b.png"], "label": ["risk", None]}, index=["acct-11", "acct-99887766"]
    )
    with pytest.raises(ConfigurationError, match="Record 1 has no label") as excinfo:
        ImageDataset.from_pandas(frame)
    assert "99887766" not in str(excinfo.value) and "risk" not in str(excinfo.value)


def test_non_dataframe_is_rejected():
    with pytest.raises(ConfigurationError, match="expects a pandas DataFrame, got dict"):
        ImageDataset.from_pandas({"image": ["a.png"], "label": [0]})


def test_trains_end_to_end(caplog, logged_messages):
    images, labels = make_bar_images(8, num_classes=3, seed=0)
    frame = pd.DataFrame({"picture": images, "kind": labels, "acct": "acct-99887766"})
    train = ImageDataset.from_pandas(
        frame, columns={"image": "picture", "label": "kind"}, metadata_columns=["acct"]
    )
    model = EmbedFitModel.from_pretrained("dummy/tiny", labels=list(BAR_CLASSES[:3]))
    with caplog.at_level(logging.DEBUG):
        trainer = Trainer(model, TrainingArguments(training_mode="frozen"), train)
        trainer.train()
        assert trainer.evaluate(train)["accuracy"] > 0.5
    assert "99887766" not in logged_messages()


def test_a_duplicated_column_name_is_rejected():
    frame = pd.DataFrame([["a.png", 0, 1]], columns=["image", "label", "label"])
    with pytest.raises(ConfigurationError, match="column 'label' appears more than once"):
        ImageDataset.from_pandas(frame)
    # A duplicated column that is not used does not matter.
    frame = pd.DataFrame([["a.png", 0, 1, 2]], columns=["image", "label", "x", "x"])
    assert ImageDataset.from_pandas(frame).labels == [0]


def test_zero_dimensional_labels_are_unwrapped_and_unhashable_labels_rejected():
    frame = pd.DataFrame({"image": ["a.png", "b.png"], "label": [torch.tensor(1), np.array(1)]})
    assert ImageDataset.from_pandas(frame).labels == [1, 1]
    frame = pd.DataFrame({"image": ["a.png", "b.png"], "label": [0, ["risk"]]})
    with pytest.raises(ConfigurationError, match="Record 1 has an unhashable label of type list"):
        ImageDataset.from_pandas(frame)
