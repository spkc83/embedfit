import contextlib
import io
from datetime import datetime

import numpy as np
import pytest
import torch
from PIL import Image

from embedfit import ImageDataset
from embedfit.data import to_rgb_tensor
from embedfit.exceptions import ConfigurationError, ImageInputError
from embedfit.testing import make_bar_images

datasets = pytest.importorskip("datasets")


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


@pytest.fixture
def bars():
    return make_bar_images(2, num_classes=3, seed=0)


def _image_dataset(images, labels, **extra):
    features = datasets.Features(
        {
            "image": datasets.Image(),
            "label": datasets.ClassLabel(names=sorted(set(labels))),
            **{name: feature for name, (feature, _) in extra.items()},
        }
    )
    names = sorted(set(labels))
    return datasets.Dataset.from_dict(
        {
            "image": images,
            "label": [names.index(label) for label in labels],
            **{name: values for name, (_, values) in extra.items()},
        },
        features=features,
    )


def test_image_feature_is_stored_as_encoded_bytes(bars):
    images, labels = bars
    dataset = ImageDataset.from_hf_dataset(_image_dataset(images, labels))
    assert all(type(image) is bytes for image in dataset.images)  # Never decoded up front.
    for stored, original in zip(dataset.images, images, strict=True):
        assert torch.equal(to_rgb_tensor(stored), to_rgb_tensor(original))
    # ClassLabel columns keep their integer values; int2str gives the names.
    assert dataset.labels == [sorted(set(labels)).index(label) for label in labels]


def test_row_selection_is_respected(bars):
    images, labels = bars
    source = _image_dataset(images, labels).shuffle(seed=0).select([3, 0])
    dataset = ImageDataset.from_hf_dataset(source)
    assert dataset.labels == source["label"]
    for stored, row in zip(dataset.images, source, strict=True):
        assert torch.equal(to_rgb_tensor(stored), to_rgb_tensor(row["image"]))


def test_every_field_can_be_mapped():
    source = datasets.Dataset.from_dict(
        {
            "path": ["a.png", "b.png"],
            "y": ["risk", "legitimate"],
            "id": ["s1", None],
            "account": ["g1", "g2"],
            "seen": [datetime(2024, 1, 1), None],
            "value": [12.5, None],
            "branch": ["north", None],
        }
    )
    dataset = ImageDataset.from_hf_dataset(
        source,
        columns={
            "image": "path",
            "label": "y",
            "sample_id": "id",
            "group_id": "account",
            "timestamp": "seen",
            "amount": "value",
        },
        metadata_columns=["branch"],
    )
    first, second = dataset
    assert dataset.images == ["a.png", "b.png"]  # Path columns pass through.
    assert (first.sample_id, first.group_id, first.amount) == ("s1", "g1", 12.5)
    assert first.timestamp == datetime(2024, 1, 1) and dict(first.metadata) == {"branch": "north"}
    assert (second.sample_id, second.timestamp, second.amount) == (None, None, None)
    assert dict(second.metadata) == {"branch": None}


@pytest.mark.parametrize("output_format", [None, "numpy", "torch"])
def test_array_columns_use_the_declared_dtype_whatever_the_format(output_format):
    # datasets 4.5's numpy, torch, and jax formatters fail once torchvision 0.27 is imported;
    # the adapter must not depend on them.
    with contextlib.suppress(ImportError):
        import torchvision  # noqa: F401

    arrays = [np.full((4, 4, 3), value, np.uint8) for value in (0, 255)]
    source = datasets.Dataset.from_dict(
        {"image": arrays, "label": [0, 1]},
        features=datasets.Features(
            {"image": datasets.Array3D((4, 4, 3), "uint8"), "label": datasets.Value("int64")}
        ),
    ).with_format(output_format)
    dataset = ImageDataset.from_hf_dataset(source)
    assert dataset.labels == [0, 1] and all(type(label) is int for label in dataset.labels)
    assert all(image.dtype == np.uint8 for image in dataset.images)
    assert torch.equal(to_rgb_tensor(dataset.images[1]), torch.ones(3, 4, 4))


def test_nested_list_columns_become_arrays_of_the_declared_dtype():
    grey = [[[0.0, 1.0], [1.0, 0.0]]]
    source = datasets.Dataset.from_dict(
        {"image": grey, "label": ["a"]},
        features=datasets.Features(
            {
                "image": datasets.List(datasets.List(datasets.Value("float32"))),
                "label": datasets.Value("string"),
            }
        ),
    )
    (image,) = ImageDataset.from_hf_dataset(source).images
    assert image.dtype == np.float32 and image.shape == (2, 2)


def test_a_torch_formatted_dataset_keeps_python_labels(bars):
    source = _image_dataset(*bars).with_format("torch")
    dataset = ImageDataset.from_hf_dataset(source)
    assert all(type(label) is int for label in dataset.labels)
    assert sorted(set(dataset.labels)) == [0, 1, 2]
    assert all(isinstance(image, bytes) for image in dataset.images)


def test_a_list_of_images_per_row_is_rejected(bars):
    images, _ = bars
    source = datasets.Dataset.from_dict(
        {"image": [[images[0]], [images[1]]], "label": [0, 1]},
        features=datasets.Features(
            {"image": datasets.List(datasets.Image()), "label": datasets.Value("int64")}
        ),
    )
    with pytest.raises(ConfigurationError, match="one image per row"):
        ImageDataset.from_hf_dataset(source)


def test_nan_becomes_none_as_for_pandas():
    source = datasets.Dataset.from_dict(
        {
            "image": ["a.png", "b.png", "c.png"],
            "label": [0, 1, 1],
            "amount": [1.5, float("nan"), None],
            "score": [float("nan"), 0.5, 0.25],
        }
    )
    dataset = ImageDataset.from_hf_dataset(
        source, columns={"amount": "amount"}, metadata_columns=["score"]
    )
    assert [sample.amount for sample in dataset] == [1.5, None, None]
    assert [sample.metadata["score"] for sample in dataset] == [None, 0.5, 0.25]
    nan_label = datasets.Dataset.from_dict({"image": ["a.png"], "label": [float("nan")]})
    with pytest.raises(ConfigurationError, match="Record 0 has no label"):
        ImageDataset.from_hf_dataset(nan_label)


def test_corrupt_image_payload(bars):
    images, labels = bars
    source = datasets.Dataset.from_dict(
        {"image": [{"bytes": b"not an image", "path": None}, images[0]], "label": [0, 1]},
        features=datasets.Features({"image": datasets.Image(), "label": datasets.Value("int64")}),
    )
    with pytest.raises(Exception, match="cannot identify image"):
        ImageDataset(source)  # Without the adapter the payload is decoded while reading rows.
    raised = ImageDataset.from_hf_dataset(source)
    with pytest.raises(ImageInputError, match="could not be decoded"):
        to_rgb_tensor(raised.images[0])
    skipped = ImageDataset.from_hf_dataset(source, on_corrupt_image="skip")
    assert skipped.labels == [1] and skipped.num_skipped == 1


def test_relative_path_without_bytes(bars):
    source = datasets.Dataset.from_dict(
        {"image": [{"bytes": None, "path": "page.png"}, bars[0][0]], "label": [0, 1]},
        features=datasets.Features({"image": datasets.Image(), "label": datasets.Value("int64")}),
    )
    with pytest.raises(ImageInputError, match="Record 0: .*relative path"):
        ImageDataset.from_hf_dataset(source)
    skipped = ImageDataset.from_hf_dataset(source, on_corrupt_image="skip")
    assert skipped.labels == [1] and skipped.num_skipped == 1


def test_invalid_sources_and_columns(bars):
    source = _image_dataset(*bars)
    with pytest.raises(ConfigurationError, match="select a split"):
        ImageDataset.from_hf_dataset(datasets.DatasetDict({"train": source}))
    with pytest.raises(ConfigurationError, match="map-style"):
        ImageDataset.from_hf_dataset(source.to_iterable_dataset())
    with pytest.raises(ConfigurationError, match="expects a Hugging Face datasets.Dataset"):
        ImageDataset.from_hf_dataset([{"image": "a.png", "label": 0}])
    with pytest.raises(ConfigurationError, match=r"column 'img' .*not in the dataset"):
        ImageDataset.from_hf_dataset(source, columns={"image": "img"})


def test_a_missing_label_names_the_position_only():
    source = datasets.Dataset.from_dict({"image": ["a.png", "b.png"], "label": ["risk", None]})
    with pytest.raises(ConfigurationError, match="Record 1 has no label") as excinfo:
        ImageDataset.from_hf_dataset(source)
    assert "risk" not in str(excinfo.value)
