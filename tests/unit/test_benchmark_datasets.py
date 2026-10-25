"""benchmarks/few_shot.py dataset loaders, on fake downloads: splits, caps, and leakage."""

import argparse
import importlib.util
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from embedfit import ImageDataset

torchvision_datasets = pytest.importorskip("torchvision.datasets")

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"
NAMES = ["banded", "dotted", "zigzag"]


@pytest.fixture(scope="module")
def few_shot():
    spec = importlib.util.spec_from_file_location("few_shot_datasets", BENCHMARKS / "few_shot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def quality():
    sys.path.insert(0, str(BENCHMARKS))  # The script imports few_shot from its own folder.
    try:
        spec = importlib.util.spec_from_file_location(
            "embedding_quality_datasets", BENCHMARKS / "embedding_quality.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(BENCHMARKS))
    return module


def picture(code: int) -> Image.Image:
    """An 8x8 image whose pixels differ for every ``code``."""
    return Image.new("RGB", (8, 8), (code % 256, code // 256 % 256, code // 65536))


def fake_torchvision(per_split: dict[str, int], files_attribute: str, *, duplicate=None):
    """A torchvision dataset class whose splits hold ``per_split[split]`` images per class.

    Every image is unique, except that with ``duplicate=(split, other)`` the first image of
    ``split`` has the pixels of the first image of ``other``.
    """
    calls = []

    class Fake:
        def __init__(self, root, split, download, **options):
            calls.append({"root": root, "split": split, "download": download, **options})
            offset = 1000 * list(per_split).index(split)
            folder = Path(root) / "fake" / split
            folder.mkdir(parents=True, exist_ok=True)
            files, labels = [], []
            for position in range(per_split[split] * len(NAMES)):
                path = folder / f"{position}_{position % len(NAMES)}.png"  # Its label.
                code = offset + position
                if duplicate and duplicate[0] == split and position == 0:
                    code = 1000 * list(per_split).index(duplicate[1])
                picture(code).save(path)
                files.append(path)
                labels.append(position % len(NAMES))
            setattr(self, files_attribute, files)
            self._labels = labels
            self.classes = list(NAMES)

    Fake.calls = calls
    return Fake


def load(few_shot, name, root, *, pool=100, evaluation=100, evaluate_on="test", seed=0):
    return few_shot.LOADERS[name](
        root, pool=pool, evaluation=evaluation, evaluate_on=evaluate_on, seed=seed
    )


def check_labels(items, classes):
    """Every drawn file's label is the class its file name records."""
    for path, label in items:
        assert label == classes[int(path.stem.split("_")[1])]


def counts(items):
    return Counter(label for _, label in items)


def paths(items):
    return {path for path, _ in items}


@pytest.fixture
def dtd(monkeypatch):
    fake = fake_torchvision({"train": 5, "val": 4, "test": 3}, "_image_files")
    monkeypatch.setattr(torchvision_datasets, "DTD", fake)
    return fake


@pytest.fixture
def pets(monkeypatch):
    fake = fake_torchvision({"trainval": 24, "test": 6}, "_images")
    monkeypatch.setattr(torchvision_datasets, "OxfordIIITPet", fake)
    return fake


@pytest.fixture
def flowers(monkeypatch):
    fake = fake_torchvision({"train": 2, "val": 2, "test": 13}, "_image_files")
    monkeypatch.setattr(torchvision_datasets, "Flowers102", fake)
    return fake


# DTD: the official partition's three splits.


def test_dtd_uses_its_official_partition_and_splits(few_shot, dtd, tmp_path):
    splits, classes = load(few_shot, "dtd", tmp_path, evaluate_on="validation")
    assert classes == NAMES
    assert set(splits) == {"pool", "validation"}
    assert counts(splits["pool"]) == dict.fromkeys(NAMES, 5)  # The cap of 100 takes all.
    assert counts(splits["validation"]) == dict.fromkeys(NAMES, 4)
    assert {call["split"] for call in dtd.calls} == {"train", "val", "test"}
    assert {call["partition"] for call in dtd.calls} == {few_shot.DTD_PARTITION}
    assert all(call["download"] and call["root"] == str(tmp_path) for call in dtd.calls)
    assert all(path.parent.name == "train" for path in paths(splits["pool"]))
    assert all(isinstance(path, Path) for path in paths(splits["validation"]))  # Not decoded.
    for items in splits.values():
        check_labels(items, NAMES)


def test_counts_cap_every_class_and_the_draw_is_seeded(few_shot, dtd, tmp_path):
    splits, _ = load(few_shot, "dtd", tmp_path, pool=2, evaluation=1, seed=4)
    assert counts(splits["pool"]) == dict.fromkeys(NAMES, 2)
    assert counts(splits["test"]) == dict.fromkeys(NAMES, 1)
    assert load(few_shot, "dtd", tmp_path, pool=2, evaluation=1, seed=4)[0] == splits
    others = [load(few_shot, "dtd", tmp_path, pool=2, evaluation=1, seed=s)[0] for s in range(8)]
    assert len({frozenset(paths(other["pool"])) for other in others}) > 1


def test_the_pool_does_not_depend_on_the_evaluated_split(few_shot, dtd, tmp_path):
    on_validation, _ = load(few_shot, "dtd", tmp_path, pool=3, evaluate_on="validation")
    on_test, _ = load(few_shot, "dtd", tmp_path, pool=3, evaluate_on="test")
    assert on_validation["pool"] == on_test["pool"]


# Pets: validation carved off trainval.


def test_pets_carves_validation_off_trainval(few_shot, pets, tmp_path):
    validation, classes = load(few_shot, "pets", tmp_path, evaluate_on="validation")
    test, _ = load(few_shot, "pets", tmp_path, evaluate_on="test")
    assert classes == NAMES
    carved = few_shot.CARVED_VALIDATION["pets"]
    assert counts(validation["validation"]) == dict.fromkeys(NAMES, carved)
    assert counts(validation["pool"]) == dict.fromkeys(NAMES, 24 - carved)
    assert not paths(validation["pool"]) & paths(validation["validation"])
    assert all(path.parent.name == "trainval" for path in paths(validation["validation"]))
    assert all(path.parent.name == "test" for path in paths(test["test"]))
    assert counts(test["test"]) == dict.fromkeys(NAMES, 6)
    assert test["pool"] == validation["pool"]
    for items in (*validation.values(), test["test"]):
        check_labels(items, NAMES)


def test_the_carved_validation_follows_the_split_seed(few_shot, pets, tmp_path):
    carve = [load(few_shot, "pets", tmp_path, evaluate_on="validation", seed=s)[0] for s in (0, 1)]
    assert paths(carve[0]["validation"]) != paths(carve[1]["validation"])


def test_a_class_too_small_to_carve_from_exits(few_shot, monkeypatch, tmp_path):
    fake = fake_torchvision({"trainval": 20, "test": 6}, "_images")
    monkeypatch.setattr(torchvision_datasets, "OxfordIIITPet", fake)
    with pytest.raises(SystemExit, match="too few to carve 20"):
        load(few_shot, "pets", tmp_path)


# Flowers-102: train and val as the pool, validation carved off test.


def test_flowers_pools_train_and_val_and_splits_test(few_shot, flowers, tmp_path):
    validation, classes = load(few_shot, "flowers", tmp_path, evaluate_on="validation")
    test, _ = load(few_shot, "flowers", tmp_path, evaluate_on="test")
    assert classes[:3] == ["flower_001", "flower_002", "flower_003"]
    assert len(classes) == few_shot.FLOWERS_CLASSES
    assert counts(validation["pool"]) == {name: 4 for name in classes[:3]}
    assert {path.parent.name for path in paths(validation["pool"])} == {"train", "val"}
    carved = few_shot.CARVED_VALIDATION["flowers"]
    assert counts(validation["validation"]) == {name: carved for name in classes[:3]}
    assert counts(test["test"]) == {name: 13 - carved for name in classes[:3]}
    assert not paths(validation["validation"]) & paths(test["test"])
    assert all(path.parent.name == "test" for path in paths(validation["validation"]))
    for items in (*validation.values(), test["test"]):
        check_labels(items, classes)


# PCam: a pinned Hugging Face dataset; only the needed splits download.


PCAM_LABELS = [position % 3 == 0 for position in range(18)]  # 6 True, 12 False.


def code(image):
    red, green, blue = image.getpixel((0, 0))
    return red + 256 * green + 65536 * blue


class FakeRows:
    """Parquet rows whose images record their row position in their pixels."""

    def __init__(self, labels, offset):
        self.labels, self.offset = labels, offset

    def __getitem__(self, key):
        if key == "label":
            return list(self.labels)
        return {"image": picture(self.offset + key).convert("RGBA"), "label": self.labels[key]}


@pytest.fixture
def pcam(monkeypatch):
    datasets = pytest.importorskip("datasets")
    hub = pytest.importorskip("huggingface_hub")
    calls, fetched = [], []

    def hf_hub_download(repo_id, filename, **options):
        fetched.append({"repo_id": repo_id, "filename": filename, **options})
        return f"cached/{filename}"

    def load_dataset(name, **options):
        calls.append({"name": name, **options})
        return {
            split: FakeRows(PCAM_LABELS, 1000 * index)
            for index, split in enumerate(options["data_files"])
        }

    monkeypatch.setattr(hub, "hf_hub_download", hf_hub_download)
    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    return calls, fetched


def test_pcam_downloads_the_pinned_pool_and_evaluated_split_only(few_shot, pcam, tmp_path):
    splits, classes = load(few_shot, "pcam", tmp_path, pool=4, evaluation=8, evaluate_on="test")
    assert classes == few_shot.PCAM_CLASSES == ["normal", "metastasis"]
    calls, fetched = pcam
    wanted = [path for split in ("pool", "test") for path in few_shot.PCAM_FILES[split]]
    assert [entry["filename"] for entry in fetched] == wanted
    for entry in fetched:
        assert entry["repo_id"] == few_shot.PCAM_DATASET
        assert (entry["repo_type"], entry["revision"]) == ("dataset", few_shot.PCAM_REVISION)
    (call,) = calls
    assert call["name"] == "parquet"
    assert call["cache_dir"] == str(tmp_path / "pcam")
    assert call["data_files"] == {
        split: [f"cached/{path}" for path in few_shot.PCAM_FILES[split]]
        for split in ("pool", "test")
    }
    assert counts(splits["pool"]) == {"normal": 4, "metastasis": 4}
    assert counts(splits["test"]) == {"normal": 8, "metastasis": 6}  # Capped; 6 of True exist.
    for offset, items in ((0, splits["pool"]), (1000, splits["test"])):
        for image, label in items:  # Label True is metastasis.
            assert image.mode == "RGB"
            assert label == ("metastasis" if PCAM_LABELS[code(image) - offset] else "normal")


def test_the_dataset_revision_is_part_of_the_configuration(few_shot):
    for dataset, revision in [
        ("pcam", few_shot.PCAM_REVISION),
        ("rvl-cdip", few_shot.RVL_CDIP_REVISION),
        ("dtd", None),
    ]:
        args = few_shot.parse_arguments(
            ["--dataset", dataset, "--evaluate-on", "test", "--output", "results.json"]
        )
        assert few_shot.resume_configuration(args)["dataset_revision"] == revision


# Leakage: duplicate handling applies to the new loaders.


def test_an_evaluation_image_that_matches_the_pool_is_dropped(few_shot, monkeypatch, tmp_path):
    fake = fake_torchvision(
        {"train": 3, "val": 3, "test": 3}, "_image_files", duplicate=("test", "train")
    )
    monkeypatch.setattr(torchvision_datasets, "DTD", fake)
    args = argparse.Namespace(
        dataset="dtd",
        data_root=tmp_path,
        pool_per_class=100,
        eval_per_class=100,
        evaluate_on="test",
        split_seed=0,
        max_image_side=None,
    )
    splits, labels, duplicates, _ = few_shot.load_splits(args)
    assert (duplicates, len(splits["test"]), labels) == (1, 8, NAMES)


# The embedding-quality tasks on a two-class dataset.


def two_classes(per_class):
    images = [picture(code) for code in range(2 * per_class)]
    return ImageDataset.from_lists(images, [few_shot_name(i) for i in range(2 * per_class)])


def few_shot_name(position):
    return ["normal", "metastasis"][position % 2]


def test_open_set_on_two_classes_exits_with_the_classes_needed(quality):
    with pytest.raises(SystemExit, match="leaves 0 of the 2 classes seen.*at least 4 classes"):
        quality.partition_classes(["normal", "metastasis"], 2, seed=0)
    with pytest.raises(SystemExit, match="at least 5 classes"):
        quality.partition_classes(["normal", "metastasis"], 3, seed=0)


def test_open_set_on_pcam_is_rejected_before_any_download(quality, capsys):
    required = ["--dataset", "pcam", "--evaluate-on", "test", "--output", "results.json"]
    with pytest.raises(SystemExit):
        quality.parse_arguments(required)
    assert "needs at least 5 classes" in capsys.readouterr().err
    assert quality.parse_arguments([*required, "--tasks", "rare_positive"]).dataset == "pcam"


def test_every_loader_has_a_class_count(few_shot):
    assert set(few_shot.CLASS_COUNTS) == set(few_shot.LOADERS)


def test_rare_positive_runs_on_two_classes(quality):
    pool = two_classes(10)
    positive, train = quality.rare_positive_sample(
        pool, ["normal", "metastasis"], shots=2, negatives=7, seed=0
    )
    assert Counter(train.labels) == {"positive": 2, "negative": 7}
    negative_class = "normal" if positive == "metastasis" else "metastasis"
    pixels = {picture(code).tobytes(): few_shot_name(code) for code in range(20)}
    drawn = [pixels[image.tobytes()] for image in train.images]
    assert drawn == [positive] * 2 + [negative_class] * 7


def test_rare_positive_on_two_classes_exits_when_the_pool_is_short(quality):
    with pytest.raises(SystemExit, match="--negatives asks for 11 images of a class that has 10"):
        quality.rare_positive_sample(
            two_classes(10), ["normal", "metastasis"], shots=2, negatives=11, seed=0
        )


def test_every_loader_has_a_default_image_size(few_shot):
    assert set(few_shot.DEFAULT_IMAGE_SIZE) == set(few_shot.LOADERS)
    assert np.all([size % 14 == 0 for size in few_shot.DEFAULT_IMAGE_SIZE.values()])
