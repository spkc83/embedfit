"""Few-shot benchmark on a public image dataset with a pretrained backbone.

Compares training modes as ``FewShotEvaluator`` does, and with ``--interpolate`` also scores
every trained backbone with its weights interpolated toward the pretrained ones (WiSE-FT;
``docs/guides/benchmarking.md``). With ``--heads``, every method is scored with each named head
instead of its own (ADR 0019). Six datasets are supported, all downloaded
on first use. Everything the run downloads or derives (datasets, pretrained weights, the page
cache) goes under ``--data-root``, by default the git-ignored ``.cache/`` folder of the
repository; the ``datasets`` library, Hugging Face Hub, and torch hub caches and the temporary
folder are pointed there unless ``HF_HOME``, ``TORCH_HOME`` or ``TMPDIR`` is already set. (This
applies to this script only; the library itself uses the usual caches.) A Hugging Face login
saved in the usual cache is therefore not found: set ``HF_TOKEN`` for gated or private models.
Nothing from the data is committed:

- ``eurosat``: EuroSAT RGB land cover (10 classes, 64x64 Sentinel-2 tiles), via torchvision into
  ``--data-root``. It has no official split, so disjoint per-class pool, validation, and test
  sets are drawn with ``--split-seed``.
- ``rvl-cdip``: document images from RVL-CDIP (16 classes), the Hugging Face dataset
  ``jinhybr/rvl_cdip_400_train_val_test`` (400 training, 100 validation, and 100 test images
  per class; ``pip install -e ".[hf,datasets]"``), downloaded by the ``datasets`` library.
  Its own splits are kept: the pool is drawn from the training split. Pages are
  converted to grayscale, shrunk to a shorter side of 512 px (still larger than the model
  input) and written once as PNG files under ``--data-root``; the run reads them from disk, so
  memory holds only the model and the current batch.
- ``dtd``: Describable Textures (47 classes), via torchvision. Official partition 1: the pool is
  its training split, and validation and test are its own (40 images per class each).
- ``pets``: Oxford-IIIT Pets (37 breeds), via torchvision. Its trainval split is the pool, less
  20 images per class carved off with ``--split-seed`` as the validation split; test is its own.
- ``flowers``: Oxford Flowers-102, via torchvision. Its train and val splits (10 images per
  class each) together are the pool, so that 16 shots fit; 10 images per class carved off its
  test split with ``--split-seed`` are the validation split, and the rest of it is the test split.
- ``pcam``: PatchCamelyon (2 classes, 96x96 histopathology patches), the Hugging Face dataset
  ``1aurent/PatchCamelyon`` at a pinned revision (``pip install -e ".[datasets]"``). The pool
  comes from the first of its 13 training shards; validation and test are its own splits.

For the last four, ``--pool-per-class`` and ``--eval-per-class`` are caps: a class with fewer
images in the split contributes all of them. ``eurosat`` and ``rvl-cdip`` exit instead.

Few-shot samples come from the pool. Tune with ``--evaluate-on validation`` and report with
``--evaluate-on test``. Only the pool and the evaluated split are loaded. An evaluation image
whose pixels exactly match a pool image is dropped, and the count is recorded.

    python benchmarks/few_shot.py --dataset eurosat --evaluate-on validation --shots 8 \\
        --seeds 0 --methods supcon --learning-rate supcon=1e-5 --output .runs/tune.json

Run it from the repository root, and keep run output in the git-ignored ``.runs/`` folder
until it is ready to commit.
"""

import argparse
import contextlib
import copy
import hashlib
import json
import logging
import math
import os
import platform
import subprocess
import tempfile
import time
from collections.abc import Callable, Collection, Iterator, Sequence
from dataclasses import replace
from importlib import metadata
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn

import embedfit
from embedfit import EmbedFitModel, FewShotEvaluator, ImageDataset, Trainer, TrainingArguments
from embedfit.augmentation import available_augmentations
from embedfit.backbones.base import TRAINABLE_PARAMETERS
from embedfit.data import to_rgb_tensor
from embedfit.exceptions import ConfigurationError
from embedfit.heads import LinearHead, create_head
from embedfit.sampling import sample_few_shot
from embedfit.utils.device import resolve_device
from embedfit.utils.seed import set_seed

logger = logging.getLogger(__name__)

TRAINED_METHODS = ("end_to_end", "supcon", "pair", "ce_supcon")
DEFAULT_METHODS = ("frozen", "end_to_end", "supcon", "pair")  # Kept, so that old runs resume.
METHOD_MODES = {"ce_supcon": "end_to_end"}  # Methods that are not a training mode of their own.
CONTRASTIVE_METHODS = ("supcon", "ce_supcon")
DEFAULT_CONTRASTIVE_WEIGHT = 1.0
HEADS = ("linear", "logistic_regression")
SAMPLER_DEFAULTS = {"end_to_end": "random", "supcon": "balanced", "ce_supcon": "balanced"}
DOCUMENT_SHORT_SIDE = 512
RVL_CDIP_DATASET = "jinhybr/rvl_cdip_400_train_val_test"
RVL_CDIP_REVISION = (
    "13462a42e3375e80a9bd46a64c58e1bcaba77874"  # Pinned: the page cache is keyed on it.
)
DTD_PARTITION = 1  # Of DTD's ten official train/val/test partitions.
CARVED_VALIDATION = {"pets": 20, "flowers": 10}  # Validation images carved off per class.
FLOWERS_CLASSES = 102
PCAM_DATASET = "1aurent/PatchCamelyon"
PCAM_REVISION = "e4bd149e7a868a9d811fdd9f9a9fb78c05c104ab"
PCAM_FILES = {  # The pool comes from the first of the 13 training shards (about 20,000 images).
    "pool": ["data/train-00000-of-00013-4717c3cf92578c96.parquet"],
    "validation": [
        "data/valid-00000-of-00002-0e1a29e0620125c6.parquet",
        "data/valid-00001-of-00002-aad8011eb887c9d9.parquet",
    ],
    "test": [
        "data/test-00000-of-00002-bb04e6313f58efa0.parquet",
        "data/test-00001-of-00002-3bfa172e8818685a.parquet",
    ],
}
PCAM_CLASSES = ["normal", "metastasis"]  # The dataset's labels False and True.
CLASS_COUNTS = {
    "eurosat": 10,
    "rvl-cdip": 16,
    "dtd": 47,
    "pets": 37,
    "flowers": FLOWERS_CLASSES,
    "pcam": len(PCAM_CLASSES),
}
DATASET_REVISIONS = {"rvl-cdip": RVL_CDIP_REVISION, "pcam": PCAM_REVISION}
DEFAULT_IMAGE_SIZE = {
    "eurosat": 112,
    "rvl-cdip": 224,
    "dtd": 224,
    "pets": 224,
    "flowers": 224,
    "pcam": 112,
}
REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPOSITORY / ".cache"  # Git-ignored.
MAX_SEED = 2**32 - 1  # The largest seed set_seed accepts.
RECORDED_PACKAGES = ("torchvision", "transformers", "timm", "scikit-learn")

Split = list[tuple[Image.Image | Path, str]]


def _per_class(labels: np.ndarray, count: int, rng: np.random.Generator) -> list[int]:
    """Up to ``count`` random positions of every class, in class order."""
    chosen = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        if count > len(members):
            raise SystemExit(f"Class {label} has {len(members)} images, fewer than {count}.")
        chosen.extend(rng.permutation(members)[:count].tolist())
    return chosen


def eurosat_splits(
    root: Path, *, pool: int, evaluation: int, evaluate_on: str, seed: int
) -> tuple[dict[str, Split], list[str]]:
    from torchvision.datasets import EuroSAT

    source = EuroSAT(str(root), download=True)
    targets = np.asarray(source.targets)
    rng = np.random.default_rng(seed)
    chosen: dict[str, list[int]] = {"pool": [], "validation": [], "test": []}
    for label in range(len(source.classes)):
        order = rng.permutation(np.flatnonzero(targets == label))
        cuts = np.cumsum([pool, evaluation, evaluation])
        if cuts[-1] > len(order):
            raise SystemExit(
                f"Class {label} has {len(order)} images, fewer than the {cuts[-1]} requested."
            )
        for name, part in zip(chosen, np.split(order[: cuts[-1]], cuts[:-1]), strict=True):
            chosen[name].extend(part.tolist())
    splits = {
        name: [
            (source[index][0].convert("RGB"), source.classes[source[index][1]])
            for index in chosen[name]
        ]
        for name in ("pool", evaluate_on)
    }
    return splits, list(source.classes)


def rvl_cdip_splits(
    root: Path, *, pool: int, evaluation: int, evaluate_on: str, seed: int
) -> tuple[dict[str, Split], list[str]]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            'rvl-cdip needs the datasets extra: pip install -e ".[datasets]"'
        ) from None

    source = load_dataset(RVL_CDIP_DATASET, revision=RVL_CDIP_REVISION)
    names = source["train"].features["label"].names
    cache = _page_cache(root / "rvl-cdip")

    def take(split: str, count: int, stream: int) -> Split:
        data = source[split]
        labels = np.asarray(data["label"])
        rng = np.random.default_rng([seed, stream])  # Independent of which splits are loaded.
        directory = cache / split
        directory.mkdir(parents=True, exist_ok=True)
        items: Split = []
        for position in _per_class(labels, count, rng):
            path = directory / f"{position}.png"  # Named by row position, stable for a revision.
            if not path.exists():  # Only then is the full-resolution page decoded.
                _write_page(_shrink(data[position]["image"]), path)
            items.append((path, names[int(labels[position])]))
        return items

    splits = {"pool": take("train", pool, 0)}
    splits[evaluate_on] = take(evaluate_on, evaluation, 1 if evaluate_on == "validation" else 2)
    return splits, names


def _page_cache(directory: Path) -> Path:
    """The page cache, refusing one built from another dataset revision or page size."""
    expected = {
        "dataset": RVL_CDIP_DATASET,
        "revision": RVL_CDIP_REVISION,
        "short_side": DOCUMENT_SHORT_SIDE,
    }
    manifest = directory / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != expected:
            raise SystemExit(f"{directory} holds pages of another dataset revision; move it away.")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(expected, indent=1))
    return directory


def _write_page(image: Image.Image, path: Path) -> None:
    """Write ``image`` as a PNG so that ``path`` never holds a partial file, whoever writes it."""
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.name + ".", suffix=".partial", delete=False
    ) as staging:
        image.save(staging, format="PNG")
        staging.flush()
        os.fsync(staging.fileno())
    os.replace(staging.name, path)


def _shrink(image: Image.Image) -> Image.Image:
    image = image.convert(
        "L"
    )  # Before resizing: bilevel scans would otherwise be resampled NEAREST.
    scale = DOCUMENT_SHORT_SIDE / min(image.size)
    if scale < 1:
        size = (round(image.width * scale), round(image.height * scale))
        image = image.resize(size, Image.Resampling.BICUBIC)
    return image


Source = tuple[np.ndarray, Callable[[int], Image.Image | Path]]  # Labels; the item at a position.


def _up_to_per_class(labels: np.ndarray, count: int, rng: np.random.Generator) -> list[int]:
    """Up to ``count`` random positions of every class, in class order; fewer if it has fewer."""
    return [
        position
        for label in np.unique(labels)
        for position in rng.permutation(np.flatnonzero(labels == label))[:count].tolist()
    ]


def _draw(
    sources: dict[str, Source],
    classes: list[str],
    *,
    pool: int,
    evaluation: int,
    evaluate_on: str,
    seed: int,
) -> dict[str, Split]:
    """The pool and the evaluated split, each up to its count per class from its source."""
    counts = {"pool": pool, "validation": evaluation, "test": evaluation}
    splits = {}
    for name in ("pool", evaluate_on):
        labels, item = sources[name]
        rng = np.random.default_rng([seed, list(counts).index(name)])  # Same whichever is loaded.
        chosen = _up_to_per_class(labels, counts[name], rng)
        splits[name] = [(item(position), classes[int(labels[position])]) for position in chosen]
    return splits


def _listing(source: Any) -> tuple[np.ndarray, list[Path]]:
    """The labels and image files of a torchvision dataset, without decoding any image."""
    # torchvision has no public list of a split's files; DTD and Flowers102 name it _image_files,
    # OxfordIIITPet _images, and all three name the labels _labels.
    files = getattr(source, "_image_files", None)
    if files is None:
        files = getattr(source, "_images", None)
    labels = getattr(source, "_labels", None)
    name = type(source).__name__
    if files is None or labels is None:
        raise SystemExit(
            f"This torchvision version's {name} does not list its image files and labels as "
            "the benchmark expects (_image_files or _images, and _labels); use a version "
            "the benchmark supports."
        )
    if len(files) != len(labels):
        raise SystemExit(
            f"This torchvision version's {name} lists {len(files)} image files and "
            f"{len(labels)} labels; they must correspond one to one."
        )
    return np.asarray(labels), list(files)


def _files(labels: np.ndarray, files: list[Path], keep: np.ndarray) -> Source:
    positions = np.flatnonzero(keep)
    return labels[positions], [files[position] for position in positions].__getitem__


def _carve(labels: np.ndarray, count: int, seed: int) -> np.ndarray:
    """A mask of ``count`` random positions of every class, the validation carved off a split."""
    rng = np.random.default_rng([seed, 3])
    carved = np.zeros(len(labels), dtype=bool)
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        if count >= len(members):
            raise SystemExit(f"Class {label} has {len(members)} images, too few to carve {count}.")
        carved[rng.choice(members, count, replace=False)] = True
    return carved


def dtd_splits(
    root: Path, *, pool: int, evaluation: int, evaluate_on: str, seed: int
) -> tuple[dict[str, Split], list[str]]:
    from torchvision.datasets import DTD

    sources: dict[str, Source] = {}
    for name, split in (("pool", "train"), ("validation", "val"), ("test", "test")):
        source = DTD(str(root), split=split, partition=DTD_PARTITION, download=True)
        labels, files = _listing(source)
        sources[name] = labels, files.__getitem__
    classes = list(source.classes)
    return _draw(
        sources, classes, pool=pool, evaluation=evaluation, evaluate_on=evaluate_on, seed=seed
    ), classes


def pets_splits(
    root: Path, *, pool: int, evaluation: int, evaluate_on: str, seed: int
) -> tuple[dict[str, Split], list[str]]:
    from torchvision.datasets import OxfordIIITPet

    trainval = OxfordIIITPet(str(root), split="trainval", download=True)
    labels, files = _listing(trainval)
    carved = _carve(labels, CARVED_VALIDATION["pets"], seed)
    test_labels, test_files = _listing(OxfordIIITPet(str(root), split="test", download=True))
    sources = {
        "pool": _files(labels, files, ~carved),
        "validation": _files(labels, files, carved),
        "test": (test_labels, test_files.__getitem__),
    }
    classes = list(trainval.classes)
    return _draw(
        sources, classes, pool=pool, evaluation=evaluation, evaluate_on=evaluate_on, seed=seed
    ), classes


def flowers_splits(
    root: Path, *, pool: int, evaluation: int, evaluate_on: str, seed: int
) -> tuple[dict[str, Split], list[str]]:
    from torchvision.datasets import Flowers102

    train, val, test = (
        _listing(Flowers102(str(root), split=split, download=True))
        for split in ("train", "val", "test")
    )
    labels, files = np.concatenate([train[0], val[0]]), train[1] + val[1]  # The pool.
    test_labels, test_files = test
    carved = _carve(test_labels, CARVED_VALIDATION["flowers"], seed)
    sources = {
        "pool": (labels, files.__getitem__),
        "validation": _files(test_labels, test_files, carved),
        "test": _files(test_labels, test_files, ~carved),
    }
    classes = [f"flower_{label + 1:03d}" for label in range(FLOWERS_CLASSES)]  # No names given.
    return _draw(
        sources, classes, pool=pool, evaluation=evaluation, evaluate_on=evaluate_on, seed=seed
    ), classes


def pcam_splits(
    root: Path, *, pool: int, evaluation: int, evaluate_on: str, seed: int
) -> tuple[dict[str, Split], list[str]]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit('pcam needs the datasets extra: pip install -e ".[datasets]"') from None

    from huggingface_hub import hf_hub_download

    # The parquet files of the pinned revision, fetched into the Hub cache (which a later run
    # reads with HF_HUB_OFFLINE=1) and read directly: only these download, and the dataset
    # card's split list, which names all three splits, is not checked against them.
    wanted = {
        name: [
            hf_hub_download(PCAM_DATASET, path, repo_type="dataset", revision=PCAM_REVISION)
            for path in PCAM_FILES[name]
        ]
        for name in ("pool", evaluate_on)
    }
    source = load_dataset("parquet", data_files=wanted, cache_dir=str(root / "pcam"))

    def rows(name: str) -> Source:
        data = source[name]
        return np.asarray(data["label"], dtype=int), lambda i: data[i]["image"].convert("RGB")

    sources = {name: rows(name) for name in wanted}
    return _draw(
        sources,
        PCAM_CLASSES,
        pool=pool,
        evaluation=evaluation,
        evaluate_on=evaluate_on,
        seed=seed,
    ), list(PCAM_CLASSES)


LOADERS: dict[str, Callable[..., tuple[dict[str, Split], list[str]]]] = {
    "eurosat": eurosat_splits,
    "rvl-cdip": rvl_cdip_splits,
    "dtd": dtd_splits,
    "pets": pets_splits,
    "flowers": flowers_splits,
    "pcam": pcam_splits,
}


def digest(item: Image.Image | Path) -> str:
    """SHA-256 of an image's mode, size, and pixels."""
    with Image.open(item) if isinstance(item, Path) else contextlib.nullcontext(item) as image:
        return hashlib.sha256(f"{image.mode}{image.size}".encode() + image.tobytes()).hexdigest()


def downscale(item: Image.Image | Path, side: int) -> Image.Image:
    """``item`` decoded as the library decodes it, as 8-bit RGB with its shorter side at most
    ``side``: shrunk with antialiased bicubic resampling, keeping the aspect ratio, never enlarged.
    """
    pixels = (to_rgb_tensor(item) * 255).round().to(torch.uint8)  # Exact for 8-bit sources.
    image = Image.fromarray(pixels.permute(1, 2, 0).numpy())  # [H, W, 3] uint8: RGB.
    scale = side / min(image.size)
    if scale < 1:
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(size, Image.Resampling.BICUBIC)
    return image


def drop_pool_duplicates(splits: dict[str, Split], evaluate_on: str) -> int:
    """Remove evaluation images whose pixels match a pool image; return how many."""
    pool = {digest(image) for image, _ in splits["pool"]}
    kept = [(image, label) for image, label in splits[evaluate_on] if digest(image) not in pool]
    dropped = len(splits[evaluate_on]) - len(kept)
    splits[evaluate_on] = kept
    return dropped


MEASURED = ("evaluation_duplicates_dropped", "evaluation_size")  # Known after loading.


def read_previous(output: Path, config: dict, measured: Collection[str] = MEASURED) -> dict | None:
    """Results already in ``output``, or ``None``; exit if they are for another configuration.

    ``measured`` names the configuration entries that are only known after loading the data,
    and are left out of the comparison here.
    """
    if not output.exists():
        return None
    try:
        previous = json.loads(output.read_text())
        previous["environment"]["sessions"]
        {(int(record["shots"]), int(record["seed"])) for record in previous["records"]}
        for record in previous["records"]:
            record.setdefault("alpha", None)  # Records from before --interpolate.
        saved = previous["config"]
        if "heads" in config:  # An option of this script only, not of embedding_quality.py.
            saved.setdefault("heads", None)
            for record in previous["records"]:  # Records from before --heads.
                record.setdefault("head", own_head(record["method"]))
        saved.setdefault("gradient_accumulation_steps", 1)  # Files from before the option.
        saved.setdefault("dataset_revision", None)
        saved.setdefault("augmentation", "none")
        saved.setdefault("unfreeze_last_blocks", None)
        saved.setdefault("trainable_parameters", "all")
        saved.setdefault("max_steps", None)
        saved.setdefault("projection_dim", None)
        saved.setdefault("contrastive_views", 1)
        saved.setdefault("temperature", 0.07)
        saved.setdefault("contrastive_weight", DEFAULT_CONTRASTIVE_WEIGHT)
        saved.setdefault("sampler", {})
        saved.setdefault("device", "cpu")
        saved.setdefault("interpolate", [])
        saved.setdefault("max_image_side", None)
        saved.setdefault("head_learning_rate", {})
        saved.setdefault("layer_lr_decay", {})
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        raise SystemExit(f"{output} is not a results file written by this script.") from None
    if set(saved) - set(measured) != set(config) or any(
        saved[key] != value for key, value in config.items()
    ):
        raise SystemExit(f"{output} holds results for a different configuration.")
    return previous


def positive_int(value: str) -> int:
    if not value.isdigit() or int(value) < 1:
        raise argparse.ArgumentTypeError(f"expected a positive int, got {value!r}")
    return int(value)


def positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        number = math.nan
    if not number > 0 or math.isinf(number):
        raise argparse.ArgumentTypeError(f"expected a positive number, got {value!r}")
    return number


def own_head(method: str) -> str:
    """The head ``method`` is scored with by default: ``linear`` for ``end_to_end`` and
    ``ce_supcon``, whose jointly trained classifier becomes a ``LinearHead`` (ADR 0011), and the
    default ``TrainingArguments.head`` for every other method."""
    if METHOD_MODES.get(method, method) == "end_to_end":
        return "linear"
    return str(TrainingArguments.__dataclass_fields__["head"].default)


def interpolation_weight(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        number = math.nan
    if not 0 <= number < 1:
        raise argparse.ArgumentTypeError(f"expected a number in [0, 1), got {value!r}")
    return number + 0.0  # -0 becomes 0, so that it has one summary row.


def backbone_state(backbone: nn.Module, device: torch.device | str) -> dict[str, torch.Tensor]:
    """A copy of every parameter and buffer of ``backbone`` on ``device``, by name."""
    return {
        name: tensor.detach().to(device, copy=True)
        for name, tensor in chain(backbone.named_parameters(), backbone.named_buffers())
    }


def interpolate_backbone(
    backbone: nn.Module,
    initial: dict[str, torch.Tensor],
    trained: dict[str, torch.Tensor],
    alpha: float,
) -> None:
    """Set ``backbone`` to ``(1 - alpha) * initial + alpha * trained`` in place (WiSE-FT).

    Floating-point parameters and buffers are interpolated on the backbone's device; every
    other tensor (e.g. a batch-norm batch count) is copied from ``trained``. ``alpha`` 0 gives
    ``initial`` exactly.

    Raises:
        ConfigurationError: If ``initial`` or ``trained`` does not name exactly the backbone's
            parameters and buffers.
    """
    tensors = dict(chain(backbone.named_parameters(), backbone.named_buffers()))
    if not set(initial) == set(trained) == set(tensors):
        raise ConfigurationError(
            "The initial and trained states must hold exactly the backbone's parameters and "
            "buffers."
        )
    with torch.no_grad():
        for name, tensor in tensors.items():
            if tensor.is_floating_point():
                start = initial[name].to(tensor.device)  # One tensor at a time on the device.
                tensor.copy_(torch.lerp(start, trained[name], alpha))
            else:
                tensor.copy_(trained[name])


def trained_weights(trainer: Trainer, alphas: Sequence[float]) -> Iterator[float | None]:
    """Train, yield ``None``, then set the model to each weight in ``alphas`` and yield it.

    Score the model at every yield; the next one changes it. ``frozen`` yields ``None`` only.
    At every weight the backbone is ``(1 - alpha) * initial + alpha * trained`` and a new head
    is fitted on the training set, so a weight's result does not depend on the others:
    ``end_to_end`` gets a ``LinearHead`` seeded and configured as its linear probe (ADR 0012),
    so that at ``alpha`` 0 it is that probe (``ce_supcon`` too, which trains in that mode); every
    other mode gets a copy of its head as it was before training, refitted. The initial state is
    kept on the CPU and the trained one on the model's device.
    """
    model = trainer.model
    interpolate = bool(alphas) and trainer.args.training_mode != "frozen"
    initial = backbone_state(model.backbone, "cpu") if interpolate else {}
    untrained_head = copy.deepcopy(model.head)
    trainer.train()
    yield None
    if not interpolate:
        return
    trained = backbone_state(model.backbone, model.device)  # The one extra state on the device.
    for alpha in alphas:
        interpolate_backbone(model.backbone, initial, trained, alpha)
        if trainer.args.training_mode == "end_to_end":
            model.head = LinearHead(seed=trainer.args.seed)
        else:
            model.head = copy.deepcopy(untrained_head)
        model.fit_head(
            trainer.train_dataset.images,
            trainer.train_dataset.labels,
            batch_size=trainer.args.eval_batch_size,
        )
        yield alpha


def fitted_heads(trainer: Trainer, heads: Sequence[str]) -> Iterator[str]:
    """Give the trainer's model each head in ``heads``, fitted, and yield its name.

    Score the model at every yield; the next one replaces its head. Call it at every yield of
    ``trained_weights``. The head the model has then, the method's own (``own_head``), is used
    as it is. Every other head is new and fitted on the training embeddings of the backbone as
    it is, encoded once: ``linear`` a ``LinearHead`` seeded and configured as the linear probe
    of ``end_to_end`` (ADR 0012), ``logistic_regression`` one with the default settings, as the
    other methods' own head. The backbone is not changed.
    """
    model = trainer.model
    own, fitted = model.head_name, model.head
    # The records name the head after own_head, so it must be the one the method was given.
    assert own == own_head(trainer.args.training_mode), (own, trainer.args.training_mode)
    embeddings = None
    for name in heads:
        if name == own:
            model.head = fitted
        else:
            if embeddings is None:
                embeddings = model.encode(
                    trainer.train_dataset.images, batch_size=trainer.args.eval_batch_size
                )
                assert isinstance(embeddings, np.ndarray)
            model.head = (
                LinearHead(seed=trainer.args.seed) if name == "linear" else create_head(name)
            )
            model.fit_head(labels=trainer.train_dataset.labels, embeddings=embeddings)
        yield name


def row_name(method: str, alpha: float | None, head: str | None = None) -> str:
    """The summary row of a method's records at an interpolation weight (``None``: as trained),
    scored with ``head`` (``None``: the head is not named).

    ``repr`` gives distinct weights distinct names.
    """
    name = method if head is None else f"{method} {head}"
    return name if alpha is None else f"{name} α={alpha!r}"


def row_names(
    methods: Sequence[str], alphas: Sequence[float], heads: Sequence[str | None] = (None,)
) -> list[str]:
    """The summary rows of ``methods``: each, followed by its interpolations if it is trained,
    once per head in ``heads``."""
    return [
        row_name(method, alpha, head)
        for method in methods
        for alpha in (None, *(alphas if method in TRAINED_METHODS else ()))
        for head in heads
    ]


def parse_learning_rates(values: list[str], default: float) -> dict[str, float]:
    rates = dict.fromkeys(TRAINED_METHODS, default)
    for value in values:
        method, _, rate = value.partition("=")
        if method not in rates or not rate:
            raise SystemExit(
                f"--learning-rate expects METHOD=RATE with METHOD in {TRAINED_METHODS}"
            )
        rates[method] = float(rate)
    return rates


def training_parser(description: str) -> argparse.ArgumentParser:
    """The options of every benchmark script: data, methods, training budget, and output."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--data-root",
        type=lambda value: Path(value).expanduser().resolve(),
        default=DEFAULT_DATA_ROOT,
        help="downloads and derived data (default: the repository's git-ignored .cache/)",
    )
    parser.add_argument("--dataset", choices=sorted(LOADERS), required=True)
    parser.add_argument("--backbone", default="hf/facebook/dinov2-small")
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help=f"model input side; defaults per dataset: {DEFAULT_IMAGE_SIZE}",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["frozen", *TRAINED_METHODS],
        default=list(DEFAULT_METHODS),
        help="ce_supcon is end_to_end with a supervised contrastive term (ADR 0022); default: "
        f"{' '.join(DEFAULT_METHODS)}",
    )
    parser.add_argument("--shots", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=positive_int,
        default=1,
        help="batches per optimizer step: batch 16 in 2 steps takes the same number of steps as "
        "batch 32 at half the memory, but a supcon or pair batch then holds 16 samples, not 32",
    )
    parser.add_argument(
        "--learning-rate",
        nargs="*",
        action="extend",
        default=[],
        help="per-method rates, e.g. supcon=1e-5",
    )
    parser.add_argument("--default-learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--head-learning-rate",
        nargs="*",
        action="extend",
        default=[],
        help="per-method rates of the end_to_end and ce_supcon classifier and of the projection "
        "head (supcon, ce_supcon), e.g. end_to_end=1e-3 (ADR 0020); default: the method's "
        "learning rate",
    )
    parser.add_argument(
        "--layer-lr-decay",
        nargs="*",
        action="extend",
        default=[],
        help="per-method layer-wise learning-rate decay of the backbone in (0, 1], e.g. "
        "supcon=0.8 (ADR 0020); default: none",
    )
    parser.add_argument(
        "--sampler",
        nargs="*",
        action="extend",  # Repeated options add up rather than replace each other.
        default=[],
        metavar="METHOD=NAME",
        help="per-method batch sampler, balanced or random, for "
        f"{', '.join(SAMPLER_DEFAULTS)}; defaults: "
        f"{', '.join(f'{m}={n}' for m, n in SAMPLER_DEFAULTS.items())}. "
        "end_to_end=balanced gives the batches of ce_supcon (ADR 0022)",
    )
    parser.add_argument(
        "--pairs-per-example",
        type=int,
        default=1,
        help="pair mode: positive and negative partners per sample per epoch",
    )
    parser.add_argument(
        "--train-mode-batch-norm",
        action="store_true",
        help="update batch-norm statistics during training (freeze_batch_norm=False, ADR 0013)",
    )
    parser.add_argument(
        "--augmentation",
        choices=available_augmentations(),
        default="none",
        help="trained methods: task augmentation of the training images (ADR 0014)",
    )
    parser.add_argument(
        "--unfreeze-last-blocks",
        type=positive_int,
        default=None,
        help="trained methods: train only the backbone's last N blocks (ADR 0015)",
    )
    parser.add_argument(
        "--trainable-parameters",
        choices=TRAINABLE_PARAMETERS,
        default="all",
        help="trained methods: train only the backbone's normalisation-layer ('layernorm') or "
        "bias ('bias') parameters, within the --unfreeze-last-blocks if given (ADR 0021)",
    )
    parser.add_argument(
        "--max-steps",
        type=positive_int,
        default=None,
        help="trained methods: train for this many optimizer steps instead of --epochs (ADR 0017)",
    )
    parser.add_argument(
        "--projection-dim",
        type=positive_int,
        default=None,
        help="supcon and ce_supcon: compute the contrastive loss on a projection head of this "
        "output size (ADR 0018)",
    )
    parser.add_argument(
        "--contrastive-views",
        type=positive_int,
        default=1,
        help="supcon and ce_supcon: augmented views of every drawn image; above 1 needs "
        "--augmentation (ADR 0018)",
    )
    parser.add_argument(
        "--temperature",
        type=positive_float,
        default=0.07,
        help="supcon and ce_supcon: temperature of the supervised contrastive loss",
    )
    parser.add_argument(
        "--contrastive-weight",
        type=positive_float,
        default=DEFAULT_CONTRASTIVE_WEIGHT,
        help="ce_supcon: weight w of the loss CE + w * SupCon (ADR 0022)",
    )
    parser.add_argument(
        "--interpolate",
        nargs="+",
        type=interpolation_weight,
        default=[],
        metavar="ALPHA",
        help="trained methods: also score each trained backbone at the weights "
        "(1 - ALPHA) * pretrained + ALPHA * trained (WiSE-FT), with the head refitted, for "
        "every ALPHA in [0, 1); the training is not repeated",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="device for encoding and training, e.g. cpu or cuda; no fallback to another device",
    )
    parser.add_argument(
        "--disable-cudnn",
        action="store_true",
        help="run CUDA convolutions without cuDNN, for GPUs that the installed cuDNN does not "
        "support",
    )
    parser.add_argument(
        "--max-image-side",
        type=positive_int,
        default=None,
        help="decode every loaded image once and keep it in memory as RGB with its shorter side "
        "shrunk to at most this (never enlarged); default: decode from the source at every use",
    )
    parser.add_argument("--evaluate-on", choices=["validation", "test"], required=True)
    parser.add_argument("--pool-per-class", type=int, default=100)
    parser.add_argument("--eval-per-class", type=int, default=100)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True, help="JSON file for the records")
    return parser


def check_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Every check of the ``training_parser`` options that needs no data; exits on an error."""
    for option in ("--learning-rate", "--head-learning-rate", "--layer-lr-decay"):
        methods = [value.partition("=")[0] for value in getattr(args, option[2:].replace("-", "_"))]
        if len(set(methods)) != len(methods):
            parser.error(f"{option} names a method twice")
    try:  # Before loading any data, so that a typo fails at once.
        parse_learning_rates(args.learning_rate, args.default_learning_rate)
    except ValueError:
        parser.error("--learning-rate expects METHOD=RATE with a number as RATE")
    samplers: dict[str, str] = {}
    for value in args.sampler:
        method, _, name = value.partition("=")
        if method not in SAMPLER_DEFAULTS or name not in ("balanced", "random"):
            parser.error(
                f"--sampler expects METHOD=NAME with METHOD in {list(SAMPLER_DEFAULTS)} and NAME "
                "balanced or random"
            )
        if method in samplers:
            parser.error(f"--sampler names {method} twice")
        if method not in args.methods:
            parser.error(f"--sampler names {method}, which is not selected")
        samplers[method] = name
    # Only the choices that differ from the default are recorded, so that naming a default
    # resumes like leaving it out.
    args.sampler = {
        method: samplers[method]
        for method in sorted(samplers)
        if samplers[method] != SAMPLER_DEFAULTS[method]
    }
    if args.augmentation != "none" and not set(args.methods) & set(TRAINED_METHODS):
        parser.error("--augmentation applies to trained methods only, and none is selected")
    if args.unfreeze_last_blocks is not None and not set(args.methods) & set(TRAINED_METHODS):
        parser.error("--unfreeze-last-blocks applies to trained methods only, and none is selected")
    if args.trainable_parameters != "all" and not set(args.methods) & set(TRAINED_METHODS):
        parser.error("--trainable-parameters applies to trained methods only, and none is selected")
    if args.max_steps is not None and not set(args.methods) & set(TRAINED_METHODS):
        parser.error("--max-steps applies to trained methods only, and none is selected")
    for option, values, lowest in (("--shots", args.shots, 1), ("--seeds", args.seeds, 0)):
        if len(set(values)) != len(values):
            parser.error(f"{option} names a value twice")
        if min(values) < lowest:
            parser.error(f"{option} must be at least {lowest}, got {min(values)}")
    if max(args.seeds) > MAX_SEED:
        parser.error(f"--seeds must be at most 2**32 - 1, got {max(args.seeds)}")
    if args.interpolate and not set(args.methods) & set(TRAINED_METHODS):
        parser.error("--interpolate applies to trained methods only, and none is selected")
    if len(set(args.interpolate)) != len(args.interpolate):
        parser.error("--interpolate names a weight twice")
    args.interpolate = sorted(args.interpolate)  # One order, so that resuming compares alike.
    for option, value, default in (
        ("--projection-dim", args.projection_dim, None),
        ("--contrastive-views", args.contrastive_views, 1),
        ("--temperature", args.temperature, 0.07),
    ):
        if value != default and not set(args.methods) & set(CONTRASTIVE_METHODS):
            parser.error(
                f"{option} applies to the supcon and ce_supcon methods only, and neither is "
                "selected"
            )
    if args.contrastive_weight != DEFAULT_CONTRASTIVE_WEIGHT and "ce_supcon" not in args.methods:
        parser.error(
            "--contrastive-weight applies to the ce_supcon method only, and it is not selected"
        )
    if args.device == "auto":
        parser.error("--device must name a device: 'auto' could resolve differently on resume")
    try:
        device = resolve_device(args.device)
    except ConfigurationError as error:
        parser.error(f"--device: {error}")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    args.device = str(device)  # One spelling per device, so that resuming compares like with like.
    if args.contrastive_views > 1 and args.augmentation == "none":
        parser.error("--contrastive-views above 1 needs --augmentation: the views would be equal")
    for option in ("--head-learning-rate", "--layer-lr-decay"):
        name = option[2:].replace("-", "_")
        values = {}
        for value in getattr(args, name):
            method, _, number = value.partition("=")
            if method not in set(args.methods) & set(TRAINED_METHODS):
                parser.error(f"{option} expects METHOD=VALUE with METHOD a selected trained method")
            try:
                values[method] = float(number)
            except ValueError:
                parser.error(f"{option} expects METHOD=VALUE with a number as VALUE")
        setattr(args, name, values)  # A map, so that resuming compares values, not spellings.
    try:  # Every combination TrainingArguments rejects, such as a head rate without a head.
        training_arguments(
            args, parse_learning_rates(args.learning_rate, args.default_learning_rate)
        )
    except ConfigurationError as error:
        parser.error(str(error))


def parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    """The command line, with every check that needs no data; exits on an error."""
    parser = training_parser(__doc__.splitlines()[0])
    parser.add_argument(
        "--heads",
        nargs="+",
        choices=HEADS,
        default=None,
        help="score every method with each of these heads, fitted on the training embeddings of "
        "its final backbone (ADR 0019); default: each method's own head, linear for end_to_end "
        "and ce_supcon and logistic_regression for the others",
    )
    args = parser.parse_args(argv)
    check_arguments(parser, args)
    if args.heads is not None:
        if len(set(args.heads)) != len(args.heads):
            parser.error("--heads names a head twice")
        args.heads = sorted(args.heads)  # One order, so that resuming compares alike.
    return args


def resume_configuration(args: argparse.Namespace) -> dict:
    """The options a results file is resumed under: every one that changes the records."""
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in {"data_root", "output", "threads", "disable_cudnn"}
    }
    if args.max_steps is not None:
        config["epochs"] = None  # Ignored under a step budget (ADR 0017).
    config["freeze_batch_norm"] = not args.train_mode_batch_norm
    config["dataset_revision"] = DATASET_REVISIONS.get(args.dataset)
    return config


def prepare_environment(args: argparse.Namespace) -> None:
    """Point caches and temporary files at ``--data-root`` and apply the process options."""
    # Before any download library is imported: they read these when imported.
    os.environ.setdefault("HF_HOME", str(args.data_root / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(args.data_root / "torch"))
    os.environ.setdefault("TMPDIR", str(args.data_root / "tmp"))
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = None  # Re-read TMPDIR on the next use.
    torch.set_num_threads(args.threads)
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
    # Where NNPACK cannot run, every convolution that considers it (on the CPU, and on CUDA
    # without cuDNN) logs "Could not initialize NNPACK"; switching it off stops that and changes
    # no result, as it is never used there. The availability check logs the message once.
    if not torch.backends.nnpack.is_available():
        torch.backends.nnpack.set_flags(False)
    if args.image_size is None:
        args.image_size = DEFAULT_IMAGE_SIZE[args.dataset]


def training_arguments(
    args: argparse.Namespace, rates: dict[str, float]
) -> dict[str, TrainingArguments]:
    """``TrainingArguments`` of every selected trained method; ``frozen`` needs none."""
    budget = {
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": 0,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "freeze_batch_norm": not args.train_mode_batch_norm,
        "augmentation": args.augmentation,
        "unfreeze_last_blocks": args.unfreeze_last_blocks,
        "trainable_parameters": args.trainable_parameters,
        "max_steps": args.max_steps,
    }
    arguments = {
        method: TrainingArguments(
            training_mode=METHOD_MODES.get(method, method),
            learning_rate=rates[method],
            layer_lr_decay=args.layer_lr_decay.get(method),
            **budget,
        )
        for method in args.methods
        if method in TRAINED_METHODS
    }
    contrastive = {
        "projection_dim": args.projection_dim,
        "contrastive_views": args.contrastive_views,
        "temperature": args.temperature,
    }
    if "supcon" in arguments:
        arguments["supcon"] = replace(arguments["supcon"], **contrastive)
    if "ce_supcon" in arguments:
        arguments["ce_supcon"] = replace(
            arguments["ce_supcon"], contrastive_weight=args.contrastive_weight, **contrastive
        )
    for method, name in args.sampler.items():
        arguments[method] = replace(arguments[method], sampler=name)
    if "pair" in arguments:
        arguments["pair"] = TrainingArguments(
            training_mode="pair",
            learning_rate=rates["pair"],
            positives_per_example=args.pairs_per_example,
            negatives_per_example=args.pairs_per_example,
            layer_lr_decay=args.layer_lr_decay.get("pair"),
            **budget,
        )
    return {
        method: replace(options, head_learning_rate=args.head_learning_rate.get(method))
        for method, options in arguments.items()
    }


def drop_repeats(items: Split) -> tuple[Split, int]:
    """Keep the first of every set of images with identical pixels; return how many were dropped."""
    seen: set[str] = set()
    kept = []
    for image, label in items:
        key = digest(image)
        if key not in seen:
            seen.add(key)
            kept.append((image, label))
    return kept, len(items) - len(kept)


def check_image_side(args: argparse.Namespace) -> None:
    """Exit if ``--max-image-side`` is below the shorter side the backbone resizes images to.

    Below it, the backbone would enlarge the shrunk images again. Builds the backbone once, on
    the CPU, so that this fails before any data is loaded.
    """
    if args.max_image_side is None:
        return
    backbone = EmbedFitModel.from_pretrained(
        args.backbone, image_size=args.image_size, device="cpu"
    ).backbone
    edge = getattr(backbone, "resize_size", None) or getattr(backbone, "image_size", None)
    if isinstance(edge, tuple):
        edge = min(edge)
    if isinstance(edge, int) and args.max_image_side < edge:
        raise SystemExit(
            f"--max-image-side {args.max_image_side} is below {edge}, the shorter side "
            f"{args.backbone} resizes images to at --image-size {args.image_size}; use at least "
            f"{edge}, and about twice --image-size to leave room for augmentation crops."
        )


def load_splits(
    args: argparse.Namespace, *, drop_repeated: bool = False
) -> tuple[dict[str, ImageDataset], list[str], int, int]:
    """The pool and the evaluated split, the class names, and the duplicates dropped.

    The last two are the evaluation images dropped because their pixels match a pool image, and,
    with ``drop_repeated``, those dropped because their pixels match an earlier evaluation image
    (0 otherwise). Both are judged on the original pixels, before ``--max-image-side`` shrinks
    them, so that they do not depend on it.
    """
    loaded, labels = LOADERS[args.dataset](
        args.data_root,
        pool=args.pool_per_class,
        evaluation=args.eval_per_class,
        evaluate_on=args.evaluate_on,
        seed=args.split_seed,
    )
    duplicates = drop_pool_duplicates(loaded, args.evaluate_on)  # On the original pixels.
    repeats = 0
    if drop_repeated:
        loaded[args.evaluate_on], repeats = drop_repeats(loaded[args.evaluate_on])
    if args.max_image_side is not None:  # Decoded once here, not at every training draw.
        loaded = {
            name: [(downscale(image, args.max_image_side), label) for image, label in items]
            for name, items in loaded.items()
        }
    splits = {
        name: ImageDataset.from_lists([image for image, _ in items], [label for _, label in items])
        for name, items in loaded.items()
    }
    return splits, labels, duplicates, repeats


def _git(*arguments: str) -> str | None:
    """The output of a git command in the repository, or ``None`` if it fails (outside git)."""
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=REPOSITORY,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def git_commit() -> str | None:
    """The commit checked out in the repository, or ``None`` outside a git checkout."""
    commit = (_git("rev-parse", "HEAD") or "").strip()
    return commit or None


def git_dirty() -> bool | None:
    """Whether the checkout has uncommitted changes or untracked files, or ``None`` outside a
    git checkout. Only the answer is recorded, never the files."""
    status = _git("status", "--porcelain")
    return None if status is None else bool(status.strip())


def package_version(name: str) -> str | None:
    """The installed version of the distribution ``name``, or ``None`` if it is not installed."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def new_session(args: argparse.Namespace) -> dict:
    """The environment of this session; ``elapsed_seconds`` grows as groups finish."""
    return {
        "embedfit": embedfit.__version__,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "torch": torch.__version__,
        **{name: package_version(name) for name in RECORDED_PACKAGES},
        "device": args.device,
        "cudnn": torch.backends.cudnn.enabled,
        "threads": args.threads,
        "processor": platform.processor() or platform.machine(),
        "elapsed_seconds": 0.0,
    }


def write_results(output: Path, payload: dict) -> None:
    """Write ``payload`` as JSON so that a crash mid-write never corrupts ``output``."""
    staging = output.with_name(output.name + ".partial")
    with staging.open("w") as handle:
        handle.write(json.dumps(payload, indent=1))
        handle.flush()
        os.fsync(handle.fileno())
    staging.replace(output)


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)
    rates = parse_learning_rates(args.learning_rate, args.default_learning_rate)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    prepare_environment(args)
    arguments = training_arguments(args, rates)

    # Results are written after every (shots, seed) group, and a rerun with the same output
    # file and configuration skips the groups already there. Each group's few-shot sample
    # depends only on its shots and seed, so a resumed run gives the same records. The thread
    # count is not part of the configuration, so a run can be resumed with a different one;
    # every session's environment is recorded.
    config = resume_configuration(args)
    config["learning_rates"] = {method: rates[method] for method in arguments}
    previous = read_previous(args.output, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = previous["records"] if previous else []
    sessions: list[dict] = previous["environment"]["sessions"] if previous else []
    done = {(r["shots"], r["seed"]) for r in records}
    elapsed = sum(entry["elapsed_seconds"] for entry in sessions)

    check_image_side(args)
    splits, labels, duplicates, _ = load_splits(args)
    config["evaluation_duplicates_dropped"] = duplicates
    config["evaluation_size"] = len(splits[args.evaluate_on])
    if previous and previous["config"] != config:
        raise SystemExit(f"{args.output} holds results for different evaluation data.")

    def model_factory() -> EmbedFitModel:
        return EmbedFitModel.from_pretrained(
            args.backbone, labels=labels, image_size=args.image_size, device=args.device
        )

    # Fail before any training: the pool must cover the largest shot count and every method
    # must accept the smallest, as one evaluator over all groups would have checked.
    sample_few_shot(splits["pool"], shots=max(args.shots), seed=args.seeds[0])
    smallest = sample_few_shot(splits["pool"], shots=min(args.shots), seed=args.seeds[0])
    for method in args.methods:
        options = arguments.get(method, TrainingArguments(training_mode="frozen"))
        Trainer(model_factory(), replace(options, seed=args.seeds[0]), smallest)

    session = new_session(args)
    sessions.append(session)
    for shots in args.shots:
        for seed in args.seeds:
            if (shots, seed) in done:
                continue
            started = time.perf_counter()
            sample = sample_few_shot(splits["pool"], shots=shots, seed=seed)
            for method in args.methods:
                set_seed(seed)  # For any randomly initialised part of the backbone.
                model = model_factory()
                options = arguments.get(method, TrainingArguments(training_mode="frozen"))
                trainer = Trainer(model, replace(options, seed=seed), sample)
                for alpha in trained_weights(trainer, args.interpolate):
                    for head in fitted_heads(trainer, args.heads or [own_head(method)]):
                        records += [
                            {
                                "method": method,
                                "alpha": alpha,
                                "head": head,
                                "backbone": model.backbone.identifier,
                                "shots": shots,
                                "seed": seed,
                                "metric": metric,
                                "value": float(value),
                            }
                            for metric, value in trainer.evaluate(splits[args.evaluate_on]).items()
                        ]
                logger.info("Benchmarked %s at %d shots, seed %d", method, shots, seed)
            session["elapsed_seconds"] = round(
                session["elapsed_seconds"] + time.perf_counter() - started, 1
            )
            elapsed = sum(entry["elapsed_seconds"] for entry in sessions)
            environment = {**session, "elapsed_seconds": elapsed, "sessions": sessions}
            payload = {"config": config, "environment": environment, "records": records}
            write_results(args.output, payload)

    named = args.heads is not None  # Rows name the head only when it is chosen.
    summary = FewShotEvaluator.aggregate(
        [
            {**r, "method": row_name(r["method"], r["alpha"], r["head"] if named else None)}
            for r in records
        ]
    )
    print(
        f"\n{args.evaluate_on} accuracy (mean ± std over {len(args.seeds)} seeds), "
        f"{elapsed / 60:.1f} min"
    )
    names = row_names(args.methods, args.interpolate, args.heads or [None])
    width = max(20, *(len(name) + 2 for name in names))
    print(f"{'method':{width}s}" + "".join(f"{f'{n} shots':>16s}" for n in args.shots))
    for name in names:
        cells = [
            f"{row['mean']:.3f} ± {row['std']:.3f}"
            for row in summary
            if row["method"] == name and row["metric"] == "accuracy"
        ]
        print(f"{name:{width}s}" + "".join(f"{cell:>16s}" for cell in cells))


if __name__ == "__main__":
    main()
