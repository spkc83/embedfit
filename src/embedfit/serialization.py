"""Saving and loading ``EmbedFitModel`` bundles.

Layout::

    <path>/
    ├── embedfit_config.json      framework version, backbone, pooling, normalisation, head type
    ├── labels.json               class labels in class-index order, and the positive label
    ├── training_metadata.json    aggregate training facts (hyperparameters, losses, counts)
    ├── backbone/                 written by the backbone adapter; must contain config.json
    └── head/                     written by the classification head

A bundle holds weights, configuration, and label names only. Training samples, identifiers,
and metadata are never written, and nothing is pickled, so loading a bundle does not execute
code from it.
"""

import hashlib
import json
import logging
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import torch

from embedfit._version import __version__
from embedfit.backbones.registry import load_backbone
from embedfit.exceptions import ConfigurationError
from embedfit.heads.registry import load_head

if TYPE_CHECKING:
    from embedfit.model import EmbedFitModel

ModelT = TypeVar("ModelT", bound="EmbedFitModel")

logger = logging.getLogger(__name__)

CONFIG_FILE = "embedfit_config.json"
LABELS_FILE = "labels.json"
TRAINING_FILE = "training_metadata.json"
FORMAT_VERSION = 1
_JSON_LABEL_TYPES = (str, int, float, bool)


def is_model_directory(path: str | Path) -> bool:
    """Whether ``path`` is a directory containing a saved model's ``embedfit_config.json``."""
    try:
        return (Path(path) / CONFIG_FILE).is_file()
    except OSError:  # e.g. a name too long for the file system: certainly not a bundle.
        return False


def _fingerprint(backbone_directory: Path) -> str:
    """SHA-256 of the backbone's ``config.json`` content, which holds its preprocessing.

    The JSON is re-serialised canonically first, so formatting and key order do not matter.
    """
    config_path = backbone_directory / "config.json"
    try:
        content = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(
            f"'{config_path}' is missing or is not valid JSON. A backbone adapter's "
            "save_pretrained must write its settings to config.json."
        ) from exc
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _write_json(path: Path, content: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(content, indent=2) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        content = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"'{path}' is missing or is not valid JSON.") from exc
    if not isinstance(content, dict):
        raise ConfigurationError(f"'{path}' must contain a JSON object.")
    return content


def write_json_atomic(path: Path, content: Mapping[str, Any]) -> None:
    """Write ``content`` as JSON so that ``path`` holds either the old or the new file.

    The JSON goes to a temporary file in the same directory, which is flushed to disk and then
    renamed over ``path``; on POSIX the directory is flushed too, so the rename survives a power
    loss. The file gets the permissions of an ordinary new file (``0o666`` minus the umask).
    """
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(temporary, 0o666 & ~umask)
        with os.fdopen(handle, "w") as stream:
            stream.write(json.dumps(content, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    if hasattr(os, "O_DIRECTORY"):  # Directories cannot be opened for fsync on Windows.
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def check_saveable(model: "EmbedFitModel") -> None:
    """Raise ``ConfigurationError`` if ``save_model`` cannot write ``model``'s labels or head."""
    for label in list(model.labels or []) + [model.positive_label]:
        if label is not None and type(label) not in _JSON_LABEL_TYPES:
            raise ConfigurationError(
                f"Labels of type {type(label).__name__} cannot be saved; "
                "use str, int, float, or bool labels."
            )
    if model.head.registry_name is None:
        raise ConfigurationError(
            f"{type(model.head).__name__} cannot be saved: it has no registry_name. Register "
            "the head class and set registry_name to the registered name."
        )


def save_model(
    model: "EmbedFitModel",
    path: str | Path,
    *,
    training_metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write a model bundle to ``path``.

    The bundle is assembled in a temporary sibling directory and moved into place only when
    complete, so a failed save leaves ``path`` exactly as it was. ``path`` may be missing, an
    empty directory, or an existing bundle (which is replaced as a whole).

    Args:
        model: The model to save. Its head may be unfitted.
        path: Target directory.
        training_metadata: JSON-serialisable aggregate facts about training. It must not
            contain samples or identifiers; ``Trainer.save_model`` supplies hyperparameters,
            sample counts, and losses.

    Raises:
        ConfigurationError: If ``path`` is a file or a non-empty directory that is not a
            bundle; a label is not a ``str``, ``int``, ``float``, or ``bool`` (other types do
            not survive JSON); the head or backbone does not support saving; or
            ``training_metadata`` is not JSON-serialisable.
    """
    directory = Path(path)
    if directory.exists() and not directory.is_dir():
        raise ConfigurationError(f"'{directory}' exists and is not a directory.")
    if directory.is_dir() and any(directory.iterdir()) and not is_model_directory(directory):
        raise ConfigurationError(
            f"'{directory}' is not empty and is not a saved model; refusing to replace it."
        )
    check_saveable(model)
    labels = list(model.labels) if model.labels is not None else None
    try:
        metadata_text = json.dumps(dict(training_metadata or {}), indent=2)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("training_metadata must be JSON-serialisable.") from exc

    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
    try:
        model.backbone.save_pretrained(staging / "backbone")
        model.head.save(staging / "head")
        _write_json(
            staging / CONFIG_FILE,
            {
                "framework": "embedfit",
                "version": __version__,
                "format_version": FORMAT_VERSION,
                "backbone": {
                    "provider": model.backbone.identifier.partition("/")[0],
                    "identifier": model.backbone.identifier,
                    "pooling": model.backbone.pooling,
                    "embedding_dim": model.embedding_dim,
                    "preprocessing_fingerprint": _fingerprint(staging / "backbone"),
                },
                "normalize_embeddings": model.normalize_embeddings,
                "head": {
                    "type": model.head.registry_name,
                    "fitted": model.head.is_fitted,
                    "configured_by_name": model.head_name is not None,
                },
                "num_labels": len(labels) if labels is not None else 0,
            },
        )
        _write_json(
            staging / LABELS_FILE, {"labels": labels, "positive_label": model.positive_label}
        )
        (staging / TRAINING_FILE).write_text(metadata_text + "\n")
        if directory.exists():
            shutil.rmtree(directory)
        os.replace(staging, directory)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    logger.info("Saved model bundle: backbone %s", model.backbone.identifier)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigurationError(message)


def load_model(path: str | Path, *, device: str | torch.device | None = None) -> "EmbedFitModel":
    """Rebuild an ``EmbedFitModel`` from a bundle. See ``load_model_as``."""
    from embedfit.model import EmbedFitModel

    return load_model_as(EmbedFitModel, path, device=device)


def load_model_as(
    model_class: "type[ModelT]",
    path: str | Path,
    *,
    device: str | torch.device | None = None,
) -> "ModelT":
    """Rebuild a model from a bundle written by ``save_model``. Nothing is downloaded.

    Args:
        model_class: ``EmbedFitModel`` or a subclass of it to instantiate.
        path: Bundle directory.
        device: As in ``EmbedFitModel.from_pretrained``.

    Returns:
        The model in eval mode, with the same labels, label order, positive label,
        normalisation policy, backbone weights, dtype and preprocessing, and head state. The
        backbone is not frozen, whatever its state was when saved.

    Raises:
        ConfigurationError: If a file is missing or malformed; the format version is not one
            this library reads; the backbone's configuration no longer matches the recorded
            fingerprint, identifier, pooling, or embedding dimension; the head is not of the
            recorded type; or the number of labels disagrees with the recorded count or with
            the fitted head's class count.
    """
    from embedfit.utils.device import resolve_device

    directory = Path(path)
    config = _read_json(directory / CONFIG_FILE)
    label_file = _read_json(directory / LABELS_FILE)
    try:
        version = config["format_version"]
        _check(
            config["framework"] == "embedfit"
            and isinstance(version, int)
            and not isinstance(version, bool)
            and 1 <= version <= FORMAT_VERSION,
            f"'{directory}' was written by format version {version!r} of "
            f"{config['framework']!r}; this version reads 'embedfit' 1 to {FORMAT_VERSION}.",
        )
        normalize = config["normalize_embeddings"]
        _check(isinstance(normalize, bool), "normalize_embeddings in the bundle must be a bool.")
        recorded, head_config = config["backbone"], config["head"]
        _check(
            _fingerprint(directory / "backbone") == recorded["preprocessing_fingerprint"],
            f"The backbone configuration in '{directory}' changed after the model was saved, "
            "so its preprocessing may not match the weights.",
        )
        backbone = load_backbone(directory / "backbone")
        for name, actual in (
            ("identifier", backbone.identifier),
            ("pooling", backbone.pooling),
            ("embedding_dim", backbone.embedding_dim),
        ):
            _check(
                recorded[name] == actual,
                f"'{directory}' records backbone {name} {recorded[name]!r}, but the loaded "
                f"backbone has {actual!r}.",
            )
        head = load_head(directory / "head")
        _check(
            head.registry_name == head_config["type"],
            f"'{directory}' records head type {head_config['type']!r}, but head/ holds "
            f"{head.registry_name!r}.",
        )
        labels = label_file["labels"]
        num_labels = len(labels) if labels is not None else 0
        _check(
            num_labels == config["num_labels"],
            f"'{directory}' records {config['num_labels']} labels but labels.json holds "
            f"{num_labels}.",
        )
        _check(
            head.num_classes is None or head.num_classes == num_labels,
            f"The head in '{directory}' was fitted on {head.num_classes} classes but "
            f"labels.json holds {num_labels} labels.",
        )
        model = model_class(
            backbone,
            labels=labels,
            positive_label=label_file["positive_label"],
            normalize_embeddings=normalize,
            head=head,
        )
        if head_config["configured_by_name"] is True:
            model.head_name = head_config["type"]
    except (KeyError, TypeError, OSError) as exc:
        raise ConfigurationError(f"'{directory}' is not a complete model bundle.") from exc
    if device is not None:
        model.to(resolve_device(device))
    model.eval()
    return model
