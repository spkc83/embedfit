"""Saving and loading ``EmbedFitModel`` bundles.

Layout::

    <path>/
    ├── embedfit_config.json      framework version, backbone, pooling, normalisation, head type
    ├── labels.json               class labels in class-index order, and the positive label
    ├── training_metadata.json    aggregate training facts (hyperparameters, losses, counts)
    ├── backbone/                 written by the backbone adapter
    └── head/                     written by the classification head

A bundle holds weights, configuration, and label names only. Training samples, identifiers,
and metadata are never written, and nothing is pickled, so loading a bundle does not execute
code from it.
"""

import hashlib
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from embedfit._version import __version__
from embedfit.backbones.registry import load_backbone
from embedfit.exceptions import ConfigurationError
from embedfit.heads.registry import load_head

if TYPE_CHECKING:
    from embedfit.model import EmbedFitModel

logger = logging.getLogger(__name__)

CONFIG_FILE = "embedfit_config.json"
LABELS_FILE = "labels.json"
TRAINING_FILE = "training_metadata.json"
FORMAT_VERSION = 1
_JSON_LABEL_TYPES = (str, int, float, bool)


def is_model_directory(path: str | Path) -> bool:
    """Whether ``path`` is a directory containing a saved model's ``embedfit_config.json``."""
    return (Path(path) / CONFIG_FILE).is_file()


def _fingerprint(backbone_directory: Path) -> str:
    """SHA-256 of the backbone's ``config.json``, which holds its preprocessing settings."""
    return hashlib.sha256((backbone_directory / "config.json").read_bytes()).hexdigest()


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


def save_model(
    model: "EmbedFitModel",
    path: str | Path,
    *,
    training_metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write a model bundle to ``path`` (created if missing; existing files are overwritten).

    Args:
        model: The model to save. Its head may be unfitted.
        path: Target directory.
        training_metadata: JSON-serialisable aggregate facts about training. It must not
            contain samples or identifiers; ``Trainer.save_model`` supplies hyperparameters,
            sample counts, and losses.

    Raises:
        ConfigurationError: If a label is not a ``str``, ``int``, ``float``, or ``bool``
            (other types do not survive JSON), the head does not support saving, or
            ``training_metadata`` is not JSON-serialisable.
    """
    labels = list(model.labels) if model.labels is not None else None
    for label in (labels or []) + [model.positive_label]:
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
    try:
        metadata_text = json.dumps(dict(training_metadata or {}), indent=2)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("training_metadata must be JSON-serialisable.") from exc

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(directory / "backbone")
    model.head.save(directory / "head")
    _write_json(
        directory / CONFIG_FILE,
        {
            "framework": "embedfit",
            "version": __version__,
            "format_version": FORMAT_VERSION,
            "backbone": {
                "provider": model.backbone.identifier.partition("/")[0],
                "identifier": model.backbone.identifier,
                "pooling": model.backbone.pooling,
                "embedding_dim": model.embedding_dim,
                "preprocessing_fingerprint": _fingerprint(directory / "backbone"),
            },
            "normalize_embeddings": model.normalize_embeddings,
            "head": {"type": model.head.registry_name, "fitted": model.head.is_fitted},
            "num_labels": len(labels) if labels is not None else 0,
        },
    )
    _write_json(directory / LABELS_FILE, {"labels": labels, "positive_label": model.positive_label})
    (directory / TRAINING_FILE).write_text(metadata_text + "\n")
    logger.info("Saved model bundle: backbone %s", model.backbone.identifier)


def load_model(path: str | Path, *, device: str | torch.device | None = None) -> "EmbedFitModel":
    """Rebuild a model from a bundle written by ``save_model``. Nothing is downloaded.

    Args:
        path: Bundle directory.
        device: As in ``EmbedFitModel.from_pretrained``.

    Returns:
        The model in eval mode, with the same labels, label order, positive label,
        normalisation policy, backbone weights and preprocessing, and head state.

    Raises:
        ConfigurationError: If a file is missing or malformed, the bundle was written by a
            newer format version, or the backbone's configuration no longer matches the
            fingerprint or embedding dimension recorded when the bundle was saved.
    """
    from embedfit.model import EmbedFitModel
    from embedfit.utils.device import resolve_device

    directory = Path(path)
    config = _read_json(directory / CONFIG_FILE)
    label_file = _read_json(directory / LABELS_FILE)
    try:
        if config["framework"] != "embedfit" or config["format_version"] > FORMAT_VERSION:
            raise ConfigurationError(
                f"'{directory}' was written by format version {config['format_version']!r} of "
                f"{config['framework']!r}; this version reads 'embedfit' up to {FORMAT_VERSION}."
            )
        recorded = config["backbone"]
        if _fingerprint(directory / "backbone") != recorded["preprocessing_fingerprint"]:
            raise ConfigurationError(
                f"The backbone configuration in '{directory}' changed after the model was "
                "saved, so its preprocessing may not match the weights."
            )
        backbone = load_backbone(directory / "backbone")
        if backbone.embedding_dim != recorded["embedding_dim"]:
            raise ConfigurationError(
                f"'{directory}' records embedding dimension {recorded['embedding_dim']}, but "
                f"the loaded backbone produces {backbone.embedding_dim}."
            )
        model = EmbedFitModel(
            backbone,
            labels=label_file["labels"],
            positive_label=label_file["positive_label"],
            normalize_embeddings=config["normalize_embeddings"],
            head=load_head(directory / "head"),
        )
        model.head_name = config["head"]["type"]
    except (KeyError, TypeError, OSError) as exc:
        raise ConfigurationError(f"'{directory}' is not a complete model bundle.") from exc
    if device is not None:
        model.to(resolve_device(device))
    model.eval()
    return model
