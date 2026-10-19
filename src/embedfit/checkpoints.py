"""Representation checkpoints: a model bundle for the best and the last epoch (ADR 0016).

Layout::

    <checkpoint_dir>/
    ├── checkpoints.json   best and last epoch, and each epoch's losses
    ├── epoch-0003/        model bundle of the best epoch, with an unfitted head
    └── epoch-0005/        model bundle of the last epoch

Each epoch is written to a new directory and never replaced; ``checkpoints.json`` is replaced
in one step afterwards, and only then are older epochs deleted. So ``checkpoints.json`` only
ever names complete bundles. Bundles hold what ``save_model`` writes (ADR 0009): weights,
configuration, label names, and aggregate training facts, never samples or paths.
"""

import copy
import logging
import shutil
from collections.abc import Sequence
from pathlib import Path

from embedfit.exceptions import ConfigurationError
from embedfit.heads.base import ClassificationHead
from embedfit.model import EmbedFitModel
from embedfit.serialization import check_saveable, save_model, write_json_atomic
from embedfit.training_args import TrainingArguments, _recorded_arguments

logger = logging.getLogger(__name__)

INDEX_FILE = "checkpoints.json"
FORMAT_VERSION = 1


def _epoch_directory(epoch: int) -> str:
    return f"epoch-{epoch:04d}"


def checkpoint_model(model: EmbedFitModel, args: TrainingArguments) -> EmbedFitModel:
    """Check that checkpoints can be written, and return the model they are written from.

    The returned model shares ``model``'s backbone, so it always holds the current weights,
    and has an unfitted head of the type that head fitting will use: ``linear`` in
    ``end_to_end`` mode, ``args.head`` for a model created with a head name, or a copy of the
    model's head instance.

    Raises:
        ConfigurationError: If ``args.checkpoint_dir`` is not a missing or empty directory,
            the model's head instance is already fitted (on embeddings of a backbone that
            training changes), or the labels or head cannot be saved.
    """
    assert args.checkpoint_dir is not None
    directory = Path(args.checkpoint_dir)
    if directory.exists() and not directory.is_dir():
        raise ConfigurationError(
            "checkpoint_dir is not a directory; representation checkpoints need a missing or "
            "empty directory."
        )
    if directory.exists() and any(directory.iterdir()):
        raise ConfigurationError(
            "checkpoint_dir is not empty; representation checkpoints need a missing or empty "
            "directory, so that two runs are never mixed."
        )
    head: str | ClassificationHead
    if args.training_mode == "end_to_end":
        head = "linear"
    elif model.head_name is not None:
        head = args.head
    elif model.head.is_fitted:
        raise ConfigurationError(
            "checkpoint_dir needs an unfitted head instance, but the model's is a fitted head: "
            "it was fitted on another backbone's embeddings. Create the model with a new head "
            "instance or a head name."
        )
    else:
        head = copy.deepcopy(model.head)
    view = EmbedFitModel(
        model.backbone,
        labels=model.label_mapping,
        positive_label=model.positive_label,
        normalize_embeddings=model.normalize_embeddings,
        head=head,
    )
    check_saveable(view)
    return view


class RepresentationCheckpoints:
    """Writes a checkpoint after every representation-training epoch.

    Args:
        model: The model being trained.
        args: Training arguments with ``checkpoint_dir`` set.

    Raises:
        ConfigurationError: As ``checkpoint_model``.
    """

    def __init__(self, model: EmbedFitModel, args: TrainingArguments) -> None:
        self._model = checkpoint_model(model, args)
        assert args.checkpoint_dir is not None
        self._directory = Path(args.checkpoint_dir)
        self._arguments = _recorded_arguments(args)
        self._kept: set[int] = set()

    def save(
        self,
        epoch: int,
        *,
        best_epoch: int,
        steps: int,
        train_losses: Sequence[float],
        eval_losses: Sequence[float],
    ) -> None:
        """Write the current weights as ``epoch`` (1-based) and keep only the best and last."""
        save_model(
            self._model,
            self._directory / _epoch_directory(epoch),
            training_metadata={
                "training_arguments": self._arguments,
                "checkpoint": {
                    "epoch": epoch,
                    "steps": steps,
                    "train_loss": train_losses[-1],
                    "eval_loss": eval_losses[-1] if eval_losses else None,
                },
            },
        )
        write_json_atomic(
            self._directory / INDEX_FILE,
            {
                "format_version": FORMAT_VERSION,
                "best_epoch": best_epoch,
                "last_epoch": epoch,
                "epochs": [
                    {
                        "epoch": number,
                        "train_loss": loss,
                        "eval_loss": eval_losses[number - 1] if eval_losses else None,
                    }
                    for number, loss in enumerate(train_losses, start=1)
                ],
            },
        )
        kept = {best_epoch, epoch}
        for stale in self._kept - kept:
            shutil.rmtree(self._directory / _epoch_directory(stale))
        self._kept = kept
        logger.info("Saved the representation checkpoint of epoch %d", epoch)
