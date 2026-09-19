"""Two-phase training: representation adaptation, then head fitting (ADR 0002)."""

import logging
from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np

from embedfit.data.dataset import ImageDataset
from embedfit.exceptions import ConfigurationError
from embedfit.heads.registry import create_head
from embedfit.model import EmbedFitModel
from embedfit.training_args import TrainingArguments
from embedfit.utils.device import resolve_device
from embedfit.utils.seed import set_seed

logger = logging.getLogger(__name__)

_IMPLEMENTED_MODES = ("frozen",)


@dataclass(frozen=True)
class TrainOutput:
    """Summary of a completed ``Trainer.train`` call. Contains aggregate values only.

    Attributes:
        training_mode: The mode that ran.
        num_train_samples: Number of training samples used.
        representation_steps: Optimizer steps taken on the backbone (0 in ``frozen`` mode).
    """

    training_mode: str
    num_train_samples: int
    representation_steps: int


@dataclass(frozen=True)
class PredictionOutput:
    """Predictions for a dataset.

    Attributes:
        predictions: Predicted labels, shaped ``[N]``.
        probabilities: Class probabilities, shaped ``[N, num_classes]``.
        label_order: The label of each probability column.
        labels: The dataset's true labels, shaped ``[N]``, in the same order.
    """

    predictions: np.ndarray
    probabilities: np.ndarray
    label_order: tuple[Hashable, ...]
    labels: list[Hashable]


class Trainer:
    """Runs representation training and head fitting for an ``EmbedFitModel``.

    Args:
        model: The model to train. It must have been created with ``labels``.
        args: Training configuration. ``args.head`` names the head to fit; when it differs from
            the name the model was created with, the model's head is replaced by a new one. A
            model created with a head *instance* keeps it, and ``args.head`` must be left at
            its default.
        train_dataset: Training samples. Evaluation and test data must never be passed here.
        eval_dataset: Optional held-out samples used by ``predict`` when no dataset is given.

    Raises:
        ConfigurationError: If the model has no labels, the training set is empty or contains
            labels unknown to the model, ``args.training_mode`` is not available,
            ``freeze_backbone_before_head=False`` is combined with ``frozen`` mode, or
            ``args.head`` conflicts with the model's head instance.
    """

    def __init__(
        self,
        model: EmbedFitModel,
        args: TrainingArguments,
        train_dataset: ImageDataset,
        eval_dataset: ImageDataset | None = None,
    ) -> None:
        if args.training_mode not in _IMPLEMENTED_MODES:
            raise ConfigurationError(
                f"training_mode '{args.training_mode}' is not available in this version; "
                f"available modes: {list(_IMPLEMENTED_MODES)}."
            )
        if args.training_mode == "frozen" and not args.freeze_backbone_before_head:
            raise ConfigurationError(
                "freeze_backbone_before_head=False contradicts training_mode='frozen', "
                "which never updates the backbone."
            )
        default_head = TrainingArguments.__dataclass_fields__["head"].default
        if model.head_name is None and args.head != default_head:
            raise ConfigurationError(
                f"The model was created with a head instance, which args.head='{args.head}' "
                "would replace. Leave args.head at its default to keep the instance, or create "
                "the model with a head name."
            )
        if model.label_mapping is None:
            raise ConfigurationError(
                "Trainer needs the model's class labels; pass labels=[...] when creating the model."
            )
        for name, dataset in (("train_dataset", train_dataset), ("eval_dataset", eval_dataset)):
            if dataset is not None and not isinstance(dataset, ImageDataset):
                raise ConfigurationError(
                    f"{name} must be an ImageDataset, got {type(dataset).__name__}."
                )
        if len(train_dataset) == 0:
            raise ConfigurationError("train_dataset is empty.")
        model.label_mapping.encode(train_dataset.labels)  # Reject unknown labels up front.

        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self._device = resolve_device(args.device) if args.device is not None else None

    def train(self) -> TrainOutput:
        """Run both phases and leave the model ready for prediction.

        Returns:
            A ``TrainOutput`` summary.
        """
        set_seed(self.args.seed, deterministic=self.args.deterministic)
        if self._device is not None:
            self.model.to(self._device)
        logger.info(
            "Training mode '%s' on %d samples", self.args.training_mode, len(self.train_dataset)
        )

        logger.info("Phase 1/2 (representation): skipped, backbone stays frozen")
        self.model.freeze_backbone()

        if self.model.head_name is not None and self.model.head_name != self.args.head:
            self.model.head = create_head(self.args.head)
            self.model.head_name = self.args.head
        logger.info("Phase 2/2 (head): fitting %s", type(self.model.head).__name__)
        self.model.fit_head(
            self.train_dataset.images,
            self.train_dataset.labels,
            batch_size=self.args.eval_batch_size,
        )
        return TrainOutput(
            training_mode=self.args.training_mode,
            num_train_samples=len(self.train_dataset),
            representation_steps=0,
        )

    def predict(self, dataset: ImageDataset | None = None) -> PredictionOutput:
        """Predict labels and probabilities for a dataset.

        Args:
            dataset: Samples to predict. Defaults to ``eval_dataset``.

        Raises:
            ConfigurationError: If no dataset is available.
            HeadNotFittedError: If ``train`` has not run.
        """
        target = dataset if dataset is not None else self.eval_dataset
        if target is None:
            raise ConfigurationError("predict needs a dataset: pass one or set eval_dataset.")
        probabilities = self.model.predict_proba(
            target.images, batch_size=self.args.eval_batch_size
        )
        assert self.model.label_mapping is not None
        return PredictionOutput(
            predictions=self.model.label_mapping.decode(probabilities.argmax(axis=1)),
            probabilities=probabilities,
            label_order=self.model.label_mapping.labels,
            labels=target.labels,
        )
