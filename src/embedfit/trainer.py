"""Two-phase training: representation adaptation, then head fitting (ADR 0002)."""

import logging
from collections.abc import Hashable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from embedfit.checkpoints import checkpoint_model
from embedfit.data.dataset import ImageDataset
from embedfit.evaluation.calibration import brier_score
from embedfit.evaluation.classification import classification_metrics, pr_auc, roc_auc
from embedfit.exceptions import ConfigurationError, UndefinedMetricError
from embedfit.heads.linear import LinearHead
from embedfit.heads.registry import create_head
from embedfit.model import EmbedFitModel
from embedfit.representation import (
    require_every_class,
    train_representation,
    validate_representation_setup,
)
from embedfit.sampling.pairs import RelationshipRule
from embedfit.training_args import TrainingArguments, _paths_overlap, _recorded_arguments
from embedfit.utils.device import resolve_device
from embedfit.utils.seed import set_seed

logger = logging.getLogger(__name__)

_IMPLEMENTED_MODES = ("frozen", "supcon", "pair", "end_to_end")


@dataclass(frozen=True)
class TrainOutput:
    """Summary of a completed ``Trainer.train`` call. Contains aggregate values only.

    Attributes:
        training_mode: The mode that ran.
        num_train_samples: Number of training samples used.
        representation_steps: Optimizer steps taken on the backbone (0 in ``frozen`` mode).
        train_losses: Mean representation loss of each epoch (empty in ``frozen`` mode).
        eval_losses: Validation representation loss of each epoch (empty without
            ``eval_dataset`` or in ``frozen`` mode).
        best_epoch: 1-based epoch whose backbone weights were kept (0 in ``frozen`` mode).
    """

    training_mode: str
    num_train_samples: int
    representation_steps: int
    train_losses: tuple[float, ...] = ()
    eval_losses: tuple[float, ...] = ()
    best_epoch: int = 0


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
        eval_dataset: Optional validation samples. ``predict`` and ``evaluate`` default to
            them, and representation training uses them to choose which epoch's weights to
            keep. They must not be test data: a checkpoint chosen on the test set leaks it.
        relationship_rules: ``pair`` mode only: rules that declare pairs similar or dissimilar
            from a field other than the class label. Rejected in every other mode.

    Raises:
        ConfigurationError: If the model has no labels, the training set is empty, contains
            labels unknown to the model, or lacks samples for one of the model's labels,
            ``args.training_mode`` is not available,
            ``freeze_backbone_before_head=False`` is combined with ``frozen`` mode,
            ``args.head`` conflicts with the model's head instance, or contrastive training
            cannot run (batch size below 4, a class with fewer than 2 samples, or a
            mixed-precision mode the device does not support).
    """

    def __init__(
        self,
        model: EmbedFitModel,
        args: TrainingArguments,
        train_dataset: ImageDataset,
        eval_dataset: ImageDataset | None = None,
        *,
        relationship_rules: Sequence[RelationshipRule] = (),
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
        if args.training_mode == "end_to_end" and (
            model.head_name is None or args.head not in (default_head, "linear")
        ):
            raise ConfigurationError(
                "training_mode='end_to_end' trains its own linear classifier, which becomes the "
                "model's head; create the model with a head name and leave args.head at its "
                "default or 'linear'."
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
        # Reject unknown labels, and labels without samples, before any work starts: the head
        # needs every class, and finding that out after representation training wastes a run.
        require_every_class(model, train_dataset)

        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self._device = resolve_device(args.device) if args.device is not None else None
        self.relationship_rules = tuple(relationship_rules)
        self._last_output: TrainOutput | None = None
        if args.training_mode != "frozen":
            validate_representation_setup(
                model,
                train_dataset,
                eval_dataset,
                args,
                self._device or model.device,
                self.relationship_rules,
            )
        elif self.relationship_rules:
            raise ConfigurationError(
                "relationship_rules only apply to training_mode='pair', not 'frozen'."
            )

    def train(self) -> TrainOutput:
        """Run both phases and leave the model ready for prediction.

        Representation training first unfreezes the whole backbone, or only its last
        ``args.unfreeze_last_blocks`` blocks, so any freezing applied beforehand is discarded;
        ``freeze_backbone_before_head`` decides the state afterwards.

        Returns:
            A ``TrainOutput`` summary.
        """
        set_seed(self.args.seed, deterministic=self.args.deterministic)
        if self._device is not None:
            self.model.to(self._device)
        logger.info(
            "Training mode '%s' on %d samples", self.args.training_mode, len(self.train_dataset)
        )

        result = None
        if self.args.training_mode == "frozen":
            logger.info("Phase 1/2 (representation): skipped, backbone stays frozen")
        else:
            logger.info(
                "Phase 1/2 (representation): '%s' for %d epochs",
                self.args.training_mode,
                self.args.num_epochs,
            )
            if self.args.checkpoint_dir is not None:
                checkpoint_model(self.model, self.args)  # A used directory changes nothing.
            self.model.unfreeze_backbone(self.args.unfreeze_last_blocks)
            result = train_representation(
                self.model,
                self.train_dataset,
                self.eval_dataset,
                self.args,
                self.relationship_rules,
                validate=False,  # Done when this Trainer was created.
            )
        if self.args.freeze_backbone_before_head:
            self.model.freeze_backbone()
        self.model.eval()

        if result is not None and result.classifier is not None:
            # The jointly trained classifier becomes the head and is then refitted to
            # convergence like every other mode's head (ADR 0011). Otherwise this baseline
            # would be scored with a half-trained classifier while the other methods get a
            # converged one, and the comparison would measure the head, not the backbone.
            self.model.head = LinearHead.from_module(result.classifier, seed=self.args.seed)
            self.model.head_name = "linear"
        elif self.model.head_name is not None and self.model.head_name != self.args.head:
            self.model.head = create_head(self.args.head)
            self.model.head_name = self.args.head
        logger.info("Phase 2/2 (head): fitting %s", type(self.model.head).__name__)
        self.model.fit_head(
            self.train_dataset.images,
            self.train_dataset.labels,
            batch_size=self.args.eval_batch_size,
        )
        self._last_output = TrainOutput(
            training_mode=self.args.training_mode,
            num_train_samples=len(self.train_dataset),
            representation_steps=result.steps if result else 0,
            train_losses=tuple(result.train_losses) if result else (),
            eval_losses=tuple(result.eval_losses) if result else (),
            best_epoch=result.best_epoch if result else 0,
        )
        return self._last_output

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

    def evaluate(self, dataset: ImageDataset | None = None) -> dict[str, float]:
        """Compute aggregate metrics on held-out data.

        ``eval_dataset`` is the default, and in every mode that trains the backbone it also
        chose which epoch's weights were kept, so metrics on it are optimistically biased. Pass
        a separate test set for an unbiased estimate.

        Args:
            dataset: Samples to evaluate. Defaults to ``eval_dataset``.

        Returns:
            The metrics of ``embedfit.evaluation.classification_metrics``. When the model has
            a ``positive_label``, also ``roc_auc``, ``pr_auc``, and ``brier_score`` computed
            from the positive class's predicted probability; one that is undefined for the
            data (for example, no positive samples) is reported as NaN and logged as a
            warning. Accuracy alone is misleading
            for rare events, so prefer ``pr_auc`` and the ranking metrics in
            ``embedfit.evaluation`` there.

        Raises:
            ConfigurationError: If no dataset is available or it is empty.
            HeadNotFittedError: If ``train`` has not run.
        """
        output = self.predict(dataset)
        if len(output.labels) == 0:
            raise ConfigurationError("Cannot evaluate an empty dataset.")
        positive = self.model.positive_label
        results = classification_metrics(
            output.labels,
            output.predictions,
            labels=output.label_order,
            positive_label=positive,
        )
        if positive is not None:
            assert self.model.label_mapping is not None
            scores = output.probabilities[:, self.model.label_mapping.index_of(positive)]
            for name, metric in (
                ("roc_auc", roc_auc),
                ("pr_auc", pr_auc),
                ("brier_score", brier_score),
            ):
                try:
                    results[name] = metric(output.labels, scores, positive_label=positive)
                except UndefinedMetricError as exc:
                    logger.warning("%s reported as NaN: %s", name, exc)
                    results[name] = float("nan")
        logger.info("Evaluated %d samples: %d metrics", len(output.labels), len(results))
        return results

    def save_model(self, path: str | Path | None = None) -> Path:
        """Save the trained model together with aggregate training metadata.

        The metadata holds the training arguments and the ``TrainOutput`` summary (mode,
        sample count, steps, losses). It contains no samples, labels, or identifiers, and
        ``output_dir`` and ``checkpoint_dir`` are left out because a local path can reveal a
        user name.

        Args:
            path: Target directory. Defaults to ``args.output_dir``.

        Returns:
            The directory written, loadable with ``EmbedFitModel.from_pretrained``.

        Raises:
            ConfigurationError: If ``train`` has not run, ``path`` overlaps
                ``args.checkpoint_dir``, or the model cannot be saved.
        """
        if self._last_output is None:
            raise ConfigurationError("Nothing to save yet: call train() first.")
        directory = Path(path if path is not None else self.args.output_dir)
        if self.args.checkpoint_dir is not None and _paths_overlap(
            directory, self.args.checkpoint_dir
        ):
            raise ConfigurationError(
                "The model cannot be saved into, around, or as checkpoint_dir: saving replaces "
                "the target directory as a whole."
            )
        self.model.save_pretrained(
            directory,
            training_metadata={
                "training_arguments": _recorded_arguments(self.args),
                "train_output": asdict(self._last_output),
            },
        )
        return directory
