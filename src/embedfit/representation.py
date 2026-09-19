"""Phase 1 of training: adapting the backbone under a contrastive objective."""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from embedfit.data.dataset import ImageDataset
from embedfit.exceptions import ConfigurationError, TrainingDivergedError
from embedfit.losses.registry import create_loss
from embedfit.model import EmbedFitModel
from embedfit.sampling.balanced import BalancedClassBatchSampler
from embedfit.training_args import TrainingArguments

logger = logging.getLogger(__name__)

_EVAL_SEED_OFFSET = 1_000_003  # Keeps validation batches independent of training batches.


@dataclass(frozen=True)
class RepresentationResult:
    """Aggregate record of a representation-training run.

    Attributes:
        steps: Optimizer steps taken.
        train_losses: Mean training loss of each epoch.
        eval_losses: Validation loss after each epoch; empty without a validation set.
        best_epoch: 1-based epoch whose weights were kept: the one with the lowest validation
            loss (the earliest on ties), or the last epoch without a validation set.
    """

    steps: int
    train_losses: list[float] = field(default_factory=list)
    eval_losses: list[float] = field(default_factory=list)
    best_epoch: int = 0


class _PixelDataset(Dataset[tuple[torch.Tensor, int]]):
    """Preprocessed pixel tensors and class indices for a ``DataLoader``."""

    def __init__(self, model: EmbedFitModel, dataset: ImageDataset, targets: np.ndarray) -> None:
        self._backbone = model.backbone
        self._images = dataset.images
        self._targets = targets.tolist()

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, int]:
        return self._backbone.preprocess(self._images[position]), self._targets[position]


def batch_shape(batch_size: int, num_classes: int) -> tuple[int, int]:
    """Split ``batch_size`` into ``(classes_per_batch, samples_per_class)``.

    As many classes as fit with at least two samples each are used, and the batch is divided
    evenly between them, so the realised batch size can be slightly below ``batch_size``.

    Raises:
        ConfigurationError: If there are fewer than 2 classes, or ``batch_size`` is below 4,
            the minimum for two classes with a positive pair each.
    """
    if num_classes < 2:
        raise ConfigurationError(
            f"Contrastive training needs at least 2 classes in the data, got {num_classes}."
        )
    if batch_size < 4:
        raise ConfigurationError(
            f"Contrastive training needs batch_size >= 4 (2 classes x 2 samples), got {batch_size}."
        )
    classes_per_batch = min(num_classes, batch_size // 2)
    return classes_per_batch, batch_size // classes_per_batch


def _loader(
    model: EmbedFitModel, dataset: ImageDataset, args: TrainingArguments, seed: int
) -> tuple[DataLoader[tuple[torch.Tensor, int]], BalancedClassBatchSampler]:
    assert model.label_mapping is not None
    targets = model.label_mapping.encode(dataset.labels)
    classes_per_batch, samples_per_class = batch_shape(args.batch_size, len(np.unique(targets)))
    sampler = BalancedClassBatchSampler(
        targets,
        classes_per_batch=classes_per_batch,
        samples_per_class=samples_per_class,
        seed=seed,
    )
    loader = DataLoader(
        _PixelDataset(model, dataset, targets),
        batch_sampler=sampler,
        num_workers=args.num_workers,
    )
    return loader, sampler


def validate_representation_setup(
    model: EmbedFitModel,
    train_dataset: ImageDataset,
    eval_dataset: ImageDataset | None,
    args: TrainingArguments,
    device: torch.device,
) -> None:
    """Fail before training starts if contrastive training cannot run with this setup.

    Raises:
        ConfigurationError: If the batch size is too small, the training (or validation) data
            is empty, has fewer than 2 classes, or has a class with fewer than 2 samples, the
            backbone has no trainable parameters, or the mixed-precision mode is not supported
            on the device.
    """
    for name, dataset in (("train_dataset", train_dataset), ("eval_dataset", eval_dataset)):
        if dataset is not None:
            if len(dataset) == 0:
                raise ConfigurationError(f"{name} is empty.")
            _loader(model, dataset, args, seed=args.seed)
    if not any(True for _ in model.backbone.parameters()):
        raise ConfigurationError("The backbone has no parameters to train.")
    if args.mixed_precision == "fp16" and device.type != "cuda":
        raise ConfigurationError(
            f"mixed_precision='fp16' requires a CUDA device, but training runs on '{device}'. "
            "Use 'bf16' or None."
        )
    if args.mixed_precision == "bf16" and device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            raise ConfigurationError("mixed_precision='bf16' is not supported by this GPU.")
    if args.mixed_precision is not None and device.type == "mps":
        raise ConfigurationError("mixed_precision is not supported on MPS; use None.")


def _autocast(args: TrainingArguments, device: torch.device) -> torch.autocast:
    dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
    return torch.autocast(device.type, dtype=dtype, enabled=args.mixed_precision is not None)


def _schedule(total_steps: int, warmup_ratio: float) -> Callable[[int], float]:
    """Linear warm-up to the base learning rate, then linear decay to zero."""
    warmup_steps = math.ceil(total_steps * warmup_ratio)

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    return factor


def train_representation(
    model: EmbedFitModel,
    train_dataset: ImageDataset,
    eval_dataset: ImageDataset | None,
    args: TrainingArguments,
) -> RepresentationResult:
    """Adapt the backbone with the supervised contrastive loss.

    Each epoch draws class-balanced batches from ``train_dataset`` only. When ``eval_dataset``
    is given, its contrastive loss is measured after every epoch (without gradients) on a
    fixed set of class-balanced batches, so it is a balanced estimate rather than the mean over
    every validation sample, and the weights of the best epoch are restored at the end;
    otherwise the last epoch is kept. ``eval_dataset`` must be validation data: selecting a
    checkpoint on test data leaks it.

    Images are preprocessed on the fly each epoch; nothing is cached. With ``num_workers > 0``
    each worker process receives a copy of the backbone (its ``preprocess`` method does the
    work), which costs start-up time that only pays off on larger datasets.

    This function only updates backbone weights. Freezing the backbone and switching the
    model to eval mode afterwards is the caller's job.

    Args:
        model: The model whose backbone is trained. It must have labels.
        train_dataset: Training samples.
        eval_dataset: Optional validation samples for checkpoint selection.
        args: Hyperparameters: ``num_epochs``, ``batch_size``, ``learning_rate``,
            ``weight_decay``, ``warmup_ratio``, ``temperature``,
            ``gradient_accumulation_steps``, ``max_grad_norm``, ``mixed_precision``,
            ``num_workers``, and ``seed``.

    Returns:
        A ``RepresentationResult``.

    Raises:
        ConfigurationError: If the setup is invalid (see ``validate_representation_setup``).
        TrainingDivergedError: If the loss becomes NaN or infinite.
    """
    device = model.device
    validate_representation_setup(model, train_dataset, eval_dataset, args, device)
    loss_fn = create_loss("supcon", temperature=args.temperature)
    loader, sampler = _loader(model, train_dataset, args, seed=args.seed)
    eval_loader = None
    if eval_dataset is not None:
        eval_loader, _ = _loader(model, eval_dataset, args, seed=args.seed + _EVAL_SEED_OFFSET)

    accumulation = args.gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(loader) / accumulation)
    total_steps = steps_per_epoch * args.num_epochs
    parameters = [p for p in model.backbone.parameters() if p.requires_grad]
    if not parameters:
        raise ConfigurationError(
            "Every backbone parameter is frozen; call unfreeze_backbone() before training."
        )
    dtype = parameters[0].dtype
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _schedule(total_steps, args.warmup_ratio)
    )
    scaler = torch.amp.GradScaler(device.type, enabled=args.mixed_precision == "fp16")

    train_losses: list[float] = []
    eval_losses: list[float] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch, steps = 0, 0
    for epoch in range(args.num_epochs):
        model.train()
        sampler.set_epoch(epoch)
        batch_losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        for index, (pixels, targets) in enumerate(loader):
            with _autocast(args, device):
                loss = loss_fn(model(pixels.to(device=device, dtype=dtype)), targets.to(device))
            if not torch.isfinite(loss):
                raise TrainingDivergedError(
                    f"Contrastive loss became non-finite at epoch {epoch + 1}; "
                    "lower the learning rate or raise the temperature."
                )
            # The last group of an epoch can hold fewer than `accumulation` batches; dividing
            # by its real size keeps every optimizer step a mean over its batches.
            group_start = index - index % accumulation
            group_size = min(accumulation, len(loader) - group_start)
            scaler.scale(loss / group_size).backward()
            batch_losses.append(loss.item())
            if index + 1 == group_start + group_size:
                if args.max_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale() >= scale_before:
                    # fp16 only: a step with overflowing gradients is skipped by the scaler
                    # and must not advance the schedule or the step count.
                    scheduler.step()
                    steps += 1
        train_losses.append(float(np.mean(batch_losses)))

        if eval_loader is not None:
            model.eval()
            with torch.no_grad(), _autocast(args, device):
                losses = [
                    loss_fn(model(pixels.to(device=device, dtype=dtype)), targets.to(device)).item()
                    for pixels, targets in eval_loader
                ]
            eval_losses.append(float(np.mean(losses)))
            if len(eval_losses) == 1 or eval_losses[-1] < min(eval_losses[:-1]):  # Ties: earliest.
                best_epoch = epoch + 1
                # Kept on the CPU so that a GPU backbone is not held in memory twice.
                best_state = {
                    name: value.detach().to("cpu", copy=True)
                    for name, value in model.backbone.state_dict().items()
                }
            logger.info(
                "Epoch %d/%d: train loss %.4f, validation loss %.4f",
                epoch + 1,
                args.num_epochs,
                train_losses[-1],
                eval_losses[-1],
            )
        else:
            best_epoch = epoch + 1
            logger.info(
                "Epoch %d/%d: train loss %.4f", epoch + 1, args.num_epochs, train_losses[-1]
            )

    if best_state is not None and best_epoch != args.num_epochs:
        model.backbone.load_state_dict(best_state)
        logger.info("Restored backbone weights from epoch %d", best_epoch)
    return RepresentationResult(steps, train_losses, eval_losses, best_epoch)
