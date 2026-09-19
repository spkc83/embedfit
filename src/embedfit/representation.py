"""Phase 1 of training: adapting the backbone under a contrastive objective.

The optimisation loop is shared by every objective. An objective supplies the batches (and how
they change from epoch to epoch) and turns one batch into a loss.
"""

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler

from embedfit.data.dataset import ImageDataset
from embedfit.exceptions import ConfigurationError, TrainingDivergedError
from embedfit.losses.registry import create_loss
from embedfit.model import EmbedFitModel
from embedfit.sampling.balanced import BalancedClassBatchSampler
from embedfit.sampling.pairs import ImagePair, PairSampler, RelationshipRule
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
        classifier: ``end_to_end`` mode only: the linear layer trained jointly with the
            backbone, holding the kept epoch's weights.
        best_epoch: 1-based epoch whose weights were kept: the one with the lowest validation
            loss (the earliest on ties), or the last epoch without a validation set.
    """

    steps: int
    train_losses: list[float] = field(default_factory=list)
    eval_losses: list[float] = field(default_factory=list)
    best_epoch: int = 0
    classifier: nn.Linear | None = None


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


class _PairDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int, float]]):
    """The current epoch's pairs as preprocessed pixel tensors, targets, and weights."""

    def __init__(self, model: EmbedFitModel, dataset: ImageDataset, sampler: PairSampler) -> None:
        self._backbone = model.backbone
        self._images = dataset.images
        self._sampler = sampler
        # Every epoch is sampled once: the step count needs all epochs' sizes up front, and
        # training then reuses the same lists. At most E * N * (positives + negatives) pairs.
        self._cache: dict[int, list[ImagePair]] = {}
        self.pairs: list[ImagePair] = self.pairs_for(0)

    def pairs_for(self, epoch: int) -> list[ImagePair]:
        if epoch not in self._cache:
            self._cache[epoch] = self._sampler.sample(epoch)
        return self._cache[epoch]

    def set_epoch(self, epoch: int) -> None:
        self.pairs = self.pairs_for(epoch)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor, int, float]:
        pair = self.pairs[position]
        preprocess = self._backbone.preprocess
        return (
            preprocess(self._images[pair.left]),
            preprocess(self._images[pair.right]),
            pair.target,
            pair.weight,
        )


@dataclass
class _Objective:
    """Batches and loss of one representation-learning objective."""

    loader: DataLoader[Any]
    set_epoch: Callable[[int], None]
    batches_in_epoch: Callable[[int], int]
    loss: Callable[[Any], torch.Tensor]


def _objective(
    model: EmbedFitModel,
    dataset: ImageDataset,
    args: TrainingArguments,
    rules: Sequence[RelationshipRule],
    seed: int,
    classifier: nn.Linear | None = None,
) -> _Objective:
    """Build the objective selected by ``args.training_mode`` for one dataset.

    ``classifier`` is the jointly trained linear layer of ``end_to_end`` mode. Validation
    builds the objective without one, which checks the data but cannot compute a loss.
    """
    parameter = next(model.backbone.parameters(), None)
    device = model.device
    dtype = parameter.dtype if parameter is not None else torch.float32

    if args.training_mode == "supcon":
        loader, sampler = _loader(model, dataset, args, seed=seed)
        supcon = create_loss("supcon", temperature=args.temperature)

        def supcon_loss(batch: Any) -> torch.Tensor:
            pixels, targets = batch
            loss: torch.Tensor = supcon(
                model(pixels.to(device=device, dtype=dtype)), targets.to(device)
            )
            return loss

        return _Objective(loader, sampler.set_epoch, lambda epoch: len(sampler), supcon_loss)

    assert model.label_mapping is not None
    if args.training_mode == "end_to_end":
        targets = model.label_mapping.encode(dataset.labels)
        generator = torch.Generator()
        pixels_dataset = _PixelDataset(model, dataset, targets)
        uniform: DataLoader[Any] = DataLoader(
            pixels_dataset,
            batch_size=args.batch_size,
            sampler=RandomSampler(pixels_dataset, generator=generator),
            num_workers=args.num_workers,
        )

        def reseed(epoch: int) -> None:
            generator.manual_seed(seed * 100_003 + epoch)

        def cross_entropy(batch: Any) -> torch.Tensor:
            if classifier is None:
                raise ConfigurationError("end_to_end training needs a classifier layer.")
            pixels, labels = batch
            logits = classifier(model(pixels.to(device=device, dtype=dtype)))
            loss: torch.Tensor = nn.functional.cross_entropy(logits.float(), labels.to(device))
            return loss

        reseed(0)
        return _Objective(uniform, reseed, lambda epoch: len(uniform), cross_entropy)

    model.label_mapping.encode(dataset.labels)  # Unknown labels are an error in every mode.
    pair_sampler = PairSampler(
        dataset,
        positives_per_example=args.positives_per_example,
        negatives_per_example=args.negatives_per_example,
        seed=seed,
        rules=rules,
    )
    pairs = _PairDataset(model, dataset, pair_sampler)
    pair_loader: DataLoader[Any] = DataLoader(
        pairs, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    cosine = create_loss("cosine_pair")

    def pair_loss(batch: Any) -> torch.Tensor:
        left, right, targets, weights = batch
        # One forward pass over both sides keeps batch statistics common to the pair.
        embeddings = model(torch.cat([left, right]).to(device=device, dtype=dtype))
        first, second = embeddings[: len(left)], embeddings[len(left) :]
        loss: torch.Tensor = cosine(first, second, targets.to(device), weights.to(device))
        return loss

    def batches(epoch: int) -> int:
        return math.ceil(len(pairs.pairs_for(epoch)) / args.batch_size)

    return _Objective(pair_loader, pairs.set_epoch, batches, pair_loss)


def validate_representation_setup(
    model: EmbedFitModel,
    train_dataset: ImageDataset,
    eval_dataset: ImageDataset | None,
    args: TrainingArguments,
    device: torch.device,
    rules: Sequence[RelationshipRule] = (),
) -> None:
    """Fail before training starts if contrastive training cannot run with this setup.

    Raises:
        ConfigurationError: If the training (or validation) data is empty or cannot supply
            the objective (``supcon``: batch size below 4, fewer than 2 classes, or a class
            with fewer than 2 samples; ``pair``: no positive or no negative pair can be
            formed), relationship rules are given for a mode other than ``pair``, the backbone
            has no trainable parameters, or the mixed-precision mode is not supported on the
            device.
    """
    rules = tuple(rules)
    if rules and args.training_mode != "pair":
        raise ConfigurationError(
            f"relationship_rules only apply to training_mode='pair', not '{args.training_mode}'."
        )
    for name, dataset in (("train_dataset", train_dataset), ("eval_dataset", eval_dataset)):
        if dataset is not None:
            if len(dataset) == 0:
                raise ConfigurationError(f"{name} is empty.")
            _objective(model, dataset, args, rules, seed=args.seed)
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
    rules: Sequence[RelationshipRule] = (),
    *,
    validate: bool = True,
) -> RepresentationResult:
    """Adapt the backbone with the objective named by ``args.training_mode``.

    ``"supcon"`` draws class-balanced batches and applies the supervised contrastive loss.
    ``"pair"`` draws a bounded set of positive and negative pairs each epoch (``PairSampler``)
    and applies the cosine pair loss; ``rules`` optionally decide pair targets and weights
    from fields other than the class label. ``"end_to_end"`` is the comparison baseline: a
    linear classifier on the model's embeddings is trained jointly with the backbone under
    cross-entropy on uniformly shuffled batches, and returned in ``classifier``. The classifier
    sees exactly the embeddings the model produces (normalised or not) and ``temperature`` plays
    no part, so the baseline shares no hyperparameter with the contrastive losses.

    Each epoch draws its batches from ``train_dataset`` only. When ``eval_dataset``
    is given, its loss is measured after every epoch (without gradients) on a
    fixed set of batches, so it is a sampled estimate rather than the mean over
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
            ``num_workers``, ``seed``, and for pairs ``positives_per_example`` and
            ``negatives_per_example``.
        rules: Relationship rules for ``"pair"`` mode.
        validate: Run ``validate_representation_setup`` first. ``Trainer`` has already done so
            when it was created and passes ``False``.

    Returns:
        A ``RepresentationResult``.

    Raises:
        ConfigurationError: If the setup is invalid (see ``validate_representation_setup``).
        TrainingDivergedError: If the loss becomes NaN or infinite.
    """
    device = model.device
    rules = tuple(rules)
    if validate:
        validate_representation_setup(model, train_dataset, eval_dataset, args, device, rules)
    classifier = None
    if args.training_mode == "end_to_end":
        assert model.label_mapping is not None
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.seed)
            classifier = nn.Linear(model.embedding_dim, len(model.label_mapping))
        first = next(model.backbone.parameters(), None)
        classifier.to(device=device, dtype=first.dtype if first is not None else torch.float32)
    objective = _objective(model, train_dataset, args, rules, seed=args.seed, classifier=classifier)
    loader = objective.loader
    validation = None
    if eval_dataset is not None:
        validation = _objective(
            model,
            eval_dataset,
            args,
            rules,
            seed=args.seed + _EVAL_SEED_OFFSET,
            classifier=classifier,
        )
    # Everything that training updates, so that the best epoch can be restored as a whole.
    trained = nn.ModuleList([model.backbone] + ([classifier] if classifier is not None else []))

    accumulation = args.gradient_accumulation_steps
    total_steps = sum(
        math.ceil(objective.batches_in_epoch(epoch) / accumulation)
        for epoch in range(args.num_epochs)
    )
    parameters = [p for p in trained.parameters() if p.requires_grad]
    if not parameters:
        raise ConfigurationError(
            "Every backbone parameter is frozen; call unfreeze_backbone() before training."
        )
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
        objective.set_epoch(epoch)
        batch_losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        for index, batch in enumerate(loader):
            with _autocast(args, device):
                loss = objective.loss(batch)
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

        if validation is not None:
            model.eval()
            with torch.no_grad(), _autocast(args, device):
                losses = [validation.loss(batch).item() for batch in validation.loader]
            eval_losses.append(float(np.mean(losses)))
            if len(eval_losses) == 1 or eval_losses[-1] < min(eval_losses[:-1]):  # Ties: earliest.
                best_epoch = epoch + 1
                # Kept on the CPU so that a GPU backbone is not held in memory twice.
                best_state = {
                    name: value.detach().to("cpu", copy=True)
                    for name, value in trained.state_dict().items()
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
        trained.load_state_dict(best_state)
        logger.info("Restored backbone weights from epoch %d", best_epoch)
    return RepresentationResult(steps, train_losses, eval_losses, best_epoch, classifier)
