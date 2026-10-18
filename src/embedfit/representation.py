"""Phase 1 of training: adapting the backbone under a contrastive objective.

The optimisation loop is shared by every objective. An objective supplies the batches (and how
they change from epoch to epoch) and turns one batch into a loss.
"""

import logging
import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from embedfit.augmentation.pipeline import Augmentation, augment_image, probe_augmentation
from embedfit.augmentation.registry import create_augmentation
from embedfit.data.dataset import ImageDataset
from embedfit.data.image import to_rgb_tensor
from embedfit.exceptions import ConfigurationError, TrainingDivergedError
from embedfit.heads.linear import LinearHead
from embedfit.losses.registry import create_loss
from embedfit.model import EmbedFitModel
from embedfit.sampling.balanced import batch_shape as batch_shape  # Kept importable from here.
from embedfit.sampling.pairs import ImagePair, PairSampler, RelationshipRule
from embedfit.sampling.registry import BatchSampler, create_sampler, needs_batch_check
from embedfit.training_args import TrainingArguments

logger = logging.getLogger(__name__)

_EVAL_SEED_OFFSET = 1_000_003  # Keeps validation batches independent of training batches.
_DEFAULT_SAMPLERS = {"supcon": "balanced", "end_to_end": "random"}


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


class _AugmentedPixelDataset(Dataset[tuple[torch.Tensor, int]]):
    """Like ``_PixelDataset``, but each image is augmented before preprocessing.

    Indexed by the ``(epoch, draw, position)`` keys of ``_DrawKeys``, so every draw gets its
    own seed, whichever worker process loads it.
    """

    def __init__(
        self,
        model: EmbedFitModel,
        dataset: ImageDataset,
        targets: np.ndarray,
        augmentation: Augmentation,
        seed: int,
    ) -> None:
        self._backbone = model.backbone
        self._images = dataset.images
        self._targets = targets.tolist()
        self._augmentation = augmentation
        self._seed = seed

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, key: tuple[int, int, int]) -> tuple[torch.Tensor, int]:
        epoch, draw, position = key
        image = augment_image(
            self._augmentation,
            to_rgb_tensor(self._images[position]),
            seed=self._seed,
            epoch=epoch,
            draw=draw,
        )
        return self._backbone.preprocess(image), self._targets[position]


class _DrawKeys:
    """Turn a batch sampler's positions into ``(epoch, draw, position)`` keys.

    ``draw`` counts the images drawn so far in the epoch, so a sample drawn twice gets two
    different augmentation seeds. Length and ``set_epoch`` are forwarded to the wrapped sampler.
    """

    def __init__(self, batches: BatchSampler) -> None:
        self._batches = batches
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        self._batches.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self._batches)

    def __iter__(self) -> Iterator[list[tuple[int, int, int]]]:
        draw = 0
        for batch in self._batches:
            yield [(self._epoch, draw + offset, position) for offset, position in enumerate(batch)]
            draw += len(batch)


def _batch_sampler(args: TrainingArguments, targets: np.ndarray, seed: int) -> BatchSampler:
    """The batch sampler of ``supcon`` and ``end_to_end`` training: ``args.sampler``, or the
    mode's default. ``targets`` are the class indices of the samples, in dataset order."""
    name = args.sampler or _DEFAULT_SAMPLERS[args.training_mode]
    return create_sampler(name, targets, batch_size=args.batch_size, seed=seed)


def _loader(
    model: EmbedFitModel,
    dataset: ImageDataset,
    args: TrainingArguments,
    seed: int,
    augmentation: Augmentation | None = None,
) -> tuple[DataLoader[tuple[torch.Tensor, int]], BatchSampler | _DrawKeys]:
    """A loader of preprocessed images and class indices, and the sampler to set epochs on."""
    assert model.label_mapping is not None
    targets = model.label_mapping.encode(dataset.labels)
    sampler = _batch_sampler(args, targets, seed)
    if augmentation is None:
        loader = DataLoader(
            _PixelDataset(model, dataset, targets),
            batch_sampler=sampler,
            num_workers=args.num_workers,
        )
        return loader, sampler
    keys = _DrawKeys(sampler)
    augmented = DataLoader(
        _AugmentedPixelDataset(model, dataset, targets, augmentation, seed),
        batch_sampler=keys,
        num_workers=args.num_workers,
    )
    return augmented, keys


def _check_batches(
    sampler: BatchSampler, targets: np.ndarray, epochs: range, *, need_positive: bool
) -> None:
    """Iterate every epoch's batches, positions only, so that a malformed batch from a custom
    sampler fails before training instead of part-way through it.

    Raises:
        ConfigurationError: If an epoch yields no batches, a batch is empty, not a flat list of
            integer positions, holds a position outside the dataset, or (``need_positive``)
            holds no two samples of the same class; or if ``len(sampler)`` differs from the
            number of batches yielded or between epochs (the learning-rate schedule is sized
            from it once).
    """
    first_length = None
    for epoch in epochs:
        sampler.set_epoch(epoch)
        count = 0
        for batch in sampler:
            count += 1
            positions = np.asarray(batch)
            where = f"Sampler batch {count} of epoch {epoch + 1}"
            if positions.size == 0:
                raise ConfigurationError(f"{where} is an empty batch.")
            if positions.ndim != 1 or not np.issubdtype(positions.dtype, np.integer):
                raise ConfigurationError(f"{where} must be a flat list of integer positions.")
            if positions.min() < 0 or positions.max() >= len(targets):
                raise ConfigurationError(
                    f"{where} holds a position outside the {len(targets)} samples."
                )
            if need_positive and len(np.unique(targets[positions])) == len(positions):
                raise ConfigurationError(
                    f"{where} holds no two samples of the same class, which the contrastive "
                    "loss needs."
                )
        if count == 0:
            raise ConfigurationError(f"The sampler yielded no batches in epoch {epoch + 1}.")
        if count != len(sampler):
            raise ConfigurationError(
                f"The sampler reports {len(sampler)} batches but yielded {count} in epoch "
                f"{epoch + 1}."
            )
        if first_length is None:
            first_length = count
        elif count != first_length:
            raise ConfigurationError(
                "The sampler must yield the same number of batches in every epoch, got "
                f"{first_length} in epoch 1 and {count} in epoch {epoch + 1}."
            )


class _PairDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int, float]]):
    """The current epoch's pairs as preprocessed pixel tensors, targets, and weights.

    Without mining, every epoch's pairs depend only on the seed, so they are drawn once and
    cached (the step count needs all epochs' sizes up front). With hard-negative mining the
    pairs depend on the backbone's current weights, so each epoch is drawn when it starts,
    from embeddings of this dataset that are refreshed every ``refresh_epochs`` epochs.
    """

    def __init__(
        self,
        model: EmbedFitModel,
        dataset: ImageDataset,
        sampler: PairSampler,
        *,
        hard_negative_fraction: float = 0.0,
        refresh_epochs: int = 1,
        encode_batch_size: int = 64,
        augmentation: Augmentation | None = None,
        seed: int = 0,
    ) -> None:
        self._model = model
        self._backbone = model.backbone
        self._images = dataset.images
        self._sampler = sampler
        self._fraction = hard_negative_fraction
        self._refresh_epochs = refresh_epochs
        self._encode_batch_size = encode_batch_size
        self._augmentation = augmentation
        self._seed = seed
        self._epoch = 0
        self._embeddings: np.ndarray | None = None
        self._cache: dict[int, list[ImagePair]] = {}
        self.pairs: list[ImagePair] = [] if self.mines else self.pairs_for(0)

    @property
    def mines(self) -> bool:
        return self._fraction > 0

    def pairs_for(self, epoch: int) -> list[ImagePair]:
        if epoch not in self._cache:
            self._cache[epoch] = self._sampler.sample(epoch)
        return self._cache[epoch]

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        if not self.mines:
            self.pairs = self.pairs_for(epoch)
            return
        if self._embeddings is None or epoch % self._refresh_epochs == 0:
            # encode() runs without gradients in eval mode and restores the training mode.
            encoded = self._model.encode(self._images, batch_size=self._encode_batch_size)
            assert isinstance(encoded, np.ndarray)
            self._embeddings = encoded
            logger.info("Refreshed %d training embeddings for hard-negative mining", len(encoded))
        self.pairs = self._sampler.sample(
            epoch, embeddings=self._embeddings, hard_negative_fraction=self._fraction
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor, int, float]:
        pair = self.pairs[position]
        return (
            self._pixels(pair.left, 2 * position),
            self._pixels(pair.right, 2 * position + 1),
            pair.target,
            pair.weight,
        )

    def _pixels(self, index: int, draw: int) -> torch.Tensor:
        if self._augmentation is None:
            return self._backbone.preprocess(self._images[index])
        image = augment_image(
            self._augmentation,
            to_rgb_tensor(self._images[index]),
            seed=self._seed,
            epoch=self._epoch,
            draw=draw,
        )
        return self._backbone.preprocess(image)


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
    mine: bool = False,
    augmentation: Augmentation | None = None,
) -> _Objective:
    """Build the objective selected by ``args.training_mode`` for one dataset.

    ``classifier`` is the jointly trained linear layer of ``end_to_end`` mode; without one the
    objective can check the data but not compute a loss.
    ``mine`` enables hard-negative mining (``pair`` mode, training data only).
    ``augmentation`` is applied to every image drawn (training data only).
    """
    parameter = next(model.backbone.parameters(), None)
    device = model.device
    dtype = parameter.dtype if parameter is not None else torch.float32

    if args.training_mode == "supcon":
        loader, sampler = _loader(model, dataset, args, seed=seed, augmentation=augmentation)
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
        loader, sampler = _loader(model, dataset, args, seed=seed, augmentation=augmentation)

        def cross_entropy(batch: Any) -> torch.Tensor:
            if classifier is None:
                raise ConfigurationError("end_to_end training needs a classifier layer.")
            pixels, labels = batch
            logits = classifier(model(pixels.to(device=device, dtype=dtype)))
            loss: torch.Tensor = nn.functional.cross_entropy(logits.float(), labels.to(device))
            return loss

        return _Objective(loader, sampler.set_epoch, lambda epoch: len(sampler), cross_entropy)

    model.label_mapping.encode(dataset.labels)  # Unknown labels are an error in every mode.
    pair_sampler = PairSampler(
        dataset,
        positives_per_example=args.positives_per_example,
        negatives_per_example=args.negatives_per_example,
        seed=seed,
        rules=rules,
    )
    pairs = _PairDataset(
        model,
        dataset,
        pair_sampler,
        # Only the training objective mines; validation pairs stay random and fixed.
        hard_negative_fraction=args.hard_negative_fraction if mine else 0.0,
        refresh_epochs=args.hard_negative_refresh_epochs,
        encode_batch_size=args.eval_batch_size,
        augmentation=augmentation,
        seed=seed,
    )
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
        # Mined pairs cannot be drawn ahead of time, so the schedule is planned on the most
        # pairs an epoch can hold; the learning rate then ends slightly above zero.
        count = len(pair_sampler) if pairs.mines else len(pairs.pairs_for(epoch))
        return math.ceil(count / args.batch_size)

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
            formed; ``end_to_end``: a class of the model without training samples; ``supcon``
            and ``end_to_end``: a batch of a sampler other than the built-in ``balanced``, in
            any epoch, that is empty, holds a position outside the data, or (``supcon``) holds
            no two samples of one class, or such a sampler that yields no batches, or whose
            length differs from its batch count or between epochs),
            relationship rules are given for a mode other than ``pair``, the backbone
            has no trainable parameters, the mixed-precision mode is not supported on the
            device, or ``args.augmentation`` is not registered or its pipeline returns a
            malformed image when probed with a synthetic one.
    """
    rules = tuple(rules)
    if rules and args.training_mode != "pair":
        raise ConfigurationError(
            f"relationship_rules only apply to training_mode='pair', not '{args.training_mode}'."
        )
    checks = (
        ("train_dataset", train_dataset, args.seed, range(args.num_epochs)),
        # Validation batches are those of epoch 0 after every epoch.
        ("eval_dataset", eval_dataset, args.seed + _EVAL_SEED_OFFSET, range(1)),
    )
    for name, dataset, seed, epochs in checks:
        if dataset is not None:
            if len(dataset) == 0:
                raise ConfigurationError(f"{name} is empty.")
            _objective(model, dataset, args, rules, seed=seed)
            sampler = args.sampler or _DEFAULT_SAMPLERS.get(args.training_mode)
            if sampler is not None and needs_batch_check(sampler):
                assert model.label_mapping is not None
                targets = model.label_mapping.encode(dataset.labels)
                _check_batches(
                    _batch_sampler(args, targets, seed),
                    targets,
                    epochs,
                    need_positive=args.training_mode == "supcon",
                )
    augmentation = create_augmentation(args.augmentation)
    if augmentation is not None:
        probe_augmentation(augmentation, seed=args.seed)
    if args.training_mode == "end_to_end":
        require_every_class(model, train_dataset)  # The starting linear probe needs them all.
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


def require_every_class(model: EmbedFitModel, dataset: ImageDataset) -> None:
    """Raise ``ConfigurationError`` unless every class of the model has a sample in ``dataset``.

    The message gives counts only, never label values.
    """
    assert model.label_mapping is not None
    present = set(model.label_mapping.encode(dataset.labels).tolist())
    absent = len(model.label_mapping) - len(present)
    if absent:
        raise ConfigurationError(
            f"{absent} of the model's {len(model.label_mapping)} labels have no samples in "
            "train_dataset; every class needs at least one."
        )


def _linear_probe(
    model: EmbedFitModel, dataset: ImageDataset, args: TrainingArguments
) -> nn.Linear:
    """The starting classifier of ``end_to_end`` mode: a converged linear probe (ADR 0012).

    Fitted on the embeddings the model produces before any backbone update, with the same
    ``LinearHead`` settings as the head that is refitted after training (ADR 0011).
    """
    assert model.label_mapping is not None
    features = model.encode(dataset.images, batch_size=args.eval_batch_size)
    targets = model.label_mapping.encode(dataset.labels)
    probe = LinearHead(seed=args.seed)
    probe.fit(np.asarray(features), targets, num_classes=len(model.label_mapping))
    layer = probe.to_module()
    with torch.no_grad():
        logits = layer(torch.from_numpy(np.asarray(features)))
        loss = nn.functional.cross_entropy(logits, torch.from_numpy(targets.astype(np.int64)))
    probe_loss = loss.item()  # A mean over the training set: an aggregate, safe to log.
    logger.info("Starting from a linear probe: training cross-entropy %.4f", probe_loss)
    return layer


def _freeze_batch_norm(model: EmbedFitModel) -> None:
    """Put batch-norm layers in eval mode: running statistics fixed, affine parameters trained."""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


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
    ``args.sampler`` replaces the batches of ``"supcon"`` and ``"end_to_end"`` with those of a
    registered sampler, for training and validation alike.
    ``"pair"`` draws a bounded set of positive and negative pairs each epoch (``PairSampler``)
    and applies the cosine pair loss; ``rules`` optionally decide pair targets and weights
    from fields other than the class label. ``"end_to_end"`` is the comparison baseline:
    linear-probe-then-fine-tune (ADR 0012). A ``LinearHead`` is first fitted to convergence on
    the embeddings of the untouched backbone; that linear classifier is then trained jointly
    with the backbone under cross-entropy on uniformly shuffled batches by default, and returned in
    ``classifier``. The classifier sees exactly the embeddings the model produces (normalised
    or not) and ``temperature`` plays no part, so the baseline shares no hyperparameter with
    the contrastive losses.

    Each epoch draws its batches from ``train_dataset`` only. When ``eval_dataset``
    is given, its loss is measured after every epoch (without gradients) on a
    fixed set of batches, so it is a sampled estimate rather than the mean over
    every validation sample, and the weights of the best epoch are restored at the end;
    otherwise the last epoch is kept. ``eval_dataset`` must be validation data: selecting a
    checkpoint on test data leaks it.

    ``args.augmentation`` names a task augmentation applied to every image drawn from
    ``train_dataset``, before the backbone's preprocessing, with a seed derived from ``args.seed``,
    the epoch, and the draw's position in the epoch (ADR 0014). Validation batches, hard-negative
    mining, and the ``end_to_end`` linear probe use the original images.

    Images are preprocessed on the fly each epoch; nothing is cached. With ``num_workers > 0``
    each worker process receives a copy of the backbone (its ``preprocess`` method does the
    work), which costs start-up time that only pays off on larger datasets.

    This function only updates backbone weights. Freezing the backbone and switching the
    model to eval mode afterwards is the caller's job. Without ``eval_dataset`` the model is
    returned in train mode, with its batch-norm layers in eval mode if
    ``args.freeze_batch_norm`` is set.

    Args:
        model: The model whose backbone is trained. It must have labels.
        train_dataset: Training samples.
        eval_dataset: Optional validation samples for checkpoint selection.
        args: Hyperparameters: ``num_epochs``, ``batch_size``, ``learning_rate``,
            ``weight_decay``, ``warmup_ratio``, ``temperature``,
            ``gradient_accumulation_steps``, ``max_grad_norm``, ``mixed_precision``,
            ``num_workers``, ``seed``, ``augmentation``, and for pairs
            ``positives_per_example`` and ``negatives_per_example``.
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
        # Built before the objectives, the optimizer, and the best-epoch state, which must all
        # hold this same layer.
        classifier = _linear_probe(model, train_dataset, args)
        first = next(model.backbone.parameters(), None)
        classifier.to(device=device, dtype=first.dtype if first is not None else torch.float32)
    augmentation = create_augmentation(args.augmentation)
    if augmentation is not None:
        logger.info("Augmenting training images with '%s'", args.augmentation)
    objective = _objective(
        model,
        train_dataset,
        args,
        rules,
        seed=args.seed,
        classifier=classifier,
        mine=True,
        augmentation=augmentation,
    )
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
        objective.set_epoch(epoch)  # May re-encode for mining, which restores train mode.
        if args.freeze_batch_norm:
            _freeze_batch_norm(model)
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
            # The same batches every epoch, and a mean over samples rather than over batches:
            # otherwise a reshuffled, uneven last batch makes identical weights score
            # differently from one epoch to the next and checkpoint selection follows noise.
            validation.set_epoch(0)
            total, count = 0.0, 0
            with torch.no_grad(), _autocast(args, device):
                for batch in validation.loader:
                    size = len(batch[0])
                    total += validation.loss(batch).item() * size
                    count += size
            if count == 0:
                raise ConfigurationError("The validation data produced no batches.")
            eval_losses.append(total / count)
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
