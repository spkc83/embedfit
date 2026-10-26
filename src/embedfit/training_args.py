"""Training configuration, validated when it is created."""

from dataclasses import asdict, dataclass
from pathlib import Path

from embedfit.augmentation.registry import NO_AUGMENTATION, available_augmentations
from embedfit.backbones.base import TRAINABLE_PARAMETERS
from embedfit.exceptions import ConfigurationError
from embedfit.heads.registry import available_heads
from embedfit.sampling.pairs import hard_negative_quota
from embedfit.sampling.registry import available_samplers

TRAINING_MODES = ("frozen", "supcon", "pair", "triplet", "end_to_end")
MIXED_PRECISION_MODES = (None, "fp16", "bf16")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigurationError(message)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _paths_overlap(first: str | Path, second: str | Path) -> bool:
    """Whether two directories are the same, or one lies inside the other."""
    a, b = Path(first).resolve(), Path(second).resolve()
    return a == b or a in b.parents or b in a.parents


@dataclass
class TrainingArguments:
    """Hyperparameters for ``Trainer``. Every field is validated in ``__post_init__``.

    Attributes:
        training_mode: ``"frozen"`` (no backbone updates), ``"supcon"``, ``"pair"``,
            ``"triplet"``, or ``"end_to_end"``.
        num_epochs: Representation-training epochs (>= 1). Unused in ``frozen`` mode, and
            ignored when ``max_steps`` is set.
        batch_size: Representation-training batch size (>= 1). Unused in ``frozen`` mode.
        eval_batch_size: Images per forward pass when encoding for head fitting and prediction.
        learning_rate: Backbone learning rate (> 0); with ``layer_lr_decay``, that of the last
            block and the parameters after it.
        weight_decay: Weight decay (>= 0).
        warmup_ratio: Fraction of steps used for learning-rate warm-up, in [0, 1).
        temperature: Contrastive-loss temperature (> 0).
        triplet_margin: Triplet-loss margin (>= 0).
        gradient_accumulation_steps: Batches per optimizer step (>= 1).
        mixed_precision: ``None``, ``"fp16"``, or ``"bf16"``.
        max_grad_norm: Gradient-clipping norm (> 0), or ``None`` to disable clipping.
        seed: Seed for every random number generator, in ``[0, 2**32 - 1]``.
        num_workers: Data-loading worker processes for representation training (>= 0). Each
            worker costs start-up time, so 0 is faster on small datasets. Unused in ``frozen``
            mode.
        device: ``None`` leaves the model on its current device; ``"auto"`` selects
            CUDA → MPS → CPU; any other value is an explicit request that never falls back.
        freeze_backbone_before_head: Freeze the backbone before the head is fitted. Must be
            true in ``frozen`` mode.
        head: Name of a registered classification head. A model created with a configured
            head instance keeps that instance; see ``Trainer``.
        output_dir: Directory that saved models are written to.
        deterministic: Request deterministic torch algorithms (slower).
        positives_per_example: ``pair`` mode: positive partners drawn per sample each epoch.
        negatives_per_example: ``pair`` mode: negative partners drawn per sample each epoch.
            At least one of the two must be positive.
        hard_negative_fraction: ``pair`` mode: share of each sample's negative partners, in
            [0, 1], chosen as its most similar different-class training samples under the
            current backbone instead of at random. The share is rounded half up to a whole
            number of negatives and must not round to zero. 0 disables mining.
        hard_negative_refresh_epochs: Re-encode the training set for mining every this many
            epochs (>= 1). Encoding runs without gradients and never sees validation data.
        freeze_batch_norm: Keep batch-normalisation layers in eval mode during representation
            training, so their running statistics stay those of the pretrained backbone while
            their scale and shift parameters still train (ADR 0013). Few-shot batches are too
            small for reliable batch statistics. No effect on backbones without batch norm, or on
            batch-norm layers built without running statistics.
        sampler: Name of a registered batch sampler for ``supcon`` and ``end_to_end`` training,
            used for both training and validation batches. ``None`` selects the mode's default:
            ``"balanced"`` (``BalancedClassBatchSampler``) for ``supcon``, ``"random"`` for
            ``end_to_end``. Rejected in other modes.
        augmentation: Task augmentation applied to the images that representation training
            draws from the training set: ``"none"``, ``"natural"``, ``"document"``, or a name
            given to ``embedfit.augmentation.register_augmentation``. It runs before the
            backbone's own preprocessing; validation, mining, head fitting, and prediction
            always see the original images (ADR 0014). Must be ``"none"`` in ``frozen`` mode.
        unfreeze_last_blocks: Train only the last this many blocks of the backbone (>= 1),
            plus the parameters after them (such as a final norm), in every mode that trains
            the backbone; the earlier blocks and the stem stay frozen, including their
            batch-norm statistics (ADR 0015). What a block is depends on the backbone
            (``ImageBackbone.parameter_blocks``). ``None`` trains the whole backbone.
        checkpoint_dir: Directory that representation training writes a checkpoint to after
            every epoch, keeping the best and the last (ADR 0016). It must be missing or empty
            when training starts and must not overlap ``output_dir``. ``None`` writes nothing.
        max_steps: Representation-training budget in optimizer steps (>= 1), replacing
            ``num_epochs``, which is then ignored: training runs as many epochs as it needs and
            stops after this many steps, part-way through an epoch if need be, and the
            learning-rate schedule is sized on it (ADR 0017). ``None`` trains for
            ``num_epochs`` epochs. Rejected in ``frozen`` mode.
        projection_dim: ``supcon`` mode: compute the contrastive loss on a projection head
            ``Linear(D, D) → ReLU → Linear(D, projection_dim)`` over the backbone's
            embeddings, trained with the backbone and discarded afterwards; the classification
            head and every encoding use the backbone's embeddings (ADR 0018). ``None`` applies
            the loss to the embeddings directly. Rejected in other modes.
        contrastive_views: ``supcon`` mode: augmented views of every drawn image (>= 1), each
            with its own augmentation seed; the views of one image are positives of each other
            (ADR 0018). Above 1 it needs an ``augmentation`` other than ``"none"``. Validation
            batches use one view of the original images. Must be 1 in other modes.
        head_learning_rate: Learning rate (> 0) of the parameters that representation training
            creates besides the backbone: the ``end_to_end`` classifier and the ``supcon``
            projection head (ADR 0020). ``None`` trains them at ``learning_rate``. Rejected when
            training creates no such parameters (``frozen``, ``pair``, ``triplet``, and
            ``supcon`` without ``projection_dim``).
        layer_lr_decay: Layer-wise learning-rate decay of the backbone, in (0, 1] (ADR 0020):
            block ``i`` of ``n`` (``ImageBackbone.parameter_blocks``, input side first) trains
            at ``learning_rate * layer_lr_decay ** (n - 1 - i)``, the stem before the first
            block at ``layer_lr_decay ** n`` times it, and the parameters after the last block
            at ``learning_rate``. Needs a backbone that defines blocks. ``None`` trains every
            backbone parameter at ``learning_rate``. Rejected in ``frozen`` mode.
        trainable_parameters: Which backbone parameters train, in every mode that trains the
            backbone (ADR 0021): ``"all"``; ``"layernorm"``, only the parameters of
            normalisation layers (layer, group, and batch norm, the latter's running statistics
            still governed by ``freeze_batch_norm``); or ``"bias"``, only bias terms (BitFit).
            With ``unfreeze_last_blocks`` it selects among the parameters of those blocks.
            The ``end_to_end`` classifier and the ``supcon`` projection head always train.
            Must be ``"all"`` in ``frozen`` mode.

    Raises:
        ConfigurationError: If any value is invalid. Raised at construction, never mid-training.
    """

    training_mode: str = "supcon"
    num_epochs: int = 5
    batch_size: int = 32
    eval_batch_size: int = 64
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.0
    temperature: float = 0.07
    triplet_margin: float = 0.2
    gradient_accumulation_steps: int = 1
    mixed_precision: str | None = None
    max_grad_norm: float | None = 1.0
    seed: int = 42
    num_workers: int = 4
    device: str | None = None
    freeze_backbone_before_head: bool = True
    head: str = "logistic_regression"
    output_dir: str = "./embedfit-output"
    deterministic: bool = False
    positives_per_example: int = 4
    negatives_per_example: int = 4
    hard_negative_fraction: float = 0.0
    hard_negative_refresh_epochs: int = 1
    freeze_batch_norm: bool = True
    sampler: str | None = None
    augmentation: str = NO_AUGMENTATION
    unfreeze_last_blocks: int | None = None
    checkpoint_dir: str | None = None
    max_steps: int | None = None
    projection_dim: int | None = None
    contrastive_views: int = 1
    head_learning_rate: float | None = None
    layer_lr_decay: float | None = None
    trainable_parameters: str = "all"

    def __post_init__(self) -> None:
        _require(
            self.training_mode in TRAINING_MODES,
            f"training_mode must be one of {list(TRAINING_MODES)}, got {self.training_mode!r}.",
        )
        for name in ("num_epochs", "batch_size", "eval_batch_size", "gradient_accumulation_steps"):
            value = getattr(self, name)
            _require(_is_int(value) and value >= 1, f"{name} must be an int >= 1, got {value!r}.")
        _require(
            _is_int(self.num_workers) and self.num_workers >= 0,
            f"num_workers must be an int >= 0, got {self.num_workers!r}.",
        )
        for name in ("positives_per_example", "negatives_per_example"):
            value = getattr(self, name)
            _require(_is_int(value) and value >= 0, f"{name} must be an int >= 0, got {value!r}.")
        _require(
            self.positives_per_example + self.negatives_per_example > 0,
            "positives_per_example and negatives_per_example cannot both be 0.",
        )
        _require(
            _is_number(self.hard_negative_fraction) and 0 <= self.hard_negative_fraction <= 1,
            f"hard_negative_fraction must be in [0, 1], got {self.hard_negative_fraction!r}.",
        )
        _require(
            _is_int(self.hard_negative_refresh_epochs) and self.hard_negative_refresh_epochs >= 1,
            "hard_negative_refresh_epochs must be an int >= 1, "
            f"got {self.hard_negative_refresh_epochs!r}.",
        )
        _require(
            self.hard_negative_fraction == 0
            or (self.training_mode == "pair" and self.negatives_per_example > 0),
            "hard_negative_fraction > 0 requires training_mode='pair' with "
            "negatives_per_example > 0.",
        )
        _require(
            self.hard_negative_fraction == 0
            or hard_negative_quota(self.negatives_per_example, self.hard_negative_fraction) > 0,
            f"hard_negative_fraction={self.hard_negative_fraction} of "
            f"negatives_per_example={self.negatives_per_example} rounds to no mined negative at "
            "all; raise one of them or set the fraction to 0.",
        )
        _require(
            _is_int(self.seed) and 0 <= self.seed <= 2**32 - 1,
            f"seed must be an int in [0, 2**32 - 1], got {self.seed!r}.",
        )
        for name in ("learning_rate", "temperature"):
            value = getattr(self, name)
            _require(_is_number(value) and value > 0, f"{name} must be > 0, got {value!r}.")
        for name in ("weight_decay", "triplet_margin"):
            value = getattr(self, name)
            _require(_is_number(value) and value >= 0, f"{name} must be >= 0, got {value!r}.")
        _require(
            _is_number(self.warmup_ratio) and 0 <= self.warmup_ratio < 1,
            f"warmup_ratio must be in [0, 1), got {self.warmup_ratio!r}.",
        )
        _require(
            self.max_grad_norm is None
            or (_is_number(self.max_grad_norm) and self.max_grad_norm > 0),
            f"max_grad_norm must be > 0 or None, got {self.max_grad_norm!r}.",
        )
        _require(
            self.mixed_precision in MIXED_PRECISION_MODES,
            f"mixed_precision must be one of {list(MIXED_PRECISION_MODES)}, "
            f"got {self.mixed_precision!r}.",
        )
        _require(
            self.head in available_heads(),
            f"head must be one of {available_heads()}, got {self.head!r}.",
        )
        _require(
            self.sampler is None or self.sampler in available_samplers(),
            f"sampler must be one of {available_samplers()} or None, got {self.sampler!r}.",
        )
        _require(
            self.sampler is None or self.training_mode in ("supcon", "end_to_end"),
            f"sampler only applies to training_mode 'supcon' or 'end_to_end', not "
            f"'{self.training_mode}'.",
        )
        _require(
            isinstance(self.augmentation, str) and self.augmentation in available_augmentations(),
            f"augmentation must be one of {available_augmentations()}, got {self.augmentation!r}.",
        )
        _require(
            self.augmentation == NO_AUGMENTATION or self.training_mode != "frozen",
            f"augmentation={self.augmentation!r} has no effect in training_mode='frozen', which "
            "trains no representation; use 'none'.",
        )
        _require(
            self.device is None or isinstance(self.device, str),
            f"device must be a str or None, got {type(self.device).__name__}.",
        )
        for name in ("freeze_backbone_before_head", "deterministic", "freeze_batch_norm"):
            value = getattr(self, name)
            _require(isinstance(value, bool), f"{name} must be a bool, got {value!r}.")
        _require(
            isinstance(self.output_dir, str) and bool(self.output_dir),
            "output_dir must be a non-empty str.",
        )
        _require(
            self.unfreeze_last_blocks is None
            or (_is_int(self.unfreeze_last_blocks) and self.unfreeze_last_blocks >= 1),
            f"unfreeze_last_blocks must be an int >= 1 or None, got {self.unfreeze_last_blocks!r}.",
        )
        _require(
            self.checkpoint_dir is None
            or (isinstance(self.checkpoint_dir, str) and bool(self.checkpoint_dir)),
            "checkpoint_dir must be a non-empty str or None.",
        )
        for name in ("max_steps", "projection_dim"):
            value = getattr(self, name)
            _require(
                value is None or (_is_int(value) and value >= 1),
                f"{name} must be an int >= 1 or None, got {value!r}.",
            )
        _require(
            _is_int(self.contrastive_views) and self.contrastive_views >= 1,
            f"contrastive_views must be an int >= 1, got {self.contrastive_views!r}.",
        )
        _require(
            self.projection_dim is None or self.training_mode == "supcon",
            f"projection_dim only applies to training_mode='supcon', not '{self.training_mode}'.",
        )
        _require(
            self.contrastive_views == 1 or self.training_mode == "supcon",
            "contrastive_views only applies to training_mode='supcon', not "
            f"'{self.training_mode}'.",
        )
        _require(
            self.contrastive_views == 1 or self.augmentation != NO_AUGMENTATION,
            f"contrastive_views={self.contrastive_views} needs an augmentation: without one "
            "every view of an image is identical.",
        )
        _require(
            self.head_learning_rate is None
            or (_is_number(self.head_learning_rate) and self.head_learning_rate > 0),
            f"head_learning_rate must be > 0 or None, got {self.head_learning_rate!r}.",
        )
        _require(
            self.head_learning_rate is None
            or self.training_mode == "end_to_end"
            or self.projection_dim is not None,
            "head_learning_rate applies to the classifier of training_mode='end_to_end' and the "
            "projection head of 'supcon' (projection_dim); this setup trains neither.",
        )
        _require(
            self.layer_lr_decay is None
            or (_is_number(self.layer_lr_decay) and 0 < self.layer_lr_decay <= 1),
            f"layer_lr_decay must be in (0, 1] or None, got {self.layer_lr_decay!r}.",
        )
        _require(
            self.trainable_parameters in TRAINABLE_PARAMETERS,
            f"trainable_parameters must be one of {list(TRAINABLE_PARAMETERS)}, "
            f"got {self.trainable_parameters!r}.",
        )
        _require(
            self.trainable_parameters == "all" or self.training_mode != "frozen",
            "trainable_parameters only applies to modes that train the backbone, not 'frozen'.",
        )
        for name in ("unfreeze_last_blocks", "checkpoint_dir", "max_steps", "layer_lr_decay"):
            _require(
                self.training_mode != "frozen" or getattr(self, name) is None,
                f"{name} only applies to modes that train the backbone, not 'frozen'.",
            )
        _require(
            self.checkpoint_dir is None or not _paths_overlap(self.checkpoint_dir, self.output_dir),
            "checkpoint_dir and output_dir must not be the same directory or contain each "
            "other: saving the model replaces output_dir as a whole.",
        )


def _recorded_arguments(args: TrainingArguments) -> dict[str, object]:
    """The training arguments stored with a saved model or checkpoint. ``output_dir`` and
    ``checkpoint_dir`` are left out because a local path can reveal a user name."""
    return {
        name: value
        for name, value in asdict(args).items()
        if name not in ("output_dir", "checkpoint_dir")
    }
