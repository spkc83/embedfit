"""Training configuration, validated when it is created."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from embedfit.exceptions import ConfigurationError
from embedfit.heads.registry import available_heads
from embedfit.sampling.pairs import hard_negative_quota

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
        num_epochs: Representation-training epochs (>= 1). Unused in ``frozen`` mode.
        batch_size: Representation-training batch size (>= 1). Unused in ``frozen`` mode.
        eval_batch_size: Images per forward pass when encoding for head fitting and prediction.
        learning_rate: Backbone learning rate (> 0).
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
        unfreeze_last_blocks: Train only the last this many blocks of the backbone (>= 1),
            plus the parameters after them (such as a final norm), in every mode that trains
            the backbone; the earlier blocks and the stem stay frozen, including their
            batch-norm statistics (ADR 0015). What a block is depends on the backbone
            (``ImageBackbone.parameter_blocks``). ``None`` trains the whole backbone.
        checkpoint_dir: Directory that representation training writes a checkpoint to after
            every epoch, keeping the best and the last (ADR 0016). It must be missing or empty
            when training starts and must not overlap ``output_dir``. ``None`` writes nothing.

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
    unfreeze_last_blocks: int | None = None
    checkpoint_dir: str | None = None

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
        for name in ("unfreeze_last_blocks", "checkpoint_dir"):
            _require(
                self.training_mode != "frozen" or getattr(self, name) is None,
                f"{name} only applies to modes that train the backbone, not 'frozen'.",
            )
        _require(
            self.checkpoint_dir is None or not _paths_overlap(self.checkpoint_dir, self.output_dir),
            "checkpoint_dir and output_dir must not be the same directory or contain each "
            "other: saving the model replaces output_dir as a whole.",
        )


def _recorded_arguments(args: TrainingArguments) -> dict[str, Any]:
    """The arguments that saved models and checkpoints record: all but the local paths.

    A local path can reveal a user name.
    """
    return {
        name: value
        for name, value in asdict(args).items()
        if name not in ("output_dir", "checkpoint_dir")
    }
