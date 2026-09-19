"""Bounded positive/negative pair sampling for pairwise contrastive training."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from embedfit.data.dataset import ImageDataset, ImageSample
from embedfit.data.labels import label_key
from embedfit.exceptions import ConfigurationError

_RELATIONS = ("same", "different")
_PAIR_TYPES = ("positive", "negative")
_SAMPLE_FIELDS = ("label", "sample_id", "group_id", "timestamp", "amount")


@dataclass(frozen=True)
class ImagePair:
    """Two dataset positions and whether their images should embed close together.

    Attributes:
        left: Position of the first image in the dataset.
        right: Position of the second image.
        target: 1 for a similar (positive) pair, 0 for a dissimilar (negative) pair.
        weight: Relative weight of the pair in the loss.
        relationship: Name of the ``RelationshipRule`` that decided ``target``, or ``None``
            when the class labels did.
    """

    left: int
    right: int
    target: int
    weight: float = 1.0
    relationship: str | None = None


@dataclass(frozen=True)
class RelationshipRule:
    """Declare pairs similar or dissimilar from a field other than the class label.

    The library attaches no meaning to the field: a rule only states that two samples whose
    values are the same (or different) form a positive (or negative) pair.

    Attributes:
        field: A sample field (``"group_id"``, ``"sample_id"``, ``"timestamp"``, ``"amount"``,
            ``"label"``) or a key of ``metadata``. Samples without a value never match.
        relation: ``"same"`` or ``"different"``.
        pair_type: ``"positive"`` or ``"negative"``: the target given to matching pairs,
            overriding the class labels.
        weight: Loss weight of matching pairs (> 0).

    Raises:
        ConfigurationError: If an argument is invalid.
    """

    field: str
    relation: str = "same"
    pair_type: str = "positive"
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.field, str) or not self.field:
            raise ConfigurationError("RelationshipRule.field must be a non-empty str.")
        if self.relation not in _RELATIONS:
            raise ConfigurationError(f"relation must be one of {list(_RELATIONS)}.")
        if self.pair_type not in _PAIR_TYPES:
            raise ConfigurationError(f"pair_type must be one of {list(_PAIR_TYPES)}.")
        valid = isinstance(self.weight, int | float) and not isinstance(self.weight, bool)
        if not valid or not self.weight > 0:
            raise ConfigurationError(f"weight must be a number > 0, got {self.weight!r}.")

    @property
    def name(self) -> str:
        """Identifier recorded on the pairs this rule decides, e.g. ``"same:group_id"``."""
        return f"{self.relation}:{self.field}"


def _field_value(sample: ImageSample, name: str) -> Any:
    if name in _SAMPLE_FIELDS:
        return getattr(sample, name)
    return sample.metadata.get(name) if sample.metadata is not None else None


def _codes(values: list[Any], what: str) -> np.ndarray:
    """Map values to integer codes (equal value and type → equal code); ``None`` → -1."""
    table: dict[Any, int] = {}
    codes = np.full(len(values), -1, dtype=np.int64)
    try:
        for position, value in enumerate(values):
            if value is not None:
                codes[position] = table.setdefault(label_key(value), len(table))
    except TypeError as exc:
        raise ConfigurationError(f"{what} values must be hashable.") from exc
    return codes


class PairSampler:
    """Draw a bounded, balanced set of positive and negative pairs each epoch.

    For every sample (the anchor), up to ``positives_per_example`` partners are drawn from its
    positive pool and up to ``negatives_per_example`` from its negative pool, without
    replacement. By default the positive pool is the other samples of the anchor's class and
    the negative pool is every sample of another class. The number of pairs is therefore at
    most ``N * (positives_per_example + negatives_per_example)``; all ``N²`` pairs are never
    materialised. A pair is never formed between a sample and itself, and an unordered pair is
    produced at most once per epoch.

    ``rules`` change pool membership, in the order given: the first rule that matches a pair
    decides its target and weight, and only pairs matched by no rule fall back to the class
    labels. Without rules, positive pairs always share a class and negative pairs never do.

    Args:
        dataset: The training samples.
        positives_per_example: Positive partners drawn per anchor (>= 0).
        negatives_per_example: Negative partners drawn per anchor (>= 0). At least one of the
            two must be positive.
        seed: Base seed. Pairs depend only on ``seed`` and the ``epoch`` passed to ``sample``.
        rules: Optional relationship rules.

    Raises:
        ConfigurationError: If an argument is invalid, the dataset has fewer than 2 samples, or
            no positive or no negative pair can be formed at all.
    """

    def __init__(
        self,
        dataset: ImageDataset,
        *,
        positives_per_example: int = 4,
        negatives_per_example: int = 4,
        seed: int,
        rules: Sequence[RelationshipRule] = (),
    ) -> None:
        if not isinstance(dataset, ImageDataset):
            raise ConfigurationError(
                f"dataset must be an ImageDataset, got {type(dataset).__name__}."
            )
        for name, value in (
            ("positives_per_example", positives_per_example),
            ("negatives_per_example", negatives_per_example),
            ("seed", seed),
        ):
            if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 0:
                raise ConfigurationError(f"{name} must be a non-negative int, got {value!r}.")
        if positives_per_example + negatives_per_example == 0:
            raise ConfigurationError(
                "positives_per_example and negatives_per_example cannot both be 0."
            )
        if any(not isinstance(rule, RelationshipRule) for rule in rules):
            raise ConfigurationError("rules must contain RelationshipRule objects.")
        if len(dataset) < 2:
            raise ConfigurationError(f"Pair sampling needs at least 2 samples, got {len(dataset)}.")

        self.positives_per_example = int(positives_per_example)
        self.negatives_per_example = int(negatives_per_example)
        self.seed = int(seed)
        self.rules = tuple(rules)
        self._classes = _codes(dataset.labels, "label")
        self._rule_codes = [
            _codes([_field_value(sample, rule.field) for sample in dataset], rule.field)
            for rule in self.rules
        ]
        pools = [self._pools(anchor) for anchor in range(len(dataset))]
        if self.positives_per_example and not any(len(positive) for positive, _, _ in pools):
            raise ConfigurationError(
                "No positive pair can be formed: no two samples share a class or match a "
                "positive relationship rule."
            )
        if self.negatives_per_example and not any(len(negative) for _, negative, _ in pools):
            raise ConfigurationError(
                "No negative pair can be formed: every sample has the same class and no "
                "negative relationship rule matches."
            )

    def __len__(self) -> int:
        """Upper bound on the number of pairs per epoch."""
        return len(self._classes) * (self.positives_per_example + self.negatives_per_example)

    def _pools(self, anchor: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the anchor's positive pool, negative pool, and the deciding rule per sample.

        The rule array holds, for every sample, the index of the first rule that decides its
        pairing with the anchor, or -1 when the class labels decide.
        """
        count = len(self._classes)
        positive: np.ndarray = self._classes == self._classes[anchor]
        decided_by: np.ndarray = np.full(count, -1, dtype=np.int64)
        for index in reversed(range(len(self.rules))):  # Earlier rules overwrite later ones.
            rule, codes = self.rules[index], self._rule_codes[index]
            if codes[anchor] < 0:
                continue
            same = codes == codes[anchor]
            matches = (same if rule.relation == "same" else ~same) & (codes >= 0)
            positive = np.where(matches, rule.pair_type == "positive", positive)
            decided_by = np.where(matches, index, decided_by)
        others = np.arange(count) != anchor
        return (
            np.flatnonzero(positive & others),
            np.flatnonzero(~positive & others),
            decided_by,
        )

    def sample(self, epoch: int = 0) -> list[ImagePair]:
        """Return the pairs for one epoch, in a deterministic shuffled order."""
        rng = np.random.default_rng([self.seed, int(epoch)])
        seen: set[tuple[int, int]] = set()
        pairs: list[ImagePair] = []
        for anchor in range(len(self._classes)):
            positive_pool, negative_pool, decided_by = self._pools(anchor)
            for pool, quota, target in (
                (positive_pool, self.positives_per_example, 1),
                (negative_pool, self.negatives_per_example, 0),
            ):
                # Partners already paired with this anchor (as its anchor) are skipped first,
                # so the quota is spent on new pairs where possible.
                fresh = [
                    p
                    for p in rng.permutation(pool).tolist()
                    if (min(anchor, p), max(anchor, p)) not in seen
                ]
                for partner in fresh[:quota]:
                    seen.add((min(anchor, partner), max(anchor, partner)))
                    rule_index = int(decided_by[partner])
                    rule = self.rules[rule_index] if rule_index >= 0 else None
                    pairs.append(
                        ImagePair(
                            left=anchor,
                            right=partner,
                            target=target,
                            weight=float(rule.weight) if rule else 1.0,
                            relationship=rule.name if rule else None,
                        )
                    )
        order = rng.permutation(len(pairs)).tolist()
        return [pairs[position] for position in order]
