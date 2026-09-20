"""Bounded positive/negative pair sampling for pairwise contrastive training."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from embedfit.data.dataset import ImageDataset, ImageSample
from embedfit.data.labels import label_key
from embedfit.exceptions import ConfigurationError

_MINING_BLOCK = 512  # Anchors per similarity matrix product: [512, N] float64 at a time.
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
            ``"label"``) or a key of ``metadata``; a sample field takes precedence over a
            metadata key of the same name. Samples without a value never match, whatever the
            relation, and a field with no value on any sample is an error.
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


def hard_negative_quota(negatives_per_example: int, hard_negative_fraction: float) -> int:
    """Negatives per anchor that are mined: ``negatives * fraction`` rounded half up."""
    return int(negatives_per_example * hard_negative_fraction + 0.5)


def _most_similar(pool: np.ndarray, similarity: np.ndarray, count: int) -> list[int]:
    """The ``count`` pool members with the highest similarity, ties by dataset position.

    Everything tied with the last admitted value is kept too, so the order never depends on how
    a partial sort happens to break ties.
    """
    if count < len(pool):
        cutoff = np.partition(similarity, len(pool) - count)[len(pool) - count]
        keep = np.flatnonzero(similarity >= cutoff)
        pool, similarity = pool[keep], similarity[keep]
    ranked: list[int] = pool[np.lexsort((pool, -similarity))].tolist()
    return ranked


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
    produced at most once per epoch: a partner already paired with the anchor is excluded, so
    an anchor can end an epoch below its quota and the pair count varies slightly by epoch.

    Memory is ``O(N)``; time is ``O(N^2)`` per epoch in vectorised array operations, because
    each anchor's pools are computed over all samples (a few seconds at 10 000 samples).
    Hard-negative mining adds an ``N x N`` similarity computation per epoch, done in blocks
    of 512 anchors (``O(N^2 D)`` time, ``O(512 N)`` extra memory).

    Quotas are per anchor, so the split between positive and negative pairs is only roughly
    even: a small class runs out of positive partners, and its samples are drawn often as
    negatives for the larger classes.

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
            no positive or no negative pair can be formed at all, or a rule's field has no value
            on any sample.
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
        rules = tuple(rules)  # Materialise first: a generator must not be consumed by validation.
        if any(not isinstance(rule, RelationshipRule) for rule in rules):
            raise ConfigurationError("rules must contain RelationshipRule objects.")
        if len(dataset) < 2:
            raise ConfigurationError(f"Pair sampling needs at least 2 samples, got {len(dataset)}.")

        self.positives_per_example = int(positives_per_example)
        self.negatives_per_example = int(negatives_per_example)
        self.seed = int(seed)
        self.rules = rules
        self._classes = _codes(dataset.labels, "label")
        self._rule_codes = [
            _codes([_field_value(sample, rule.field) for sample in dataset], rule.field)
            for rule in self.rules
        ]
        for rule, codes in zip(self.rules, self._rule_codes, strict=True):
            if (codes < 0).all():
                raise ConfigurationError(
                    f"Relationship rule field '{rule.field}' has no value on any sample, so the "
                    "rule can never match; check the field name."
                )
        pools = [self._pools(anchor) for anchor in range(len(dataset))]
        if self.positives_per_example and not any(len(positive) for positive, _, _ in pools):
            raise ConfigurationError(
                "No positive pair can be formed: after applying the class labels and the "
                "relationship rules, no sample has a positive partner."
            )
        if self.negatives_per_example and not any(len(negative) for _, negative, _ in pools):
            raise ConfigurationError(
                "No negative pair can be formed: after applying the class labels and the "
                "relationship rules, no sample has a negative partner."
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

    def sample(
        self,
        epoch: int = 0,
        *,
        embeddings: np.ndarray | None = None,
        hard_negative_fraction: float = 0.0,
    ) -> list[ImagePair]:
        """Return the pairs for one epoch, in a deterministic shuffled order.

        Args:
            epoch: Epoch number; with the seed it determines every random choice.
            embeddings: Current ``[N, D]`` embeddings of this sampler's own samples, in dataset
                order. Required for hard-negative mining. Only training data can be mined,
                because the sampler has no other samples to look at.
            hard_negative_fraction: Share of each anchor's negative quota, in [0, 1] and
                rounded half up to a whole number of negatives, filled
                with its *hardest* negatives: the members of its negative pool with the
                highest cosine similarity to it (ties by dataset position). The rest of the
                quota stays random, which keeps training stable and the negatives varied.
                Positives are always random.

        Raises:
            ConfigurationError: If the fraction is outside [0, 1], is positive but rounds to
                zero mined negatives, or mining is requested without valid embeddings.
        """
        valid = isinstance(hard_negative_fraction, int | float) and not isinstance(
            hard_negative_fraction, bool
        )
        if not valid or not 0 <= hard_negative_fraction <= 1:
            raise ConfigurationError(
                f"hard_negative_fraction must be in [0, 1], got {hard_negative_fraction!r}."
            )
        unit = None
        if hard_negative_fraction > 0:
            matrix = np.asarray(embeddings) if embeddings is not None else None
            if (
                matrix is None
                or matrix.ndim != 2
                or len(matrix) != len(self._classes)
                or not np.issubdtype(matrix.dtype, np.floating)
                or not np.isfinite(matrix).all()
            ):
                raise ConfigurationError(
                    "Hard-negative mining needs finite float embeddings shaped "
                    f"[{len(self._classes)}, D], one row per sample of this sampler's dataset."
                )
            norms = np.linalg.norm(matrix.astype(np.float64), axis=1, keepdims=True)
            unit = matrix / np.where(norms == 0, 1.0, norms)
        hard_quota = hard_negative_quota(self.negatives_per_example, hard_negative_fraction)
        if hard_negative_fraction > 0 and hard_quota == 0:
            raise ConfigurationError(
                f"hard_negative_fraction={hard_negative_fraction} of "
                f"negatives_per_example={self.negatives_per_example} rounds to no mined negative "
                "at all; raise one of them or set the fraction to 0."
            )
        rng = np.random.default_rng([self.seed, int(epoch)])
        seen: set[tuple[int, int]] = set()
        paired = np.zeros(len(self._classes), dtype=np.int64)  # Pairs each sample is already in.
        pairs: list[ImagePair] = []
        block: np.ndarray | None = None
        for anchor in range(len(self._classes)):
            positive_pool, negative_pool, decided_by = self._pools(anchor)
            if unit is not None and anchor % _MINING_BLOCK == 0:
                # One matrix product per block of anchors instead of one gather per anchor.
                block = unit[anchor : anchor + _MINING_BLOCK] @ unit.T
            for pool, quota, target in (
                (positive_pool, self.positives_per_example, 1),
                (negative_pool, self.negatives_per_example, 0),
            ):
                # Partners already paired with this anchor earlier in the epoch are excluded, so
                # an anchor can end up under its quota. Stop as soon as the quota is met: only
                # a few partners are needed, however large the pool is.
                candidates = rng.permutation(pool).tolist()
                hardest: list[int] = []
                if block is not None and target == 0 and len(pool):
                    # Up to `paired` of the top candidates may already be taken, so look that
                    # much further down the ranking.
                    wanted = hard_quota + int(paired[anchor])
                    hardest = _most_similar(pool, block[anchor % _MINING_BLOCK, pool], wanted)
                fresh: list[int] = []
                for ordering, limit in ((hardest, min(hard_quota, quota)), (candidates, quota)):
                    for candidate in ordering:
                        if len(fresh) >= limit:
                            break
                        key = (anchor, candidate) if anchor < candidate else (candidate, anchor)
                        if key not in seen and candidate not in fresh:
                            fresh.append(candidate)
                for partner in fresh:
                    seen.add((min(anchor, partner), max(anchor, partner)))
                    paired[anchor] += 1
                    paired[partner] += 1
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
