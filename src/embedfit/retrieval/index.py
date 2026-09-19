"""``RetrievalIndex``: identifiers, metadata, groups, and exclusions over a search backend."""

import logging
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np

from embedfit.exceptions import ConfigurationError, IndexNotFittedError
from embedfit.retrieval.base import RetrievalBackend, RetrievalResult, validate_embeddings
from embedfit.retrieval.brute_force import BruteForceCosineBackend

logger = logging.getLogger(__name__)

_BACKENDS: dict[str, Callable[[], RetrievalBackend]] = {"cosine": BruteForceCosineBackend}
"""Backend factory per metric. A future FAISS or pgvector backend is added here."""


def _plain(value: Any) -> Any:
    """Unwrap numpy scalars so ``np.str_("a")`` and ``"a"`` are the same identifier."""
    return value.item() if isinstance(value, np.generic) else value


def _as_sequence(values: Any, *, name: str) -> list[Any]:
    """Return ``values`` as a list of plain Python values.

    Raises:
        ConfigurationError: If ``values`` is not a sequence or 1-D numpy array. A bare string is
            rejected because iterating it would silently yield one entry per character.
    """
    if isinstance(values, np.ndarray) and values.ndim == 1:
        values = values.tolist()
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise ConfigurationError(
            f"{name} must be a sequence with one entry per item, e.g. a list or 1-D numpy array."
        )
    return [_plain(value) for value in values]


def _as_collection(values: Any, *, name: str) -> list[Any]:
    """Return ``values`` as a list of plain Python values.

    Raises:
        ConfigurationError: If ``values`` is not iterable, or is a bare string, which would
            otherwise be read as one entry per character.
    """
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if isinstance(values, str) or not isinstance(values, Iterable):
        raise ConfigurationError(
            f"{name} must be a collection of ids, e.g. a list or set; a bare string is not one."
        )
    return [_plain(value) for value in values]


def _validate_ids(ids: Any, count: int) -> tuple[tuple[Hashable, ...], dict[Hashable, int]]:
    """Return the identifiers in index order and the position of each one."""
    values: tuple[Hashable, ...] = (
        tuple(range(count)) if ids is None else tuple(_as_sequence(ids, name="ids"))
    )
    if len(values) != count:
        raise ConfigurationError(
            f"ids must have one entry per embedding: got {len(values)} ids for {count} embeddings."
        )
    try:
        positions: dict[Hashable, int] = {value: position for position, value in enumerate(values)}
    except TypeError as exc:
        raise ConfigurationError("ids must be hashable values such as str or int.") from exc
    if len(positions) != count:
        raise ConfigurationError(
            "ids must be unique; values that compare equal, such as 0 and False, are one id."
        )
    return values, positions


def _validate_metadata(metadata: Any, count: int) -> tuple[Mapping[str, Any] | None, ...] | None:
    """Return one read-only mapping (or ``None``) per item, or ``None`` if none was supplied."""
    if metadata is None:
        return None
    if isinstance(metadata, Mapping) or not isinstance(metadata, Sequence):
        raise ConfigurationError(
            "metadata must be a sequence with one mapping or None per embedding, "
            "not a single mapping."
        )
    if len(metadata) != count:
        raise ConfigurationError(
            f"metadata must have one entry per embedding: got {len(metadata)} entries "
            f"for {count} embeddings."
        )
    entries: list[Mapping[str, Any] | None] = []
    for position, entry in enumerate(metadata):
        if entry is None:
            entries.append(None)
        elif isinstance(entry, Mapping):
            entries.append(MappingProxyType(dict(entry)))
        else:
            raise ConfigurationError(
                f"metadata[{position}] must be a mapping or None, got {type(entry).__name__}."
            )
    return tuple(entries)


def _validate_groups(groups: Any, count: int) -> dict[Hashable, np.ndarray] | None:
    """Return the item positions belonging to each group, or ``None`` if no groups were given."""
    if groups is None:
        return None
    values = _as_sequence(groups, name="groups")
    if len(values) != count:
        raise ConfigurationError(
            f"groups must have one entry per embedding: got {len(values)} entries "
            f"for {count} embeddings."
        )
    members: dict[Hashable, list[int]] = {}
    for position, group in enumerate(values):
        if group is None:
            continue
        try:
            members.setdefault(group, []).append(position)
        except TypeError as exc:
            raise ConfigurationError(
                "group ids must be hashable values such as str or int, or None."
            ) from exc
    return {group: np.array(positions, dtype=np.int64) for group, positions in members.items()}


class RetrievalIndex:
    """Nearest-neighbour search over stored embeddings, with leakage-safe exclusions.

    The index owns identifiers, metadata, and group membership; the similarity search itself is
    delegated to a ``RetrievalBackend`` chosen by ``metric``. It works on embeddings alone and
    never calls a model (``ARCHITECTURE.md``).

    Args:
        metric: Similarity to rank by. Only ``"cosine"`` exists in v0.1.

    Raises:
        ConfigurationError: If ``metric`` is not a supported metric.
    """

    def __init__(self, metric: str = "cosine") -> None:
        if metric not in _BACKENDS:
            raise ConfigurationError(
                f"Unknown metric {metric!r}. Valid metrics: {sorted(_BACKENDS)}."
            )
        self._metric = metric
        self._backend: RetrievalBackend | None = None
        self._ids: tuple[Hashable, ...] = ()
        self._positions: dict[Hashable, int] = {}
        self._metadata: tuple[Mapping[str, Any] | None, ...] | None = None
        self._group_members: dict[Hashable, np.ndarray] | None = None

    def __repr__(self) -> str:
        return (
            f"RetrievalIndex(metric={self._metric!r}, size={self.size}, "
            f"embedding_dim={self.embedding_dim})"
        )

    @property
    def metric(self) -> str:
        """The similarity the index ranks by."""
        return self._metric

    @property
    def is_fitted(self) -> bool:
        """Whether ``fit`` has completed."""
        return self._backend is not None

    @property
    def size(self) -> int:
        """Number of indexed items; ``0`` before ``fit``."""
        return 0 if self._backend is None else self._backend.size

    @property
    def embedding_dim(self) -> int | None:
        """Width of the indexed embeddings, or ``None`` before ``fit``."""
        return None if self._backend is None else self._backend.embedding_dim

    @property
    def ids(self) -> tuple[Hashable, ...]:
        """The identifiers in index order; empty before ``fit``."""
        return self._ids

    def fit(
        self,
        embeddings: np.ndarray,
        ids: Sequence[Hashable] | np.ndarray | None = None,
        metadata: Sequence[Mapping[str, Any] | None] | None = None,
        groups: Sequence[Hashable | None] | np.ndarray | None = None,
    ) -> None:
        """Index ``embeddings``, replacing anything indexed before.

        Args:
            embeddings: Finite float matrix ``[N, D]`` with ``N >= 1``, one row per item. Rows
                keep their input order, which is what equal scores are broken by. For cosine,
                no row may be the zero vector.
            ids: One unique hashable identifier per row, as a sequence or 1-D numpy array;
                numpy scalars are unwrapped to Python values. Defaults to ``0..N-1``. Values
                that compare equal, such as ``0`` and ``False``, count as one identifier.
            metadata: One mapping, or ``None``, per row. Each mapping is copied into a read-only
                view and is returned unchanged with its item; nothing is inspected, logged, or
                used for ranking.
            groups: One hashable group identifier, or ``None``, per row, naming items that come
                from the same underlying thing (the same physical document, customer, or
                session). ``None`` means "belongs to no group" and never matches another item.
                Required by ``search(..., exclude_same_group=True)``.

        Raises:
            ConfigurationError: If ``embeddings`` is not a finite 2-D float array with at least
                one row, contains a zero-norm row, or ``ids``, ``metadata``, or ``groups`` has
                the wrong length, type, or duplicate identifiers. Messages describe structure
                only, never embedding or metadata values.
        """
        matrix = validate_embeddings(embeddings, name="embeddings")
        count = len(matrix)
        if count == 0:
            raise ConfigurationError("embeddings must contain at least one row, got 0.")
        item_ids, positions = _validate_ids(ids, count)
        item_metadata = _validate_metadata(metadata, count)
        group_members = _validate_groups(groups, count)
        backend = _BACKENDS[self._metric]()
        backend.fit(matrix)  # Raises before any previously indexed state is discarded.
        self._backend = backend
        self._ids = item_ids
        self._positions = positions
        self._metadata = item_metadata
        self._group_members = group_members
        logger.debug(
            "Indexed %d embeddings of dimension %d for %s retrieval",
            count,
            matrix.shape[1],
            self._metric,
        )

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        *,
        query_ids: Hashable | Sequence[Hashable] | None = None,
        exclude_self: bool = False,
        exclude_ids: Iterable[Hashable] | None = None,
        query_groups: Hashable | Sequence[Hashable] | None = None,
        exclude_same_group: bool = False,
        exclude_groups: Iterable[Hashable] | None = None,
    ) -> list[RetrievalResult] | list[list[RetrievalResult]]:
        """Return the most similar indexed items for one query or a batch of queries.

        Every exclusion is applied before the top ``k`` is cut, so ``k`` results are still
        returned whenever ``k`` items remain after exclusion. Fewer than ``k`` results are
        returned when the index, minus the exclusions, holds fewer than ``k`` items; the result
        list is then shorter rather than padded.

        Args:
            query: One embedding shaped ``[D]``, or a batch shaped ``[Q, D]``. The shape alone
                decides which, and it alone decides the return type.
            k: Number of neighbours per query; a positive int.
            query_ids: The identifier of the query item: one value for a single query, or one
                per row for a batch. Required by, and only used with, ``exclude_self``.
            exclude_self: Drop the indexed item whose id equals the query's id, for evaluating
                indexed items without scoring them against themselves. Only that one item is
                dropped: a genuine duplicate of the query stored under a different id is still
                returned, and a ``query_ids`` value that is not in the index drops nothing.
            exclude_ids: Identifiers that must never appear in the results. For a batch, the one
                collection applies to every query. Identifiers absent from the index are ignored.
            query_groups: The group of the query item: one value for a single query, or one per
                row for a batch. Required by, and only used with, ``exclude_same_group``. A
                ``None`` entry in a batch is a query that belongs to no group and therefore
                excludes nothing; a single query has no way to say that, because
                ``query_groups=None`` means the argument was omitted, so a group-less single
                query simply does not ask for ``exclude_same_group``.
            exclude_same_group: Drop every indexed item whose group equals the query's group.
                This is the leakage-safe mode: near-duplicates of the same underlying item
                cannot be counted as evidence that the representation generalises
                (``AGENTS.md`` rule 5). Requires groups at fit time.
            exclude_groups: Group identifiers whose items must never appear in the results, for
                a query whose forbidden groups are not its own. Requires groups at fit time.

        Returns:
            For a ``[D]`` query, a ``list[RetrievalResult]`` ordered by descending score with
            ``rank`` counting from 1. For a ``[Q, D]`` batch, one such list per query row, in
            row order; a batch with zero rows returns an empty list. Equal scores are ordered by
            ascending index position, so results never depend on chance. A batched matrix
            product accumulates in a different order from a single-row one, so a query's scores
            can differ in the last float32 digits between the two, and items whose scores differ
            by less than that error can swap.

        Raises:
            IndexNotFittedError: If the index has not been fitted.
            ConfigurationError: If ``query`` is not a finite 1-D or 2-D float array whose width
                matches the index, ``k`` is not a positive int, an exclusion option is given
                without the data it needs, or the per-query values have the wrong length.
        """
        backend = self._require_fitted()
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ConfigurationError(f"k must be a positive int, got {k!r}.")
        queries, is_batch = self._as_batch(query)
        count = len(queries)
        per_query_ids = self._per_query(
            query_ids,
            name="query_ids",
            flag="exclude_self",
            enabled=exclude_self,
            count=count,
            is_batch=is_batch,
        )
        per_query_groups = self._per_query(
            query_groups,
            name="query_groups",
            flag="exclude_same_group",
            enabled=exclude_same_group,
            count=count,
            is_batch=is_batch,
        )
        if exclude_same_group and self._group_members is None:
            raise ConfigurationError(
                "exclude_same_group=True requires groups to be supplied to fit."
            )
        blocked = self._blocked_positions(exclude_ids, exclude_groups)
        if count == 0:
            return []

        masks = np.repeat(blocked[None, :], count, axis=0)
        group_members = self._group_members or {}
        for row in range(count):
            if per_query_ids is not None:
                position = self._lookup(per_query_ids[row], name="query_ids")
                if position is not None:
                    masks[row, position] = True
            if per_query_groups is not None and per_query_groups[row] is not None:
                members = group_members.get(
                    self._hashable(per_query_groups[row], name="query_groups")
                )
                if members is not None:
                    masks[row, members] = True

        # Asking for k plus the largest number of exclusions guarantees that k allowed items
        # survive the filter whenever the index holds k allowed items, because the backend's
        # ordering is total: its top m is always a prefix of its top m + 1.
        fetch = min(self.size, k + int(masks.sum(axis=1).max()))
        positions, scores = backend.search(queries, fetch)
        results = [
            self._collect(positions[row], scores[row], allowed=~masks[row], k=k)
            for row in range(count)
        ]
        return results if is_batch else results[0]

    def _collect(
        self, positions: np.ndarray, scores: np.ndarray, *, allowed: np.ndarray, k: int
    ) -> list[RetrievalResult]:
        """Turn backend output for one query into at most ``k`` results, skipping exclusions."""
        hits: list[RetrievalResult] = []
        for position, score in zip(positions, scores, strict=True):
            if not allowed[position]:
                continue
            hits.append(
                RetrievalResult(
                    id=self._ids[position],
                    score=float(score),
                    rank=len(hits) + 1,
                    metadata=None if self._metadata is None else self._metadata[position],
                )
            )
            if len(hits) == k:
                break
        return hits

    def _require_fitted(self) -> RetrievalBackend:
        if self._backend is None:
            raise IndexNotFittedError(
                "The index has not been fitted; call fit(embeddings) before search."
            )
        return self._backend

    def _as_batch(self, query: np.ndarray) -> tuple[np.ndarray, bool]:
        """Return ``(queries [Q, D], is_batch)``; the rank of ``query`` alone decides."""
        array = np.asarray(query)
        if array.ndim not in (1, 2):
            raise ConfigurationError(
                "query must be one embedding shaped [D] or a batch shaped [Q, D]; "
                f"got a {array.ndim}-D array of shape {array.shape}."
            )
        is_batch = array.ndim == 2
        matrix = validate_embeddings(
            array if is_batch else array[None, :], name="query", expected_dim=self.embedding_dim
        )
        return matrix, is_batch

    @staticmethod
    def _per_query(
        values: Any, *, name: str, flag: str, enabled: bool, count: int, is_batch: bool
    ) -> list[Any] | None:
        """Return one value per query row, or ``None`` when the exclusion is off."""
        if not enabled:
            if values is not None:
                raise ConfigurationError(
                    f"{name} is only used when {flag}=True; pass {flag}=True or drop {name}."
                )
            return None
        if values is None:
            raise ConfigurationError(
                f"{flag}=True requires {name}: one value for a single query, or one value per "
                "row for a batch."
            )
        if not is_batch:
            return [_plain(values)]
        per_row = _as_sequence(values, name=name)
        if len(per_row) != count:
            raise ConfigurationError(
                f"{name} must have one entry per query: got {len(per_row)} entries "
                f"for {count} queries."
            )
        return per_row

    @staticmethod
    def _hashable(value: Any, *, name: str) -> Hashable:
        try:
            hash(value)
        except TypeError as exc:
            raise ConfigurationError(
                f"{name} must contain hashable values such as str or int."
            ) from exc
        hashable: Hashable = value
        return hashable

    def _lookup(self, value: Any, *, name: str) -> int | None:
        """Return the index position of one identifier, or ``None`` if it is not indexed."""
        return self._positions.get(self._hashable(value, name=name))

    def _blocked_positions(
        self, exclude_ids: Iterable[Hashable] | None, exclude_groups: Iterable[Hashable] | None
    ) -> np.ndarray:
        """Return the ``[N]`` boolean mask of items excluded for every query."""
        blocked = np.zeros(self.size, dtype=bool)
        if exclude_ids is not None:
            for value in _as_collection(exclude_ids, name="exclude_ids"):
                position = self._lookup(value, name="exclude_ids")
                if position is not None:
                    blocked[position] = True
        if exclude_groups is not None:
            if self._group_members is None:
                raise ConfigurationError("exclude_groups requires groups to be supplied to fit.")
            for value in _as_collection(exclude_groups, name="exclude_groups"):
                members = self._group_members.get(self._hashable(value, name="exclude_groups"))
                if members is not None:
                    blocked[members] = True
        return blocked
