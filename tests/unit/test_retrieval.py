import dataclasses

import numpy as np
import pytest

from embedfit.exceptions import ConfigurationError, IndexNotFittedError
from embedfit.retrieval import (
    BruteForceCosineBackend,
    RetrievalIndex,
    RetrievalResult,
)

# Hand-checkable geometry: cosines against [1, 0] are 1, 0, 1/sqrt(2), -1.
VECTORS = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
IDS = ["a", "b", "c", "d"]
RIGHT = np.array([1.0, 0.0], dtype=np.float32)
# Cosines against [1, 0] fall strictly: 1.0, 0.9939, 0.9701, 0.9191, 0.8321.
LADDER = np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]], dtype=np.float32)


@pytest.fixture
def index() -> RetrievalIndex:
    built = RetrievalIndex(metric="cosine")
    built.fit(VECTORS, ids=IDS)
    return built


@pytest.fixture
def ladder() -> RetrievalIndex:
    built = RetrievalIndex()
    built.fit(LADDER)
    return built


def test_hand_computed_cosine_scores_and_order(index):
    results = index.search(RIGHT, k=4)
    assert [result.id for result in results] == ["a", "c", "b", "d"]
    assert [result.rank for result in results] == [1, 2, 3, 4]
    assert np.allclose([result.score for result in results], [1.0, 2**-0.5, 0.0, -1.0], atol=1e-6)


def test_scores_are_independent_of_embedding_scale(index):
    scaled = RetrievalIndex()
    scaled.fit(VECTORS * np.array([[0.5], [100.0], [3.0], [0.02]], dtype=np.float32))
    baseline = index.search(RIGHT, k=4)
    rescaled = scaled.search(RIGHT * 17.0, k=4)
    assert [IDS[result.id] for result in rescaled] == [result.id for result in baseline]
    assert np.allclose(
        [result.score for result in rescaled], [result.score for result in baseline], atol=1e-6
    )


def test_every_item_is_its_own_rank_one_match(index):
    for position, identifier in enumerate(IDS):
        results = index.search(VECTORS[position], k=4)
        assert results[0].id == identifier
        assert results[0].score == pytest.approx(1.0, abs=1e-6)
        scores = [result.score for result in results]
        assert scores == sorted(scores, reverse=True)


def test_default_ids_are_index_positions():
    built = RetrievalIndex()
    built.fit(VECTORS)
    assert built.ids == (0, 1, 2, 3)
    assert [result.id for result in built.search(RIGHT, k=2)] == [0, 2]


def test_index_state_before_and_after_fit(index):
    empty = RetrievalIndex()
    assert (empty.is_fitted, empty.size, empty.embedding_dim, empty.ids) == (False, 0, None, ())
    assert (index.is_fitted, index.size, index.embedding_dim, index.metric) == (
        True,
        4,
        2,
        "cosine",
    )


def test_refitting_replaces_ids_metadata_and_vectors(index):
    index.fit(VECTORS[:2] * -1.0, ids=["x", "y"])
    assert index.size == 2 and index.ids == ("x", "y")
    results = index.search(np.array([-1.0, 0.0], dtype=np.float32), k=5)
    assert [result.id for result in results] == ["x", "y"]
    assert all(result.metadata is None for result in results)


def test_fewer_than_k_results_when_the_index_is_smaller(index):
    results = index.search(RIGHT, k=99)
    assert len(results) == 4
    assert [result.rank for result in results] == [1, 2, 3, 4]


def test_ties_are_ordered_by_insertion_position():
    identical = RetrievalIndex()
    identical.fit(np.repeat(RIGHT[None, :], 6, axis=0))
    # The tie group straddles the top-k cut, which is exactly where an argpartition would
    # otherwise be free to return any two of the six.
    assert [result.id for result in identical.search(RIGHT, k=2)] == [0, 1]
    assert [result.id for result in identical.search(RIGHT, k=4)] == [0, 1, 2, 3]


def test_ties_below_a_strictly_better_hit_keep_insertion_order():
    mixed = RetrievalIndex()
    mixed.fit(np.array([[1.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 1.0]], dtype=np.float32))
    assert [result.id for result in mixed.search(RIGHT, k=3)] == [1, 0, 2]


def test_metadata_is_returned_with_the_matching_id():
    built = RetrievalIndex()
    source = [{"owner": identifier} for identifier in IDS]
    built.fit(VECTORS, ids=IDS, metadata=source)
    for result in built.search(RIGHT, k=4):
        assert result.metadata is not None and result.metadata["owner"] == result.id


def test_metadata_is_stored_as_a_read_only_copy():
    built = RetrievalIndex()
    source = {"owner": "a"}
    built.fit(VECTORS[:2], ids=["a", "b"], metadata=[source, None])
    source["owner"] = "changed"
    result = built.search(RIGHT, k=1)[0]
    assert result.metadata is not None and result.metadata["owner"] == "a"
    with pytest.raises(TypeError):
        result.metadata["owner"] = "again"


def test_missing_metadata_entry_is_none():
    built = RetrievalIndex()
    built.fit(VECTORS[:2], ids=["a", "b"], metadata=[None, {"owner": "b"}])
    assert built.search(RIGHT, k=2)[0].metadata is None


def test_exclude_self_removes_the_query_item_but_keeps_its_duplicates():
    built = RetrievalIndex()
    built.fit(np.concatenate([VECTORS, RIGHT[None, :]]), ids=[*IDS, "a_twin"])
    results = built.search(RIGHT, k=4, exclude_self=True, query_ids="a")
    assert [result.id for result in results] == ["a_twin", "c", "b", "d"]
    assert results[0].score == pytest.approx(1.0, abs=1e-6)
    assert [result.rank for result in results] == [1, 2, 3, 4]


def test_exclude_self_still_returns_k_results(index):
    assert len(index.search(RIGHT, k=3, exclude_self=True, query_ids="a")) == 3


def test_exclude_self_with_an_unindexed_query_id_removes_nothing(index):
    results = index.search(RIGHT, k=4, exclude_self=True, query_ids="not-indexed")
    assert [result.id for result in results] == ["a", "c", "b", "d"]


def test_exclude_self_requires_query_ids(index):
    with pytest.raises(ConfigurationError, match="exclude_self=True requires query_ids"):
        index.search(RIGHT, exclude_self=True)


def test_query_ids_without_exclude_self_is_rejected(index):
    with pytest.raises(ConfigurationError, match="query_ids is only used when exclude_self=True"):
        index.search(RIGHT, query_ids="a")


def test_exclusions_are_applied_before_the_top_k(ladder):
    assert [result.id for result in ladder.search(RIGHT, k=3)] == [0, 1, 2]
    excluded = ladder.search(RIGHT, k=3, exclude_ids=[0, 1])
    assert [result.id for result in excluded] == [2, 3, 4]
    assert [result.rank for result in excluded] == [1, 2, 3]


def test_exclude_ids_ignores_unknown_ids(ladder):
    results = ladder.search(RIGHT, k=2, exclude_ids={0, 99})
    assert [result.id for result in results] == [1, 2]


def test_exclude_ids_rejects_a_bare_string(index):
    with pytest.raises(ConfigurationError, match="exclude_ids must be a collection of ids"):
        index.search(RIGHT, exclude_ids="a")


def test_exclude_same_group_removes_every_item_of_the_query_group():
    built = RetrievalIndex()
    built.fit(LADDER, groups=["scan-1", "scan-1", "scan-2", "scan-2", None])
    results = built.search(RIGHT, k=3, exclude_same_group=True, query_groups="scan-1")
    assert [result.id for result in results] == [2, 3, 4]


def test_group_none_never_matches_another_item():
    built = RetrievalIndex()
    built.fit(LADDER, groups=["scan-1", "scan-1", None, None, None])
    batch = built.search(
        np.stack([RIGHT, RIGHT]),
        k=3,
        exclude_same_group=True,
        query_groups=[None, "scan-1"],
    )
    assert [result.id for result in batch[0]] == [0, 1, 2]
    assert [result.id for result in batch[1]] == [2, 3, 4]


def test_exclude_groups_removes_named_groups():
    built = RetrievalIndex()
    built.fit(LADDER, groups=["scan-1", "scan-2", "scan-1", None, None])
    assert [result.id for result in built.search(RIGHT, k=3, exclude_groups=["scan-1"])] == [
        1,
        3,
        4,
    ]


def test_group_exclusions_require_groups_at_fit_time(index):
    with pytest.raises(ConfigurationError, match="exclude_same_group=True requires groups"):
        index.search(RIGHT, exclude_same_group=True, query_groups="scan-1")
    with pytest.raises(ConfigurationError, match="exclude_groups requires groups"):
        index.search(RIGHT, exclude_groups=["scan-1"])


def test_query_groups_without_exclude_same_group_is_rejected(index):
    with pytest.raises(ConfigurationError, match="query_groups is only used"):
        index.search(RIGHT, query_groups="scan-1")


def test_batch_results_match_the_same_queries_run_singly():
    built = RetrievalIndex()
    built.fit(VECTORS, ids=IDS, metadata=[{"owner": name} for name in IDS])
    queries = np.stack([RIGHT, VECTORS[1], VECTORS[2]])
    batch = built.search(queries, k=3, exclude_self=True, query_ids=["a", "b", "c"])
    singly = [
        built.search(queries[row], k=3, exclude_self=True, query_ids=identifier)
        for row, identifier in enumerate(["a", "b", "c"])
    ]
    assert batch == singly
    assert len(batch) == 3 and all(len(results) == 3 for results in batch)


def test_a_single_query_and_a_one_row_batch_differ_only_in_nesting(index):
    assert index.search(np.stack([RIGHT]), k=2) == [index.search(RIGHT, k=2)]


def test_empty_batch_returns_no_result_lists(index):
    assert index.search(np.empty((0, 2), dtype=np.float32), k=3) == []


def test_search_before_fit():
    with pytest.raises(IndexNotFittedError, match="call fit"):
        RetrievalIndex().search(RIGHT)


def test_unknown_metric_lists_the_valid_ones():
    with pytest.raises(ConfigurationError, match=r"Unknown metric 'euclidean'.*\['cosine'\]"):
        RetrievalIndex(metric="euclidean")


def test_zero_norm_embeddings_are_rejected_at_fit():
    with pytest.raises(ConfigurationError, match=r"zero-norm row\(s\) at positions \[1\]"):
        RetrievalIndex().fit(np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32))


def test_zero_norm_query_is_rejected_at_search(index):
    with pytest.raises(ConfigurationError, match="cosine similarity is undefined"):
        index.search(np.zeros(2, dtype=np.float32))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"embeddings": np.zeros(4, dtype=np.float32)}, "2-D float array"),
        ({"embeddings": np.zeros((2, 2), dtype=np.int64)}, "2-D float array"),
        ({"embeddings": np.empty((0, 2), dtype=np.float32)}, "at least one row"),
        ({"embeddings": np.full((2, 2), np.inf, dtype=np.float32)}, "NaN or infinite"),
        ({"embeddings": VECTORS, "ids": ["a", "b"]}, "2 ids for 4 embeddings"),
        ({"embeddings": VECTORS, "ids": ["a", "b", "c", "a"]}, "ids must be unique"),
        ({"embeddings": VECTORS, "ids": "abcd"}, "ids must be a sequence"),
        ({"embeddings": VECTORS[:2], "ids": [["a"], ["b"]]}, "ids must be hashable"),
        ({"embeddings": VECTORS, "metadata": [{}, {}]}, "2 entries for 4 embeddings"),
        ({"embeddings": VECTORS, "metadata": {"owner": "a"}}, "not a single mapping"),
        ({"embeddings": VECTORS[:2], "metadata": [{}, "text"]}, r"metadata\[1\] must be a mapping"),
        ({"embeddings": VECTORS, "groups": ["g"]}, "1 entries for 4 embeddings"),
        ({"embeddings": VECTORS, "groups": "gggg"}, "groups must be a sequence"),
    ],
)
def test_invalid_fit_inputs(kwargs, match):
    with pytest.raises(ConfigurationError, match=match):
        RetrievalIndex().fit(**kwargs)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"query": np.zeros((1, 1, 2), dtype=np.float32)}, "3-D array"),
        ({"query": np.array([1.0, 0.0, 1.0], dtype=np.float32)}, "dimension 3, expected 2"),
        ({"query": np.array([1, 0])}, "2-D float array"),
        ({"query": np.array([np.nan, 1.0], dtype=np.float32)}, "NaN or infinite"),
        ({"query": RIGHT, "k": 0}, "k must be a positive int"),
        ({"query": RIGHT, "k": 2.0}, "k must be a positive int"),
        ({"query": RIGHT, "k": True}, "k must be a positive int"),
    ],
)
def test_invalid_search_inputs(index, kwargs, match):
    with pytest.raises(ConfigurationError, match=match):
        index.search(**kwargs)


def test_per_query_values_must_cover_every_row(index):
    with pytest.raises(ConfigurationError, match="query_ids must have one entry per query"):
        index.search(np.stack([RIGHT, RIGHT]), exclude_self=True, query_ids=["a"])
    with pytest.raises(ConfigurationError, match="query_ids must be a sequence"):
        index.search(np.stack([RIGHT, RIGHT]), exclude_self=True, query_ids="ab")


def test_results_are_frozen_records(index):
    result = index.search(RIGHT, k=1)[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.score = 0.0


def test_reprs_and_errors_never_reveal_metadata_or_embedding_values():
    built = RetrievalIndex()
    secret = {"account_number": "000123456789", "payee": "Jane Roe"}
    embeddings = np.array([[3.5, 0.0], [0.0, 2.25]], dtype=np.float32)
    built.fit(embeddings, ids=["s-1", "s-2"], metadata=[secret, None])
    results = built.search(embeddings[0], k=2)

    with pytest.raises(ConfigurationError) as mismatch:
        built.search(np.array([3.5, 0.0, 9.75], dtype=np.float32))
    with pytest.raises(ConfigurationError) as degenerate:
        built.fit(np.array([[0.0, 0.0], [7.125, 1.0]], dtype=np.float32))

    text = " ".join(
        [repr(results), repr(results[0]), repr(built), str(mismatch.value), str(degenerate.value)]
    )
    for forbidden in ("000123456789", "account_number", "Jane Roe", "3.5", "9.75", "7.125"):
        assert forbidden not in text
    assert "<present>" in text and "s-1" in text


class TestBruteForceCosineBackend:
    def test_reports_size_and_dimension(self):
        backend = BruteForceCosineBackend()
        assert (backend.size, backend.embedding_dim) == (0, None)
        backend.fit(VECTORS)
        assert (backend.size, backend.embedding_dim) == (4, 2)

    def test_returns_positions_and_scores_for_every_query(self):
        backend = BruteForceCosineBackend()
        backend.fit(VECTORS)
        positions, scores = backend.search(np.stack([RIGHT, VECTORS[3]]), 2)
        assert positions.tolist() == [[0, 2], [3, 1]]
        assert np.allclose(scores, [[1.0, 2**-0.5], [1.0, 0.0]], atol=1e-6)

    def test_width_is_capped_by_the_number_of_stored_vectors(self):
        backend = BruteForceCosineBackend()
        backend.fit(VECTORS)
        positions, scores = backend.search(RIGHT[None, :], 10)
        assert positions.shape == (1, 4) and scores.shape == (1, 4)

    def test_search_before_fit(self):
        with pytest.raises(IndexNotFittedError, match="call fit"):
            BruteForceCosineBackend().search(RIGHT[None, :], 1)

    def test_rejects_a_query_of_the_wrong_width(self):
        backend = BruteForceCosineBackend()
        backend.fit(VECTORS)
        with pytest.raises(ConfigurationError, match="dimension 3, expected 2"):
            backend.search(np.zeros((1, 3), dtype=np.float32) + 1.0, 1)


def test_retrieval_result_repr_hides_metadata_but_shows_presence():
    with_metadata = RetrievalResult(id="s-1", score=0.5, rank=1, metadata={"payee": "Jane Roe"})
    without = RetrievalResult(id="s-2", score=0.25, rank=2)
    assert repr(with_metadata) == "RetrievalResult(id='s-1', score=0.5, rank=1, metadata=<present>)"
    assert repr(without) == "RetrievalResult(id='s-2', score=0.25, rank=2, metadata=None)"


def test_float64_values_that_overflow_float32_are_rejected():
    embeddings = np.array([[1e300, 0.0], [1.0, 0.0]], dtype=np.float64)
    with pytest.raises(ConfigurationError, match="too large for float32"):
        RetrievalIndex().fit(embeddings)
    index = RetrievalIndex()
    index.fit(np.eye(2, dtype=np.float32))
    with pytest.raises(ConfigurationError, match="too large for float32"):
        index.search(np.array([1e300, 0.0]))


@pytest.mark.parametrize("scale", [1e-25, 1e20])
def test_extreme_but_representable_scales_keep_cosine_exact(scale):
    # Squaring these in float32 would underflow to 0 or overflow to inf.
    embeddings = (np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]) * scale).astype(np.float32)
    index = RetrievalIndex()
    index.fit(embeddings, ids=["x", "y", "xy"])
    hits = index.search(embeddings[0], k=3)
    assert [hit.id for hit in hits] == ["x", "xy", "y"]
    assert [hit.score for hit in hits] == pytest.approx([1.0, 2**-0.5, 0.0], abs=1e-6)


def test_ids_and_groups_match_on_type_as_well_as_value():
    index = RetrievalIndex()
    index.fit(np.eye(3, dtype=np.float32), ids=[0, False, "0"], groups=[1, True, 1])
    assert [hit.id for hit in index.search(np.array([1.0, 0.0, 0.0]), k=3, exclude_ids=[0])] == [
        False,
        "0",
    ]
    hits = index.search(np.ones(3), k=3, query_groups=True, exclude_same_group=True)
    assert [hit.id for hit in hits] == [0, "0"]  # Only the item in group True is removed.
    with pytest.raises(ConfigurationError, match="ids must be unique"):
        index.fit(np.eye(2, dtype=np.float32), ids=["a", "a"])


def test_an_earlier_exact_duplicate_outranks_the_query_item_itself():
    # Ties go to the earlier insertion, so "self is rank 1" holds only up to exact duplicates;
    # exclude_self still removes exactly the query's own item.
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index = RetrievalIndex()
    index.fit(embeddings, ids=["twin", "me", "other"])
    assert [hit.id for hit in index.search(embeddings[1], k=2)] == ["twin", "me"]
    hits = index.search(embeddings[1], k=2, query_ids="me", exclude_self=True)
    assert [hit.id for hit in hits] == ["twin", "other"]


def test_excluding_the_none_group_excludes_nothing():
    index = RetrievalIndex()
    index.fit(np.eye(3, dtype=np.float32), ids=["a", "b", "c"], groups=[None, "g", None])
    hits = index.search(np.ones(3), k=3, exclude_groups=[None])
    assert [hit.id for hit in hits] == ["a", "b", "c"]


def test_batches_without_per_query_exclusions_share_one_mask():
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(50, 6)).astype(np.float32)
    index = RetrievalIndex()
    index.fit(embeddings, ids=list(range(50)))
    queries = embeddings[:7]
    shared = index.search(queries, k=5, exclude_ids=[3, 4])
    single = [index.search(query, k=5, exclude_ids=[3, 4]) for query in queries]
    assert [[hit.id for hit in hits] for hits in shared] == [
        [hit.id for hit in hits] for hits in single
    ]
    assert not {3, 4} & {hit.id for hits in shared for hit in hits}
