"""Retrieval over real encoder output: images → encode → index → neighbours (acceptance G)."""

import numpy as np
import pytest

from embedfit import EmbedFitModel, RetrievalIndex
from embedfit.retrieval import retrieval_precision_at_k
from embedfit.testing import make_bar_images

NUM_PER_CLASS = 4


@pytest.fixture
def indexed():
    """A cosine index over the embeddings of 12 synthetic images, three classes of four."""
    images, labels = make_bar_images(NUM_PER_CLASS, num_classes=3, seed=0)
    model = EmbedFitModel.from_pretrained("dummy/tiny", embedding_dim=32)
    embeddings = model.encode(images)
    ids = [f"sample-{position:02d}" for position in range(len(images))]
    index = RetrievalIndex(metric="cosine")
    index.fit(
        embeddings,
        ids=ids,
        metadata=[{"shape": label} for label in labels],
        groups=labels,
    )
    return index, np.asarray(embeddings), ids, labels


def test_query_returns_sorted_scores_with_the_self_match_first(indexed):
    index, embeddings, ids, labels = indexed
    for position, identifier in enumerate(ids):
        results = index.search(embeddings[position], k=5)
        scores = [result.score for result in results]
        assert scores == sorted(scores, reverse=True)
        assert [result.rank for result in results] == [1, 2, 3, 4, 5]
        assert results[0].id == identifier
        assert results[0].score == pytest.approx(1.0, abs=1e-5)
        # Acceptance G: the metadata travels with the id it was indexed under.
        assert all(result.metadata == {"shape": labels[ids.index(result.id)]} for result in results)


def test_exclude_self_drops_the_query_and_still_fills_k(indexed):
    index, embeddings, ids, _ = indexed
    for position, identifier in enumerate(ids):
        results = index.search(embeddings[position], k=5, exclude_self=True, query_ids=identifier)
        assert len(results) == 5
        assert identifier not in [result.id for result in results]


def test_group_exclusion_removes_every_image_of_the_query_class(indexed):
    index, embeddings, ids, labels = indexed
    results = index.search(embeddings[0], k=8, exclude_same_group=True, query_groups=labels[0])
    assert len(results) == len(ids) - NUM_PER_CLASS
    assert all(result.metadata is not None for result in results)
    assert all(result.metadata["shape"] != labels[0] for result in results)


def test_batch_search_matches_per_query_search(indexed):
    index, embeddings, ids, _ = indexed
    batch = index.search(embeddings, k=3, exclude_self=True, query_ids=ids)
    assert len(batch) == len(ids)
    for position, identifier in enumerate(ids):
        single = index.search(embeddings[position], k=3, exclude_self=True, query_ids=identifier)
        assert [result.id for result in batch[position]] == [result.id for result in single]
        assert [result.rank for result in batch[position]] == [result.rank for result in single]
        assert [result.metadata for result in batch[position]] == [
            result.metadata for result in single
        ]
        # A blocked matrix product accumulates in a different order from a single-row one, so
        # the scores agree only to float32 precision.
        assert [result.score for result in batch[position]] == pytest.approx(
            [result.score for result in single], abs=1e-6
        )


def test_neighbours_of_an_indexed_item_can_be_scored(indexed):
    index, embeddings, ids, labels = indexed
    relevant = {
        identifier for identifier, label in zip(ids, labels, strict=True) if label == labels[0]
    }
    results = index.search(embeddings[0], k=3, exclude_self=True, query_ids=ids[0])
    same_class = relevant - {ids[0]}
    hits = sum(result.id in same_class for result in results)
    assert retrieval_precision_at_k(results, same_class, 3) == hits / 3
