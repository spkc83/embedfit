# Retrieval Design

## Goal

Use the same learned embedding space for classification and similarity search.

## v0.1 backend

Brute-force cosine similarity using NumPy or PyTorch.

This is intentionally simple and testable.

## Interface

```python
index = RetrievalIndex(metric="cosine")
index.fit(embeddings, ids, metadata=None)
hits = index.search(query, k=10, exclude_ids=None)
```

## Result

```python
@dataclass(frozen=True)
class RetrievalResult:
    id: str
    score: float
    rank: int
    metadata: Mapping[str, Any] | None = None
```

## Safety against evaluation leakage

Evaluation utilities should optionally exclude:

- same sample ID
- same physical-item group
- custom forbidden groups

A near-duplicate of the same original item should not count as evidence that the representation generalizes.

## Future backends

Design an internal backend protocol compatible later with:

- FAISS
- pgvector
- Qdrant
- Milvus
- OpenSearch

Do not add these dependencies in v0.1.
