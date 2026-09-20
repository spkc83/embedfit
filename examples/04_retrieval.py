"""Nearest-neighbour retrieval in the adapted embedding space, scored without leakage.

python examples/04_retrieval.py
"""

from _common import parse_args

from embedfit import EmbedFitModel, ImageDataset, RetrievalIndex, Trainer, TrainingArguments
from embedfit.retrieval import mean_average_precision, retrieval_precision_at_k
from embedfit.testing import BAR_CLASSES, make_bar_images


def main(argv: list[str] | None = None) -> dict[str, float]:
    args = parse_args(__doc__, argv)
    labels = list(BAR_CLASSES[:3])
    train = ImageDataset.from_lists(*make_bar_images(12, num_classes=3, seed=args.seed))
    images, targets = make_bar_images(15, num_classes=3, seed=args.seed + 1)

    model = EmbedFitModel.from_pretrained(args.backbone, labels=labels)
    training = TrainingArguments(
        training_mode="supcon",
        num_epochs=args.epochs,
        batch_size=12,
        learning_rate=args.learning_rate,
        num_workers=0,
        seed=args.seed,
    )
    Trainer(model, training, train).train()

    ids = [f"item-{position}" for position in range(len(images))]
    index = RetrievalIndex(metric="cosine")
    index.fit(model.encode(images), ids=ids, metadata=[{"label": label} for label in targets])

    # Query the index with its own items. exclude_self removes each query's own entry, which
    # would otherwise be a guaranteed perfect match and inflate every metric.
    hits = index.search(model.encode(images), k=5, query_ids=ids, exclude_self=True)
    relevant = [
        {
            other
            for other, label in zip(ids, targets, strict=True)
            if label == target and other != query
        }
        for query, target in zip(ids, targets, strict=True)
    ]
    first = hits[0]
    print(
        "neighbours of item-0 (",
        targets[0],
        "):",
        [(hit.id, hit.metadata["label"], round(hit.score, 2)) for hit in first[:3]],
    )
    results = {
        "precision_at_5": sum(
            retrieval_precision_at_k(h, r, 5) for h, r in zip(hits, relevant, strict=True)
        )
        / len(hits),
        "map_at_5": mean_average_precision(hits, relevant, 5),
    }
    print({name: round(value, 2) for name, value in results.items()})
    return results


if __name__ == "__main__":
    main()
