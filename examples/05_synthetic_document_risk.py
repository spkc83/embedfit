"""A rare-event document-risk workflow on synthetic images, end to end.

The images are generated shapes: a frame, a few ruled lines, and an empty box. "Risk" items
carry a small bright mark inside the box. Nothing here is a real document, and the labels,
groups, and amounts are random numbers. What the example shows is the *workflow* that
rare-event image problems need:

1. an out-of-time test set, and train/validation splits that keep groups together;
2. representation training on the training split only, with validation choosing the epoch;
3. a class-weighted head, because positives are rare;
4. ranking metrics instead of accuracy, including value-weighted capture;
5. a decision threshold chosen on validation data and applied, unchanged, to the test set;
6. retrieval of similar earlier items for a flagged one.

    python examples/05_synthetic_document_risk.py
"""

from datetime import datetime, timedelta

import numpy as np
from _common import parse_args
from PIL import Image

from embedfit import EmbedFitModel, ImageDataset, RetrievalIndex, Trainer, TrainingArguments
from embedfit.data import ImageSample, find_split_overlap, grouped_split, temporal_split
from embedfit.evaluation import (
    dollar_recall,
    dollar_recall_at_review_rate,
    pr_auc,
    precision_at_review_rate,
    recall_at_fpr,
    recall_at_review_rate,
    select_threshold,
)
from embedfit.heads import LogisticRegressionHead

LABELS = ["routine", "risk"]


def synthetic_item(rng: np.random.Generator, risky: bool, size: int = 32) -> Image.Image:
    canvas = rng.normal(0.15, 0.04, (size, size))
    canvas[[1, size - 2], 1:-1] = canvas[1:-1, [1, size - 2]] = 0.6  # Frame.
    for row in rng.choice(np.arange(5, 17), size=3, replace=False):  # Ruled lines.
        canvas[row, 4 : int(rng.integers(14, 28))] = 0.5
    canvas[20:29, 16:29] = 0.3  # The box.
    if risky:
        top, left = int(rng.integers(21, 25)), int(rng.integers(17, 25))
        canvas[top : top + 3, left : left + 3] = 1.0  # The mark that defines the risk class.
    return Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8)).convert("RGB")


def synthetic_dataset(count: int, seed: int) -> ImageDataset:
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1)
    samples = []
    for position in range(count):
        risky = bool(rng.random() < 0.12)
        samples.append(
            ImageSample(
                image=synthetic_item(rng, risky),
                label="risk" if risky else "routine",
                sample_id=f"item-{position:04d}",
                group_id=f"source-{int(rng.integers(0, 40)):02d}",
                timestamp=start + timedelta(hours=position),
                amount=float(np.round(rng.lognormal(6.5 if risky else 5.5, 0.8), 2)),
            )
        )
    return ImageDataset(samples)


def main(argv: list[str] | None = None) -> dict[str, float]:
    args = parse_args(__doc__, argv, epochs=10)
    dataset = synthetic_dataset(480, seed=args.seed)

    # 1. Splits: the most recent quarter is the test set; groups never straddle train/validation.
    development, test = temporal_split(dataset, test_size=0.25)
    train, validation = grouped_split(development, test_size=0.25, seed=args.seed)
    # Must hold: nothing is shared between the two splits that drive training decisions.
    find_split_overlap({"train": train, "validation": validation}).raise_if_any()
    # To be judged: the same sources keep sending items, so the out-of-time test set shares
    # groups with training. That answers "how well are new items from known sources scored?".
    # To measure generalisation to unseen sources instead, drop those groups from one side.
    print(find_split_overlap({"train": train, "validation": validation, "test": test}).summary())
    print(
        {
            name: f"{len(split)} items, {split.labels.count('risk')} risk"
            for name, split in (("train", train), ("validation", validation), ("test", test))
        }
    )

    # 2-3. Adapt the representation, then fit a class-weighted head.
    model = EmbedFitModel.from_pretrained(
        args.backbone,
        labels=LABELS,
        positive_label="risk",
        head=LogisticRegressionHead(class_weight="balanced"),
    )
    training = TrainingArguments(
        training_mode="supcon",
        num_epochs=args.epochs,
        batch_size=32,
        learning_rate=args.learning_rate,
        num_workers=0,
        seed=args.seed,
    )
    Trainer(model, training, train, eval_dataset=validation).train()

    # 4. Ranking metrics on the test set, because accuracy rewards a model that flags nothing.
    never_flag_accuracy = test.labels.count("routine") / len(test)
    print(f"accuracy of never flagging anything: {never_flag_accuracy:.2f}")
    scores = model.risk_score(test.images)
    amounts = [sample.amount for sample in test]
    positive = {"positive_label": "risk"}
    results = {
        "pr_auc": pr_auc(test.labels, scores, **positive),
        "recall_at_5pct_fpr": recall_at_fpr(test.labels, scores, target_fpr=0.05, **positive),
        "precision_at_10pct_review": precision_at_review_rate(
            test.labels, scores, review_rate=0.10, **positive
        ),
        "recall_at_10pct_review": recall_at_review_rate(
            test.labels, scores, review_rate=0.10, **positive
        ),
        "dollar_recall_at_10pct_review": dollar_recall_at_review_rate(
            test.labels, scores, amounts, review_rate=0.10, **positive
        ),
    }

    # 5. The threshold comes from validation data and is applied to the test set as it is.
    chosen = select_threshold(
        validation.labels,
        model.risk_score(validation.images),
        objective="target_fpr",
        target=0.05,
        **positive,
    )
    flagged = scores >= chosen.threshold
    routine = np.array(test.labels) == "routine"
    results["never_flag_accuracy"] = never_flag_accuracy
    results["dollar_recall_at_threshold"] = dollar_recall(test.labels, flagged, amounts, **positive)
    results["test_review_rate_at_threshold"] = float(flagged.mean())
    # The operating point moves between validation and test: the threshold met its 5% target
    # on validation, but what it does on new data has to be measured, not assumed.
    results["validation_fpr_at_threshold"] = chosen.fpr
    results["test_fpr_at_threshold"] = float(flagged[routine].mean())
    for name, value in results.items():
        print(f"{name:32s} {value:.3f}")
    print(
        f"threshold {chosen.threshold:.3f}: target FPR 0.050, validation {chosen.fpr:.3f}, "
        f"test {results['test_fpr_at_threshold']:.3f}"
    )

    # 6. For the highest-scoring test item, retrieve the most similar training items.
    index = RetrievalIndex()
    index.fit(
        model.encode(train.images),
        ids=[sample.sample_id for sample in train],
        metadata=[{"label": sample.label} for sample in train],
    )
    top = int(np.argmax(scores))
    neighbours = index.search(model.encode(test.images[top])[0], k=5)
    print(
        f"most similar training items to {test[top].sample_id} (true label {test[top].label}):",
        [(hit.id, hit.metadata["label"], round(hit.score, 2)) for hit in neighbours],
    )
    results["top_item_neighbours_risk_share"] = float(
        np.mean([hit.metadata["label"] == "risk" for hit in neighbours])
    )
    return results


if __name__ == "__main__":
    main()
