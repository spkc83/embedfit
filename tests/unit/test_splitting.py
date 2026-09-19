from collections import Counter
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from embedfit import ImageDataset
from embedfit.data import (
    ImageSample,
    find_split_overlap,
    grouped_split,
    stratified_split,
    temporal_split,
)
from embedfit.exceptions import ConfigurationError, DataLeakageError

START = datetime(2024, 1, 1)


def make_dataset(num=40, *, groups=8, positives=8) -> ImageDataset:
    """Samples s0..s{num-1}: group g{i % groups}, one day apart, first ``positives`` are risk."""
    return ImageDataset(
        [
            ImageSample(
                image=f"img-{i}.png",
                label="risk" if i < positives else "ok",
                sample_id=f"s{i}",
                group_id=f"g{i % groups}",
                timestamp=START + timedelta(days=i),
            )
            for i in range(num)
        ]
    )


def ids(dataset):
    return [sample.sample_id for sample in dataset]


def assert_partition(dataset, train, test):
    assert sorted(ids(train) + ids(test)) == sorted(ids(dataset))
    assert not set(ids(train)) & set(ids(test))
    order = {sample_id: position for position, sample_id in enumerate(ids(dataset))}
    for side in (train, test):  # Original relative order is kept on each side.
        assert [order[i] for i in ids(side)] == sorted(order[i] for i in ids(side))


class TestStratifiedSplit:
    def test_keeps_class_proportions(self):
        dataset = make_dataset()
        train, test = stratified_split(dataset, test_size=0.25, seed=0)
        assert_partition(dataset, train, test)
        assert Counter(test.labels) == {"risk": 2, "ok": 8}  # round(8 * .25), round(32 * .25)
        assert Counter(train.labels) == {"risk": 6, "ok": 24}

    def test_every_class_lands_on_both_sides_even_when_rounding_says_otherwise(self):
        dataset = make_dataset(num=22, positives=2)
        train, test = stratified_split(dataset, test_size=0.1, seed=0)  # round(2 * .1) = 0
        assert Counter(test.labels)["risk"] == 1 and Counter(train.labels)["risk"] == 1
        train, test = stratified_split(dataset, test_size=0.9, seed=0)  # round(2 * .9) = 2
        assert Counter(test.labels)["risk"] == 1 and Counter(train.labels)["risk"] == 1

    def test_seed_controls_the_shuffle_and_global_rng_is_untouched(self):
        dataset = make_dataset()
        np.random.seed(5)
        expected_next = np.random.rand()
        np.random.seed(5)
        first = stratified_split(dataset, test_size=0.25, seed=1)
        assert np.random.rand() == expected_next
        again = stratified_split(dataset, test_size=0.25, seed=1)
        other = stratified_split(dataset, test_size=0.25, seed=2)
        assert ids(first[1]) == ids(again[1]) != ids(other[1])

    def test_labels_are_separated_by_type(self):
        dataset = ImageDataset.from_lists(["a"] * 8, [1, 1, 1, 1, True, True, True, True])
        _, test = stratified_split(dataset, test_size=0.5, seed=0)
        assert sorted(map(repr, test.labels)) == ["1", "1", "True", "True"]

    def test_class_with_one_sample_is_rejected(self):
        with pytest.raises(ConfigurationError, match="smallest class has 1"):
            stratified_split(make_dataset(num=10, positives=1), test_size=0.2, seed=0)


class TestGroupedSplit:
    @pytest.mark.parametrize("seed", range(5))
    def test_no_group_is_split(self, seed):
        dataset = make_dataset()
        train, test = grouped_split(dataset, test_size=0.25, seed=seed)
        assert_partition(dataset, train, test)
        train_groups = {s.group_id for s in train}
        test_groups = {s.group_id for s in test}
        assert not train_groups & test_groups
        assert len(test) == 10  # 8 groups of 5: two whole groups reach the 25% target.
        find_split_overlap({"train": train, "test": test}).raise_if_any()

    def test_whole_groups_can_overshoot_the_target(self, caplog):
        dataset = make_dataset(num=40, groups=2)
        with caplog.at_level("WARNING", logger="embedfit"):
            train, test = grouped_split(dataset, test_size=0.1, seed=0)
        assert (len(train), len(test)) == (20, 20)
        assert "test side holds 20 of 40 samples (50.0%), requested 10.0%" in caplog.text

    @pytest.mark.parametrize("seed", range(6))
    def test_a_dominant_group_can_force_an_undershoot_which_is_reported(self, seed, caplog):
        # Groups of 1, 1, and 38 samples: the possible test sizes are 1, 2, 38, and 39, so a
        # 25% request (10 samples) cannot be met. The closest reachable size is chosen.
        sizes = {"a": 1, "b": 1, "c": 38}
        dataset = ImageDataset(
            [
                ImageSample(f"{g}{i}.png", "x", sample_id=f"{g}{i}", group_id=g)
                for g, n in sizes.items()
                for i in range(n)
            ]
        )
        with caplog.at_level("WARNING", logger="embedfit"):
            train, test = grouped_split(dataset, test_size=0.25, seed=seed)
        assert 1 <= len(test) <= 2 or len(test) == 38
        assert not {s.group_id for s in train} & {s.group_id for s in test}
        assert "requested 25.0%" in caplog.text and "c0" not in caplog.text

    def test_the_closest_reachable_size_is_preferred(self):
        # Groups of 6 and target 10: one group (6) is closer than two (12)? No: |12-10| = 2
        # beats |6-10| = 4, so two groups are taken.
        dataset = make_dataset(num=36, groups=6)
        assert len(grouped_split(dataset, test_size=10 / 36, seed=0)[1]) == 12
        # Target 8: |6-8| = 2 equals |12-8| = 4? No, 2 < 4, so one group is taken.
        assert len(grouped_split(dataset, test_size=8 / 36, seed=0)[1]) == 6

    def test_a_class_confined_to_one_group_is_reported(self, caplog):
        dataset = ImageDataset(
            [
                ImageSample(f"{i}.png", "risk" if i < 5 else "ok", group_id=f"g{i // 5}")
                for i in range(40)
            ]
        )
        with caplog.at_level("WARNING", logger="embedfit"):
            grouped_split(dataset, test_size=0.25, seed=0)
        assert "1 of 2 classes have no samples on the" in caplog.text
        assert "risk" not in caplog.text

    def test_different_seeds_pick_different_groups(self):
        picks = {
            frozenset(s.group_id for s in grouped_split(make_dataset(), test_size=0.25, seed=s)[1])
            for s in range(8)
        }
        assert len(picks) > 1

    def test_requirements(self):
        with pytest.raises(ConfigurationError, match="at least 2 groups"):
            grouped_split(make_dataset(groups=1), test_size=0.5, seed=0)
        partial = ImageDataset([ImageSample("a.png", "x", group_id="g"), ImageSample("b.png", "x")])
        with pytest.raises(ConfigurationError, match="1 of 2 samples have no group_id"):
            grouped_split(partial, test_size=0.5, seed=0)


class TestTemporalSplit:
    def test_cutoff_puts_earlier_samples_in_train(self):
        dataset = make_dataset()
        train, test = temporal_split(dataset, cutoff=START + timedelta(days=30))
        assert_partition(dataset, train, test)
        assert ids(train) == [f"s{i}" for i in range(30)]  # The cut-off instant is test.
        assert max(s.timestamp for s in train) < min(s.timestamp for s in test)

    def test_test_size_takes_the_most_recent_samples(self):
        train, test = temporal_split(make_dataset(), test_size=0.25)
        assert ids(test) == [f"s{i}" for i in range(30, 40)]

    def test_input_order_does_not_matter(self):
        shuffled = make_dataset().subset(np.random.default_rng(0).permutation(40).tolist())
        _, test = temporal_split(shuffled, test_size=0.25)
        assert sorted(ids(test)) == sorted(f"s{i}" for i in range(30, 40))

    def test_samples_sharing_the_boundary_instant_stay_together(self):
        stamps = [
            START,
            START,
            START + timedelta(days=1),
            START + timedelta(days=1),
            START + timedelta(days=1),
            START + timedelta(days=2),
        ]
        dataset = ImageDataset(
            [
                ImageSample(f"{i}.png", "x", sample_id=f"s{i}", timestamp=t)
                for i, t in enumerate(stamps)
            ]
        )
        # ceil(6 * .34) = 3 starts inside the day-1 tie, so all three day-1 samples go to test.
        _, test = temporal_split(dataset, test_size=0.34)
        assert ids(test) == ["s2", "s3", "s4", "s5"]

    def test_invalid_use(self):
        dataset = make_dataset()
        with pytest.raises(ConfigurationError, match="exactly one of cutoff or test_size"):
            temporal_split(dataset)
        with pytest.raises(ConfigurationError, match="exactly one"):
            temporal_split(dataset, cutoff=START, test_size=0.2)
        with pytest.raises(ConfigurationError, match="leaves one side empty: 0 train"):
            temporal_split(dataset, cutoff=START)
        with pytest.raises(ConfigurationError, match="leaves one side empty: 40 train"):
            temporal_split(dataset, cutoff=START + timedelta(days=400))
        with pytest.raises(ConfigurationError, match="cutoff must be a datetime"):
            temporal_split(dataset, cutoff="2024-01-15")
        with pytest.raises(ConfigurationError, match="timezone-aware or all naive"):
            temporal_split(dataset, cutoff=datetime(2024, 1, 15, tzinfo=UTC))
        with pytest.raises(ConfigurationError, match="40 of 40 samples have no timestamp"):
            temporal_split(ImageDataset.from_lists(["a"] * 40, ["x"] * 40), test_size=0.2)

    @pytest.mark.parametrize(
        "stamps, found",
        [
            ([f"2024-01-0{d}" for d in range(1, 5)], "str"),
            ([1, 2, 3, 4], "int"),
            ([datetime(2024, 1, d).date() for d in range(1, 5)], "date"),
        ],
    )
    def test_only_datetime_timestamps_are_compared(self, stamps, found):
        dataset = ImageDataset(
            [ImageSample(f"{i}.png", "x", timestamp=stamp) for i, stamp in enumerate(stamps)]
        )
        with pytest.raises(
            ConfigurationError, match=rf"4 of 4 timestamps are not datetime.*{found}"
        ):
            temporal_split(dataset, test_size=0.5)

    def test_a_class_seen_only_before_the_cutoff_is_reported(self, caplog):
        with caplog.at_level("WARNING", logger="embedfit"):
            _, test = temporal_split(make_dataset(positives=10), test_size=0.25)
        assert set(test.labels) == {"ok"}
        assert "temporal_split: 1 of 2 classes have no samples on the test side" in caplog.text


class TestFindSplitOverlap:
    def test_clean_splits(self):
        train, test = grouped_split(make_dataset(), test_size=0.25, seed=0)
        report = find_split_overlap({"train": train, "test": test})
        assert not report.has_overlap and not report.missing
        assert report.summary() == "No overlap between splits."
        report.raise_if_any()

    def test_reports_shared_samples_and_groups_per_pair(self):
        dataset = make_dataset()
        splits = {"train": dataset[:20], "validation": dataset[18:30], "test": dataset[30:]}
        report = find_split_overlap(splits)
        assert report.overlaps["train", "validation", "sample_id"] == {"s18", "s19"}
        assert ("train", "test", "sample_id") not in report.overlaps
        assert report.overlaps["train", "test", "group_id"] == {f"g{i}" for i in range(8)}
        assert report.has_overlap

    def test_stratified_split_of_grouped_data_is_flagged(self):
        train, test = stratified_split(make_dataset(), test_size=0.25, seed=0)
        report = find_split_overlap({"train": train, "test": test})
        assert ("train", "test", "sample_id") not in report.overlaps
        assert report.overlaps["train", "test", "group_id"]

    def test_error_and_repr_give_counts_never_identifier_values(self):
        dataset = make_dataset()
        report = find_split_overlap({"train": dataset[:20], "test": dataset[18:]})
        with pytest.raises(DataLeakageError) as excinfo:
            report.raise_if_any()
        for text in (str(excinfo.value), repr(report), report.summary()):
            assert "2 shared sample_id value(s) between 'train' and 'test'" in text
            assert "s18" not in text and "g3" not in text

    def test_missing_values_are_counted_and_fully_absent_fields_are_an_error(self):
        first = ImageDataset([ImageSample("a", "x", sample_id="1"), ImageSample("b", "x")])
        second = ImageDataset([ImageSample("c", "x", sample_id="1")])
        report = find_split_overlap({"a": first, "b": second}, fields=["sample_id"])
        assert report.missing == {("a", "sample_id"): 1}
        assert report.overlaps == {("a", "b", "sample_id"): frozenset({"1"})}
        with pytest.raises(ConfigurationError, match="No sample in split 'a' has a group_id"):
            find_split_overlap({"a": first, "b": second})

    def test_identifiers_match_on_value_and_type(self):
        first = ImageDataset([ImageSample("a", "x", sample_id=1)])
        second = ImageDataset([ImageSample("b", "x", sample_id=True)])
        third = ImageDataset([ImageSample("c", "x", sample_id=1)])
        report = find_split_overlap({"a": first, "b": second, "c": third}, fields=["sample_id"])
        assert dict(report.overlaps) == {("a", "c", "sample_id"): frozenset({1})}

    def test_report_is_hashable_and_read_only(self):
        dataset = make_dataset()
        report = find_split_overlap({"train": dataset[:20], "test": dataset[18:]})
        assert report in {report} and report == report
        with pytest.raises(TypeError):
            report.overlaps["x", "y", "sample_id"] = frozenset({"z"})
        with pytest.raises(TypeError):
            report.missing["x", "sample_id"] = 1

    def test_unhashable_identifiers(self):
        bad = ImageDataset(
            [ImageSample("a", "x", sample_id=["s"]), ImageSample("b", "x", sample_id="t")]
        )
        with pytest.raises(ConfigurationError, match="sample_id values must be hashable"):
            find_split_overlap({"a": bad, "b": bad}, fields=["sample_id"])
        grouped = ImageDataset([ImageSample(str(i), "x", group_id=[i]) for i in range(4)])
        with pytest.raises(ConfigurationError, match="group_id values must be hashable"):
            grouped_split(grouped, test_size=0.5, seed=0)

    def test_invalid_arguments(self):
        dataset = make_dataset()
        with pytest.raises(ConfigurationError, match="at least two"):
            find_split_overlap({"train": dataset})
        with pytest.raises(ConfigurationError, match="fields must be chosen from"):
            find_split_overlap({"a": dataset, "b": dataset}, fields=["label"])
        with pytest.raises(ConfigurationError, match="must be an ImageDataset"):
            find_split_overlap({"a": dataset, "b": [1, 2]})


@pytest.mark.parametrize("test_size", [0, 1, -0.1, 1.5, "0.2", None, True])
def test_invalid_test_size(test_size):
    with pytest.raises(ConfigurationError, match="test_size"):
        stratified_split(make_dataset(), test_size=test_size, seed=0)
    with pytest.raises(ConfigurationError, match="test_size"):
        grouped_split(make_dataset(), test_size=test_size, seed=0)


def test_numpy_integer_seeds_are_accepted():
    dataset = make_dataset()
    assert ids(stratified_split(dataset, test_size=0.25, seed=np.int64(3))[1]) == ids(
        stratified_split(dataset, test_size=0.25, seed=3)[1]
    )


@pytest.mark.parametrize("seed", [-1, 1.5, "0", None, True])
def test_invalid_seed(seed):
    with pytest.raises(ConfigurationError, match="seed"):
        stratified_split(make_dataset(), test_size=0.2, seed=seed)


def test_invalid_dataset():
    with pytest.raises(ConfigurationError, match="must be an ImageDataset"):
        stratified_split([1, 2, 3], test_size=0.2, seed=0)
    with pytest.raises(ConfigurationError, match="at least 2 samples"):
        grouped_split(make_dataset(num=1), test_size=0.2, seed=0)


def test_subset():
    dataset = make_dataset(num=5)
    assert ids(dataset.subset([3, 0, -1])) == ["s3", "s0", "s4"]
    assert ids(dataset.subset(np.array([4, 1]))) == ["s4", "s1"]
    assert ids(dataset.subset(np.flatnonzero(np.array([1, 0, 1, 0, 0])))) == ["s0", "s2"]
    assert len(dataset.subset([])) == 0
    for bad in ([5], [-6], [1.0], [True]):
        with pytest.raises(ConfigurationError, match="subset positions"):
            dataset.subset(bad)
