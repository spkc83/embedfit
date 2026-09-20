# Avoiding Data Leakage

Leakage is any path by which information about evaluation samples reaches training. It makes a model look better than it will be in use, and few-shot settings are especially exposed: with a handful of training samples per class, a single leaked near-duplicate can dominate a result.

## Where leakage comes from

| Source | Example | Defence |
| --- | --- | --- |
| The same sample in two splits | One image listed twice under different file names | `find_split_overlap` on `sample_id` |
| Related samples in two splits | Front and back scans of one item; several items from one entity or template | `grouped_split`, `find_split_overlap` on `group_id` |
| The future informing the past | Training on items observed after the test period | `temporal_split` |
| Labels or fields unavailable at decision time | Outcome codes recorded weeks after the event | Keep them out of `image`; `metadata` is never a model feature |
| Tuning on the test set | Choosing a threshold, checkpoint, or few-shot examples using test data | Use a validation split for every choice; touch the test split once |

## Choosing a split

In increasing order of realism:

1. **`stratified_split`** shuffles within each class. It ignores groups and time, so it is a debugging baseline, not evidence of generalisation.
2. **`grouped_split`** keeps every `group_id` on one side. Use the coarsest identifier that links related samples: the physical item, the account, the template, the device.
3. **`temporal_split`** trains strictly before a cut-off and tests at or after it. This mirrors deployment, where a model only ever scores data newer than itself, and it exposes drift that random splits hide.

Grouping and time interact: an entity active on both sides of a cut-off still links train and test. After a temporal split, check `group_id` overlap and decide deliberately whether it is acceptable for the question being asked.

The example below assumes every sample carries a `timestamp` (a `datetime`), a `group_id`, and a `sample_id`. The split functions raise when the field they need is missing, and `find_split_overlap` raises when a field it is asked to compare is absent from every sample of a split; pass `fields=["group_id"]` to compare one field only.

```python
from embedfit.data import find_split_overlap, grouped_split, temporal_split
from embedfit.exceptions import DataLeakageError

development, test = temporal_split(dataset, test_size=0.2)   # most recent 20% held out
train, validation = grouped_split(development, test_size=0.25, seed=0)

# Must hold: nothing is shared between the two splits that drive training decisions.
find_split_overlap({"train": train, "validation": validation}).raise_if_any()

# To be judged: groups active on both sides of the cut-off link the test set to training.
report = find_split_overlap({"train": train, "validation": validation, "test": test})
print(report.summary())     # counts only; identifier values are never printed
```

`raise_if_any()` raises `DataLeakageError` when any `sample_id` or `group_id` is shared. `train` and `validation` share no groups by construction, so the first check passes. The second report typically lists `group_id` values shared with `test`; whether that is acceptable depends on the question. "How well does the model score new items from known entities?" tolerates it, while "how well does it generalise to unseen entities?" does not, and then the shared groups should be removed from one side.

`grouped_split` and `temporal_split` do not balance classes. With rare classes, check that each class is present where it is needed; both functions log a warning when a class is missing from one side.

## Few-shot sampling

`embedfit.sampling.sample_few_shot(train, shots=8, seed=0)` draws N samples per class from the pool it is given. Split first, then sample from the training side only. A class with fewer than N samples raises rather than sampling with replacement, because repeated samples would silently change what "N-shot" means.

## Rules the library follows

- Representation training and head fitting use `train_dataset` only. `Trainer` never trains on `eval_dataset`: it is scored after each epoch solely to choose which epoch's weights to keep, which is why it must be validation data and never the test set. `select_threshold` documents that it is for validation data.
- Splits are reproducible: each function takes a `seed` and uses its own random generator.
- `find_split_overlap` refuses to run when a requested identifier is absent from every sample of a split, because a check that cannot fail is worse than no check. Samples with a missing identifier are counted in `report.missing`.
- Reports and errors give counts. Identifier values are available programmatically in `report.overlaps` but never appear in messages or logs.
