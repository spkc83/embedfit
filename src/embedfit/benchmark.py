"""Few-shot benchmark harness: methods × shots × seeds, reported as tidy records."""

import logging
import math
import weakref
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from embedfit.data.dataset import ImageDataset
from embedfit.evaluation.registry import requested_metrics
from embedfit.exceptions import ConfigurationError
from embedfit.model import EmbedFitModel
from embedfit.sampling.few_shot import sample_few_shot
from embedfit.trainer import Trainer
from embedfit.training_args import TRAINING_MODES, TrainingArguments

logger = logging.getLogger(__name__)

Record = dict[str, Any]
"""One measurement: ``method``, ``backbone``, ``shots``, ``seed``, ``metric``, ``value``."""


def _positive_ints(values: Any, name: str, *, minimum: int) -> tuple[int, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence) or len(values) == 0:
        raise ConfigurationError(f"{name} must be a non-empty sequence of ints.")
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < minimum:
            raise ConfigurationError(f"{name} must contain ints >= {minimum}, got {value!r}.")
        result.append(int(value))
    if len(set(result)) != len(result):
        raise ConfigurationError(f"{name} must not contain duplicates.")
    return tuple(result)


class FewShotEvaluator:
    """Compare training methods at several training-set sizes over repeated seeds.

    For every combination of method, shot count, and seed, the evaluator draws ``shots``
    training samples per class from the training pool (``sample_few_shot`` with that seed),
    trains a fresh model from ``model_factory``, and evaluates it on the fixed test set. Every
    method sees exactly the same few-shot sample for a given ``(shots, seed)``, so differences
    between methods are not sampling noise.

    Args:
        shots: Training samples per class to evaluate, e.g. ``[4, 8, 16, 32, 64, 128]``.
        seeds: Seeds for the few-shot draw and for training; one run per seed.

    Raises:
        ConfigurationError: If ``shots`` or ``seeds`` is empty, contains duplicates, or
            contains a value that is not an int (shots >= 1, seeds >= 0).
    """

    def __init__(self, shots: Sequence[int], seeds: Sequence[int]) -> None:
        self.shots = _positive_ints(shots, "shots", minimum=1)
        self.seeds = _positive_ints(seeds, "seeds", minimum=0)

    def run(
        self,
        dataset: ImageDataset,
        *,
        test_dataset: ImageDataset,
        methods: Sequence[str],
        model_factory: Callable[[], EmbedFitModel],
        training_arguments: Mapping[str, TrainingArguments] | None = None,
        metrics: Sequence[str] = (),
    ) -> list[Record]:
        """Run the benchmark.

        Args:
            dataset: The training pool that few-shot samples are drawn from. It must not
                overlap the test set.
            test_dataset: Held-out samples every run is evaluated on. It is never sampled from
                and never passed to training, so no checkpoint or threshold depends on it.
            methods: Training modes to compare, e.g. ``["frozen", "end_to_end", "supcon"]``.
            model_factory: Returns a new, untrained ``EmbedFitModel`` with labels on every
                call. Each run gets its own model, so runs never share weights. It is also
                called once per method up front, to validate the configuration.
            training_arguments: Optional ``TrainingArguments`` per method. A method without an
                entry uses ``TrainingArguments(training_mode=method)``. The ``seed`` of each
                run replaces the one in the arguments.
            metrics: Registered metrics to record in addition to the defaults; see
                ``Trainer.evaluate``. The models need a ``positive_label``.

        Returns:
            Tidy records, one per method × shots × seed × metric, with the keys ``method``,
            ``backbone``, ``shots``, ``seed``, ``metric``, and ``value``. The metrics are those
            of ``Trainer.evaluate(metrics=metrics)``.

        Raises:
            ConfigurationError: If a method is unknown or duplicated, the arguments given for
                a method name a different ``training_mode`` or set ``checkpoint_dir``, a dataset
                has the wrong type, the factory returns something other than a new
                ``EmbedFitModel`` with a new backbone, a class in the pool has fewer samples than
                the largest shot count, a method cannot train at the smallest shot count, or
                ``metrics`` is invalid for the factory's models. All of these are raised before
                any model is trained.
        """
        if not isinstance(metrics, str) and isinstance(metrics, Iterable):
            metrics = tuple(metrics)  # Checked per model below, then used by every evaluate.
        if isinstance(methods, str) or not isinstance(methods, Sequence) or not methods:
            raise ConfigurationError("methods must be a non-empty sequence of training modes.")
        unknown = [method for method in methods if method not in TRAINING_MODES]
        if unknown or len(set(methods)) != len(methods):
            raise ConfigurationError(
                f"methods must be distinct training modes from {list(TRAINING_MODES)}, "
                f"got {list(methods)}."
            )
        for name, candidate in (("dataset", dataset), ("test_dataset", test_dataset)):
            if not isinstance(candidate, ImageDataset) or len(candidate) == 0:
                raise ConfigurationError(f"{name} must be a non-empty ImageDataset.")
        if not callable(model_factory):
            raise ConfigurationError("model_factory must be callable.")
        if training_arguments is not None and not isinstance(training_arguments, Mapping):
            raise ConfigurationError("training_arguments must map method names to arguments.")
        arguments = {}
        for method in methods:
            configured = (training_arguments or {}).get(method)
            if configured is not None and not isinstance(configured, TrainingArguments):
                raise ConfigurationError(
                    f"training_arguments['{method}'] must be a TrainingArguments, got "
                    f"{type(configured).__name__}."
                )
            arguments[method] = configured or TrainingArguments(training_mode=method)
            if arguments[method].training_mode != method:
                raise ConfigurationError(
                    f"training_arguments['{method}'] has training_mode "
                    f"'{arguments[method].training_mode}'."
                )
            if arguments[method].checkpoint_dir is not None:
                raise ConfigurationError(
                    f"training_arguments['{method}'] sets checkpoint_dir, but every run would "
                    "write to the same directory; FewShotEvaluator does not write checkpoints."
                )
        used: list[weakref.ref[Any]] = []

        def fresh_model() -> EmbedFitModel:
            model = model_factory()
            if not isinstance(model, EmbedFitModel):
                raise ConfigurationError("model_factory must return an EmbedFitModel.")
            # A reused model, backbone, or even a single shared parameter (a shallow copy, a
            # shared submodule) would carry trained weights from one run into the next.
            used[:] = [ref for ref in used if ref() is not None]  # Forget collected models.
            alive = {id(ref()) for ref in used}
            parameters = list(model.backbone.parameters())
            if alive & {id(model), id(model.backbone), *(id(p) for p in parameters)}:
                raise ConfigurationError(
                    "model_factory must return a new EmbedFitModel with a new backbone on "
                    "every call; a shared one would leak trained weights between runs."
                )
            used.extend(weakref.ref(target) for target in (model, model.backbone, *parameters))
            return model

        # Fail before any training: the pool must cover the largest shot count, and every
        # method must accept its arguments at the smallest one (e.g. contrastive modes need
        # two samples per class).
        sample_few_shot(dataset, shots=max(self.shots), seed=self.seeds[0])
        smallest = sample_few_shot(dataset, shots=min(self.shots), seed=self.seeds[0])
        for method in methods:
            model = fresh_model()
            requested_metrics(metrics, model.positive_label)
            Trainer(model, replace(arguments[method], seed=self.seeds[0]), smallest)

        records: list[Record] = []
        for shots in self.shots:
            for seed in self.seeds:
                few_shot = sample_few_shot(dataset, shots=shots, seed=seed)
                for method in methods:
                    model = fresh_model()
                    trainer = Trainer(model, replace(arguments[method], seed=seed), few_shot)
                    trainer.train()
                    for metric, value in trainer.evaluate(test_dataset, metrics=metrics).items():
                        records.append(
                            {
                                "method": method,
                                "backbone": model.backbone.identifier,
                                "shots": shots,
                                "seed": seed,
                                "metric": metric,
                                "value": float(value),
                            }
                        )
                    logger.info("Benchmarked %s at %d shots, seed %d", method, shots, seed)
        return records

    @staticmethod
    def aggregate(records: Sequence[Mapping[str, Any]]) -> list[Record]:
        """Summarise records over seeds.

        Args:
            records: Records returned by ``run``.

        Returns:
            One row per ``method`` × ``backbone`` × ``shots`` × ``metric`` with ``mean``,
            ``std`` (sample standard deviation, 0 for a single run), ``num_runs`` (runs with a
            defined value), and ``num_undefined`` (runs whose value was NaN, which are left out
            of the mean). Rows keep the order in which their group first appears.

        Raises:
            ConfigurationError: If a record lacks one of the expected keys or its value is not
                a number.
        """
        groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
        try:
            for record in records:
                record["seed"]  # Required, although runs are not grouped by it.
                key = (record["method"], record["backbone"], record["shots"], record["metric"])
                groups[key].append(float(record["value"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(
                "records must be mappings with the keys method, backbone, shots, seed, metric, "
                "and a numeric value."
            ) from exc
        rows: list[Record] = []
        for (method, backbone, shots, metric), values in groups.items():
            defined = [value for value in values if not math.isnan(value)]
            rows.append(
                {
                    "method": method,
                    "backbone": backbone,
                    "shots": shots,
                    "metric": metric,
                    "mean": float(np.mean(defined)) if defined else float("nan"),
                    "std": float(np.std(defined, ddof=1)) if len(defined) > 1 else 0.0,
                    "num_runs": len(defined),
                    "num_undefined": len(values) - len(defined),
                }
            )
        return rows
