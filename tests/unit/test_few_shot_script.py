"""benchmarks/few_shot.py: options, resuming, and WiSE-FT interpolation."""

import importlib.util
import json
import math
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from embedfit import (
    EmbedFitModel,
    FewShotEvaluator,
    ImageDataset,
    Trainer,
    TrainingArguments,
)
from embedfit.exceptions import ConfigurationError
from embedfit.heads import LinearHead
from embedfit.representation import _linear_probe
from embedfit.testing import BAR_CLASSES, make_bar_images

SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks" / "few_shot.py"


@pytest.fixture(scope="module")
def few_shot():
    spec = importlib.util.spec_from_file_location("few_shot_benchmark", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OLD = {"dataset": "eurosat", "gradient_accumulation_steps": 1, "dataset_revision": None}
CURRENT = {
    **OLD,
    "unfreeze_last_blocks": None,
    "trainable_parameters": "all",
    "augmentation": "none",
    "max_steps": None,
    "projection_dim": None,
    "contrastive_views": 1,
    "temperature": 0.07,
    "contrastive_weight": 1.0,
    "sampler": {},
    "device": "cpu",
    "interpolate": [],
    "max_image_side": None,
    "heads": None,
    "head_learning_rate": {},
    "layer_lr_decay": {},
}


def write_results(path, config):
    results = {"config": config, "environment": {"sessions": []}, "records": []}
    path.write_text(json.dumps(results))


@pytest.mark.parametrize(
    ("key", "default", "other"),
    [
        ("unfreeze_last_blocks", None, 2),
        ("trainable_parameters", "all", "layernorm"),
        ("augmentation", "none", "natural"),
        ("max_steps", None, 100),
        ("projection_dim", None, 128),
        ("contrastive_views", 1, 2),
        ("temperature", 0.07, 0.2),
        ("contrastive_weight", 1.0, 0.5),
        ("sampler", {}, {"end_to_end": "balanced"}),
        ("device", "cpu", "cuda"),
        ("interpolate", [], [0.5]),
        ("max_image_side", None, 224),
        ("heads", None, ["linear"]),
        ("head_learning_rate", {}, {"end_to_end": 1e-3}),
        ("layer_lr_decay", {}, {"supcon": 0.8}),
    ],
)
def test_a_file_from_before_an_option_resumes_with_its_default(
    few_shot, tmp_path, key, default, other
):
    output = tmp_path / "results.json"
    write_results(output, OLD)
    assert few_shot.read_previous(output, CURRENT) is not None
    with pytest.raises(SystemExit, match="different configuration"):
        few_shot.read_previous(output, {**CURRENT, key: other})


@pytest.mark.parametrize(
    "option",
    [
        "--unfreeze-last-blocks",
        "--max-steps",
        "--projection-dim",
        "--contrastive-views",
        "--max-image-side",
    ],
)
@pytest.mark.parametrize("value", ["0", "-1", "x"])
def test_count_options_must_be_positive_ints(few_shot, option, value, capsys):
    with pytest.raises(SystemExit):
        few_shot.main([option, value])
    assert f"{option}: expected a positive int" in capsys.readouterr().err


@pytest.mark.parametrize("option", ["--temperature", "--contrastive-weight"])
@pytest.mark.parametrize("value", ["0", "-0.1", "inf", "nan", "x"])
def test_temperature_and_weight_must_be_positive_numbers(few_shot, option, value, capsys):
    with pytest.raises(SystemExit):
        few_shot.main([option, value])
    assert f"{option}: expected a positive number" in capsys.readouterr().err


REQUIRED = ["--dataset", "eurosat", "--evaluate-on", "validation", "--output", "results.json"]


def test_an_unknown_augmentation_is_rejected(few_shot, capsys):
    with pytest.raises(SystemExit):
        few_shot.main([*REQUIRED, "--augmentation", "sepia"])
    assert "--augmentation: invalid choice" in capsys.readouterr().err


@pytest.mark.parametrize(
    "option",
    [
        ["--augmentation", "natural"],
        ["--unfreeze-last-blocks", "1"],
        ["--trainable-parameters", "bias"],
        ["--max-steps", "50"],
    ],
)
def test_training_options_need_a_trained_method(few_shot, option, capsys):
    with pytest.raises(SystemExit):
        few_shot.main([*REQUIRED, "--methods", "frozen", *option])
    assert "applies to trained methods only" in capsys.readouterr().err


@pytest.mark.parametrize(
    "option",
    [
        ["--projection-dim", "128"],
        ["--contrastive-views", "2", "--augmentation", "natural"],
        ["--temperature", "0.2"],
    ],
)
def test_contrastive_options_need_a_contrastive_method(few_shot, option, capsys):
    with pytest.raises(SystemExit):
        few_shot.main([*REQUIRED, "--methods", "frozen", "end_to_end", "pair", *option])
    assert "applies to the supcon and ce_supcon methods only" in capsys.readouterr().err
    for method in ("supcon", "ce_supcon"):
        few_shot.parse_arguments([*REQUIRED, "--methods", method, *option])


def test_the_contrastive_weight_needs_the_ce_supcon_method(few_shot, capsys):
    with pytest.raises(SystemExit):
        few_shot.main([*REQUIRED, "--methods", "end_to_end", "supcon", "--contrastive-weight", "2"])
    assert "applies to the ce_supcon method only" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("option", "message"),
    [
        (["end_to_end"], "expects METHOD=NAME"),
        (["pair=balanced"], "expects METHOD=NAME"),
        (["end_to_end=uniform"], "expects METHOD=NAME"),
        (["supcon=random"], "names supcon, which is not selected"),
        (["end_to_end=balanced", "end_to_end=random"], "names end_to_end twice"),
    ],
)
def test_a_sampler_names_a_selected_batch_method_and_a_builtin_sampler(
    few_shot, option, message, capsys
):
    with pytest.raises(SystemExit):
        few_shot.parse_arguments(
            [*REQUIRED, "--methods", "end_to_end", "ce_supcon", "pair", "--sampler", *option]
        )
    assert message in capsys.readouterr().err


def test_a_sampler_is_configuration_and_a_default_one_is_left_out(few_shot):
    methods = ["--methods", "end_to_end", "ce_supcon", "supcon"]
    options = [*methods, "--sampler", "ce_supcon=balanced", "end_to_end=balanced"]
    config = namespace(few_shot, *options)
    assert config["sampler"] == {"end_to_end": "balanced"}
    assert namespace(few_shot, *methods, "--sampler", "end_to_end=balanced") == config
    assert namespace(few_shot, *methods)["sampler"] == {}
    args = few_shot.parse_arguments([*REQUIRED, *options, "--sampler", "supcon=random"])
    rates = few_shot.parse_learning_rates(args.learning_rate, args.default_learning_rate)
    arguments = few_shot.training_arguments(args, rates)
    assert {method: arguments[method].sampler for method in arguments} == {
        "end_to_end": "balanced",
        "ce_supcon": None,
        "supcon": "random",
    }
    combined = few_shot.parse_arguments(
        [*REQUIRED, *methods, "--sampler", "end_to_end=balanced", "--trainable-parameters", "bias"]
    )
    joint = few_shot.training_arguments(combined, rates)["end_to_end"]
    assert (joint.sampler, joint.trainable_parameters) == ("balanced", "bias")


def test_the_default_methods_leave_ce_supcon_out(few_shot):
    assert few_shot.parse_arguments(REQUIRED).methods == ["frozen", "end_to_end", "supcon", "pair"]


def test_ce_supcon_is_end_to_end_with_a_contrastive_weight(few_shot):
    args = few_shot.parse_arguments(
        [*REQUIRED, "--methods", "ce_supcon", "supcon", "--contrastive-weight", "0.5"]
        + ["--projection-dim", "16", "--temperature", "0.2", "--learning-rate", "ce_supcon=3e-5"]
    )
    rates = few_shot.parse_learning_rates(args.learning_rate, args.default_learning_rate)
    arguments = few_shot.training_arguments(args, rates)
    joint, supcon = arguments["ce_supcon"], arguments["supcon"]
    assert (joint.training_mode, joint.contrastive_weight, joint.learning_rate) == (
        "end_to_end",
        0.5,
        3e-5,
    )
    assert (joint.projection_dim, joint.temperature) == (16, 0.2)
    assert (supcon.projection_dim, supcon.temperature, supcon.contrastive_weight) == (16, 0.2, 0)
    assert few_shot.resume_configuration(args)["contrastive_weight"] == 0.5


def test_several_views_need_an_augmentation(few_shot, capsys):
    with pytest.raises(SystemExit):
        few_shot.main([*REQUIRED, "--methods", "supcon", "--contrastive-views", "2"])
    assert "--contrastive-views above 1 needs --augmentation" in capsys.readouterr().err


def namespace(few_shot, *options):
    parser_args = few_shot.parse_arguments([*REQUIRED, *options])
    return few_shot.resume_configuration(parser_args)


def test_a_step_budget_does_not_record_the_ignored_epochs(few_shot):
    config = namespace(few_shot, "--max-steps", "100", "--epochs", "3")
    assert (config["max_steps"], config["epochs"]) == (100, None)
    assert namespace(few_shot, "--max-steps", "100", "--epochs", "7") == config
    assert namespace(few_shot, "--epochs", "3")["epochs"] == 3


def test_per_part_rates_are_configuration_by_value_and_reach_the_methods(few_shot):
    options = ["--projection-dim", "8", "--head-learning-rate", "end_to_end=1e-3"]
    config = namespace(few_shot, *options, "supcon=0.001", "--layer-lr-decay", "pair=.5")
    assert config["head_learning_rate"] == {"end_to_end": 1e-3, "supcon": 1e-3}
    assert config["layer_lr_decay"] == {"pair": 0.5}
    respelled = namespace(few_shot, *options, "supcon=1e-3", "--layer-lr-decay", "pair=0.5")
    assert respelled == config
    args = few_shot.parse_arguments([*REQUIRED, *options, "--layer-lr-decay", "pair=0.5"])
    rates = few_shot.parse_learning_rates(args.learning_rate, args.default_learning_rate)
    arguments = few_shot.training_arguments(args, rates)
    assert {m: (a.head_learning_rate, a.layer_lr_decay) for m, a in arguments.items()} == {
        "end_to_end": (1e-3, None),
        "supcon": (None, None),
        "pair": (None, 0.5),
    }
    assert namespace(few_shot)["head_learning_rate"] == {}


def test_per_method_options_add_up_over_repeats(few_shot):
    args = few_shot.parse_arguments(
        [
            *REQUIRED,
            *("--learning-rate", "supcon=1e-5", "--learning-rate", "pair=2e-5"),
            *("--head-learning-rate", "end_to_end=1e-3", "--head-learning-rate", "supcon=1e-2"),
            *("--layer-lr-decay", "pair=0.5", "--layer-lr-decay", "supcon=0.8"),
            *("--projection-dim", "8"),
        ]
    )
    rates = few_shot.parse_learning_rates(args.learning_rate, args.default_learning_rate)
    assert (rates["supcon"], rates["pair"]) == (1e-5, 2e-5)
    assert args.head_learning_rate == {"end_to_end": 1e-3, "supcon": 1e-2}
    assert args.layer_lr_decay == {"pair": 0.5, "supcon": 0.8}


@pytest.mark.parametrize("option", ["--learning-rate", "--head-learning-rate", "--layer-lr-decay"])
@pytest.mark.parametrize("separate", [False, True])
def test_a_method_named_twice_is_rejected(few_shot, option, separate, capsys):
    first, second = "end_to_end=0.5", "end_to_end=0.25"
    repeated = [option, first, option, second] if separate else [option, first, second]
    with pytest.raises(SystemExit):
        few_shot.main([*REQUIRED, *repeated])
    assert f"{option} names a method twice" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (["--head-learning-rate", "frozen=1e-3"], "METHOD a selected trained method"),
        (["--methods", "supcon", "--layer-lr-decay", "pair=0.5"], "a selected trained method"),
        (["--layer-lr-decay", "0.5"], "METHOD a selected trained method"),
        (["--head-learning-rate", "end_to_end=x"], "with a number as VALUE"),
        (["--head-learning-rate", "pair=1e-3"], "trains neither"),
        (["--head-learning-rate", "supcon=1e-3"], "trains neither"),
        (["--head-learning-rate", "end_to_end=0"], "head_learning_rate must be finite and > 0"),
        (["--layer-lr-decay", "supcon=1.5"], "layer_lr_decay must be in"),
        (["--head-learning-rate", "end_to_end=inf"], "head_learning_rate must be finite"),
        (["--learning-rate", "supcon=inf"], "learning_rate must be finite"),
    ],
)
def test_invalid_per_part_rates_are_rejected(few_shot, options, message, capsys):
    with pytest.raises(SystemExit):
        few_shot.main([*REQUIRED, *options])
    assert message in capsys.readouterr().err


def test_the_device_is_part_of_the_configuration_and_cudnn_is_not(few_shot, monkeypatch):
    monkeypatch.setattr(few_shot, "resolve_device", lambda device: few_shot.torch.device(device))
    monkeypatch.setattr(few_shot.torch.cuda, "current_device", lambda: 0)
    config = namespace(few_shot, "--device", "cuda", "--disable-cudnn")
    assert config["device"] == "cuda:0"
    assert "disable_cudnn" not in config
    assert namespace(few_shot, "--device", "cuda") == config


@pytest.mark.parametrize("device", ["auto", "gpu"])
def test_the_device_must_be_a_named_available_device(few_shot, device, capsys):
    with pytest.raises(SystemExit):
        few_shot.parse_arguments([*REQUIRED, "--device", device])
    assert "--device" in capsys.readouterr().err


def test_one_cuda_device_has_one_spelling(few_shot, monkeypatch):
    monkeypatch.setattr(few_shot, "resolve_device", lambda device: few_shot.torch.device(device))
    monkeypatch.setattr(few_shot.torch.cuda, "current_device", lambda: 0)
    assert namespace(few_shot, "--device", "cuda") == namespace(few_shot, "--device", "cuda:0")
    assert namespace(few_shot, "--device", "cuda:0")["device"] == "cuda:0"


def test_a_record_from_before_interpolation_has_no_weight(few_shot, tmp_path):
    output = tmp_path / "results.json"
    record = {"method": "supcon", "shots": 2, "seed": 0, "metric": "accuracy", "value": 0.5}
    results = {"config": OLD, "environment": {"sessions": []}, "records": [record]}
    output.write_text(json.dumps(results))
    assert few_shot.read_previous(output, CURRENT)["records"][0]["alpha"] is None


def test_a_record_from_before_heads_names_its_methods_own_head(few_shot, tmp_path):
    output = tmp_path / "results.json"
    records = [
        {"method": method, "shots": 2, "seed": 0, "metric": "accuracy", "value": 0.5}
        for method in ("frozen", "end_to_end", "supcon", "pair")
    ]
    output.write_text(
        json.dumps({"config": OLD, "environment": {"sessions": []}, "records": records})
    )
    heads = [r["head"] for r in few_shot.read_previous(output, CURRENT)["records"]]
    assert heads == ["logistic_regression", "linear", "logistic_regression", "logistic_regression"]


# Checks of the shots and seeds.


@pytest.mark.parametrize(
    ("option", "message"),
    [
        (["--shots", "2", "4", "2"], "--shots names a value twice"),
        (["--seeds", "1", "1"], "--seeds names a value twice"),
        (["--shots", "0", "2"], "--shots must be at least 1, got 0"),
        (["--seeds", "-1"], "--seeds must be at least 0, got -1"),
        (["--seeds", "0", str(2**32)], f"--seeds must be at most 2**32 - 1, got {2**32}"),
    ],
)
def test_shots_and_seeds_are_distinct_and_in_range(few_shot, option, message, capsys):
    with pytest.raises(SystemExit):
        few_shot.parse_arguments([*REQUIRED, *option])
    assert message in capsys.readouterr().err


# The environment record.


def test_the_session_records_the_commit_and_the_package_versions(few_shot):
    args = few_shot.parse_arguments(REQUIRED)
    session = few_shot.new_session(args)
    names = ("embedfit", "git_commit", "git_dirty", "torch", "torchvision", "transformers", "timm")
    for name in names:
        assert name in session
    assert session["scikit-learn"] == few_shot.metadata.version("scikit-learn")
    assert session["torchvision"] == few_shot.metadata.version("torchvision")
    commit = session["git_commit"]
    assert commit is None or re.fullmatch("[0-9a-f]{40}", commit)
    assert session["git_dirty"] in (None, True, False)
    assert few_shot.package_version("no-such-distribution-embedfit") is None


@pytest.mark.parametrize(
    "outcome",
    [
        OSError("no git"),
        subprocess.TimeoutExpired("git", 30),
        SimpleNamespace(returncode=128, stdout=""),
    ],
)
def test_the_commit_is_none_outside_a_git_checkout(few_shot, monkeypatch, outcome):
    def fake(*args, **kwargs):
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(few_shot.subprocess, "run", fake)
    assert few_shot.git_commit() is None
    assert few_shot.git_dirty() is None


@pytest.mark.parametrize(("status", "dirty"), [("", False), (" M benchmarks/few_shot.py\n", True)])
def test_a_checkout_is_dirty_with_any_change(few_shot, monkeypatch, status, dirty):
    result = SimpleNamespace(returncode=0, stdout=status)
    monkeypatch.setattr(few_shot.subprocess, "run", lambda *args, **kwargs: result)
    assert few_shot.git_dirty() is dirty


def test_the_largest_seed_is_accepted(few_shot):
    assert few_shot.parse_arguments([*REQUIRED, "--seeds", str(2**32 - 1)]).seeds == [2**32 - 1]


def test_a_head_is_only_reused_if_it_is_the_methods_own(few_shot):
    train = ImageDataset.from_lists(*bars(3, seed=1))
    model = EmbedFitModel.from_pretrained("dummy/tiny", labels=CLASSES, image_size=16)
    trainer = Trainer(model, TrainingArguments(training_mode="frozen"), train)
    trainer.train()
    model.head_name = "linear"  # Not the head the records would name for frozen.
    with pytest.raises(AssertionError):
        next(few_shot.fitted_heads(trainer, ["linear"]))


# The head every method is scored with.


def test_heads_default_to_none_and_are_configuration_in_one_order(few_shot):
    assert namespace(few_shot)["heads"] is None
    config = namespace(few_shot, "--heads", "logistic_regression", "linear")
    assert config["heads"] == ["linear", "logistic_regression"]
    assert namespace(few_shot, "--heads", "linear", "logistic_regression") == config


@pytest.mark.parametrize(
    ("heads", "message"),
    [(["linear", "linear"], "--heads names a head twice"), (["mlp"], "invalid choice")],
)
def test_heads_are_known_and_given_once(few_shot, heads, message, capsys):
    with pytest.raises(SystemExit):
        few_shot.parse_arguments([*REQUIRED, "--heads", *heads])
    assert message in capsys.readouterr().err


def test_each_method_has_its_own_head(few_shot):
    assert few_shot.own_head("end_to_end") == "linear"
    for method in ("frozen", "supcon", "pair"):
        assert few_shot.own_head(method) == "logistic_regression"


# WiSE-FT interpolation.


@pytest.mark.parametrize("value", ["-0.1", "1", "1.5", "nan", "x"])
def test_interpolation_weights_lie_in_the_unit_interval(few_shot, value, capsys):
    with pytest.raises(SystemExit):
        few_shot.parse_arguments([*REQUIRED, "--interpolate", value])
    assert "--interpolate: expected a number in [0, 1)" in capsys.readouterr().err


def test_an_interpolation_weight_is_given_once(few_shot, capsys):
    with pytest.raises(SystemExit):
        few_shot.parse_arguments([*REQUIRED, "--interpolate", "0.5", "0.5"])
    assert "--interpolate names a weight twice" in capsys.readouterr().err


def test_interpolation_needs_a_trained_method(few_shot, capsys):
    with pytest.raises(SystemExit):
        few_shot.parse_arguments([*REQUIRED, "--methods", "frozen", "--interpolate", "0.5"])
    assert "applies to trained methods only" in capsys.readouterr().err


def test_the_interpolation_weights_are_configuration_in_one_order(few_shot):
    config = namespace(few_shot, "--interpolate", "0.75", "0.25")
    assert config["interpolate"] == [0.25, 0.75]
    assert namespace(few_shot, "--interpolate", "0.25", "0.75") == config
    assert namespace(few_shot)["interpolate"] == []


def test_interpolation_mixes_floating_point_tensors_and_copies_integer_ones(few_shot):
    module = nn.Sequential(nn.Linear(3, 2), nn.BatchNorm1d(2))
    initial = few_shot.backbone_state(module, torch.device("cpu"))
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(1.0)
    module.train()
    module(torch.randn(4, 3))  # Moves the running statistics and counts the batch.
    trained = few_shot.backbone_state(module, torch.device("cpu"))
    assert trained["1.num_batches_tracked"] == 1
    few_shot.interpolate_backbone(module, initial, trained, 0.25)
    state = dict(module.named_parameters()) | dict(module.named_buffers())
    for name, tensor in state.items():
        if tensor.is_floating_point():
            torch.testing.assert_close(tensor, 0.75 * initial[name] + 0.25 * trained[name])
        else:
            assert torch.equal(tensor, trained[name])
    few_shot.interpolate_backbone(module, initial, trained, 0.0)
    for name, tensor in state.items():
        expected = initial[name] if tensor.is_floating_point() else trained[name]
        assert torch.equal(tensor, expected)


# Tiny runs on synthetic images with the dummy backbone.

CLASSES = list(BAR_CLASSES)


def bars(per_class, seed):
    return make_bar_images(per_class, num_classes=len(CLASSES), image_size=16, seed=seed)


def synthetic_loader(root, *, pool, evaluation, evaluate_on, seed):
    splits = {
        "pool": list(zip(*bars(pool, seed=10 + seed), strict=True)),
        evaluate_on: list(zip(*bars(evaluation, seed=20 + seed), strict=True)),
    }
    return splits, list(CLASSES)


TINY = [
    "--dataset", "eurosat", "--evaluate-on", "validation", "--backbone", "dummy/tiny",
    "--image-size", "16", "--pool-per-class", "6", "--eval-per-class", "3", "--shots", "2",
    "--seeds", "0", "--threads", "1", "--batch-size", "8", "--max-steps", "2",
    "--default-learning-rate", "1e-2",
]  # fmt: skip


@pytest.fixture
def run(few_shot, monkeypatch, tmp_path):
    monkeypatch.setitem(few_shot.LOADERS, "eurosat", synthetic_loader)
    for name in ("HF_HOME", "TORCH_HOME", "TMPDIR"):  # The script only sets unset ones.
        monkeypatch.setenv(name, str(tmp_path))
    threads = torch.get_num_threads()

    def call(*options, output="results.json"):
        path = tmp_path / output
        few_shot.main(
            [*TINY, "--output", str(path), "--data-root", str(tmp_path / "data"), *options]
        )
        return json.loads(path.read_text())

    yield call
    torch.set_num_threads(threads)


def values(records, method, alpha):
    return {
        r["metric"]: r["value"] for r in records if (r["method"], r["alpha"]) == (method, alpha)
    }


def test_the_plain_records_are_those_of_the_few_shot_evaluator(few_shot, run):
    methods = ["frozen", "end_to_end", "supcon", "pair"]
    records = run("--methods", *methods)["records"]
    assert {r["alpha"] for r in records} == {None}
    splits = synthetic_loader(None, pool=6, evaluation=3, evaluate_on="validation", seed=0)[0]
    pool, evaluation = (
        ImageDataset.from_lists([image for image, _ in items], [label for _, label in items])
        for items in (splits["pool"], splits["validation"])
    )
    args = few_shot.parse_arguments([*TINY, "--output", "unused.json", "--methods", *methods])
    rates = few_shot.parse_learning_rates(args.learning_rate, args.default_learning_rate)
    expected = FewShotEvaluator(shots=[2], seeds=[0]).run(
        pool,
        test_dataset=evaluation,
        methods=methods,
        model_factory=lambda: EmbedFitModel.from_pretrained(
            "dummy/tiny", labels=CLASSES, image_size=16, device="cpu"
        ),
        training_arguments=few_shot.training_arguments(args, rates),
    )
    assert [{k: v for k, v in r.items() if k not in {"alpha", "head"}} for r in records] == expected
    assert all(r["head"] == few_shot.own_head(r["method"]) for r in records)


def test_interpolation_at_zero_is_the_frozen_baseline(run, capsys):
    results = run("--methods", "frozen", "supcon", "pair", "--interpolate", "0", "0.5")
    records = results["records"]
    frozen = values(records, "frozen", None)
    assert values(records, "supcon", None) != frozen  # Training changed the backbone.
    for method in ("supcon", "pair"):
        assert values(records, method, 0.0) == frozen
        assert values(records, method, 0.5).keys() == frozen.keys()
    assert {(r["method"], r["alpha"]) for r in records} == {
        ("frozen", None),
        *((method, alpha) for method in ("supcon", "pair") for alpha in (None, 0.0, 0.5)),
    }
    printed = capsys.readouterr().out
    assert "supcon α=0" in printed and "pair α=0.5" in printed


def scores(records, method, alpha, head):
    return {
        r["metric"]: r["value"]
        for r in records
        if (r["method"], r["alpha"], r["head"]) == (method, alpha, head)
    }


def test_every_method_is_scored_with_every_requested_head(few_shot, run, capsys):
    methods = ("--methods", "frozen", "end_to_end", "supcon", "--interpolate", "0")
    both = run(*methods, "--heads", "logistic_regression", "linear", output="both.json")
    records = both["records"]
    heads = ("linear", "logistic_regression")
    assert {(r["method"], r["alpha"], r["head"]) for r in records} == {
        *(("frozen", None, head) for head in heads),
        *(
            (method, alpha, head)
            for method in ("end_to_end", "supcon")
            for alpha in (None, 0.0)
            for head in heads
        ),
    }
    # At alpha 0 every trained method is the pretrained backbone, so with the same head it
    # scores exactly as frozen does: the heads are the same whichever method they follow.
    for head in heads:
        frozen = scores(records, "frozen", None, head)
        for method in ("end_to_end", "supcon"):
            assert scores(records, method, 0.0, head) == frozen
    assert scores(records, "frozen", None, "linear") != scores(
        records, "frozen", None, "logistic_regression"
    )
    printed = capsys.readouterr().out
    assert "end_to_end logistic_regression α=0.0" in printed and "frozen linear " in printed
    # Without --heads, each method's records are those of its own head.
    own = run(*methods, output="own.json")["records"]
    assert own == [r for r in records if r["head"] == few_shot.own_head(r["method"])]


def test_the_backbone_is_seeded_before_it_is_built(few_shot, run, monkeypatch):
    seeds = []
    real = few_shot.EmbedFitModel.from_pretrained
    monkeypatch.setattr(
        few_shot.EmbedFitModel,
        "from_pretrained",
        lambda *a, **k: seeds.append(torch.initial_seed()) or real(*a, **k),
    )
    run("--methods", "frozen", "supcon", "--seeds", "3", "5")
    assert seeds[-4:] == [3, 3, 5, 5]  # After the checks before training.


def test_bias_only_training_is_recorded_and_interpolates_only_the_biases(few_shot, run):
    results = run(
        "--methods", "frozen", "supcon", "--trainable-parameters", "bias", "--interpolate", "0"
    )
    assert results["config"]["trainable_parameters"] == "bias"
    records = results["records"]
    assert values(records, "supcon", 0.0) == values(records, "frozen", None)
    args = few_shot.parse_arguments(
        [*REQUIRED, "--methods", "supcon", "--trainable-parameters", "bias"]
    )
    rates = few_shot.parse_learning_rates(args.learning_rate, args.default_learning_rate)
    assert few_shot.training_arguments(args, rates)["supcon"].trainable_parameters == "bias"


def test_interpolating_a_bias_only_run_leaves_the_other_weights_pretrained(few_shot):
    train = ImageDataset.from_lists(*bars(3, seed=1))
    model = EmbedFitModel.from_pretrained("dummy/tiny", labels=CLASSES, image_size=16)
    initial = few_shot.backbone_state(model.backbone, "cpu")
    budget = {"batch_size": 8, "num_workers": 0, "max_steps": 2, "learning_rate": 1e-2}
    arguments = TrainingArguments(training_mode="supcon", trainable_parameters="bias", **budget)
    trainer = Trainer(model, arguments, train)
    for alpha in few_shot.trained_weights(trainer, [0.5]):  # As trained, then at 0.5.
        for name, tensor in model.backbone.state_dict().items():
            assert torch.equal(tensor, initial[name]) != name.endswith(".bias"), (alpha, name)


def test_ce_supcon_is_interpolated_toward_the_same_linear_probe(few_shot, run, capsys):
    results = run("--methods", "end_to_end", "ce_supcon", "--interpolate", "0", "0.5")
    records = results["records"]
    assert values(records, "ce_supcon", None).keys() == values(records, "end_to_end", None).keys()
    assert values(records, "ce_supcon", 0.0) == values(records, "end_to_end", 0.0)
    assert set(results["config"]["learning_rates"]) == {"end_to_end", "ce_supcon"}
    assert "ce_supcon α=0.5" in capsys.readouterr().out
    # Without --heads, ce_supcon is scored with the linear head of end_to_end (ADR 0011).
    assert {r["head"] for r in records if r["method"] == "ce_supcon"} == {"linear"}
    assert few_shot.own_head("ce_supcon") == few_shot.own_head("end_to_end") == "linear"


def end_to_end_trainer():
    train = ImageDataset.from_lists(*bars(3, seed=1))
    model = EmbedFitModel.from_pretrained("dummy/tiny", labels=CLASSES, image_size=16)
    budget = {"batch_size": 8, "num_workers": 0, "max_steps": 2, "learning_rate": 1e-2}
    return Trainer(model, TrainingArguments(training_mode="end_to_end", **budget), train)


def probabilities(few_shot, alphas):
    trainer = end_to_end_trainer()
    images = bars(2, seed=30)[0]
    return {
        alpha: trainer.model.predict_proba(images)
        for alpha in few_shot.trained_weights(trainer, alphas)
    }


def test_an_interpolated_head_does_not_depend_on_the_other_weights(few_shot):
    alone, together = probabilities(few_shot, [0.5]), probabilities(few_shot, [0.0, 0.5])
    assert np.array_equal(alone[None], together[None])
    assert np.array_equal(alone[0.5], together[0.5])
    assert not np.array_equal(together[0.0], together[0.5])


def test_end_to_end_at_zero_is_its_linear_probe(few_shot):
    trainer = end_to_end_trainer()
    weights = few_shot.trained_weights(trainer, [0.0])
    assert next(weights) is None
    trained = trainer.model.head.to_module()
    assert next(weights) == 0.0
    head = trainer.model.head
    assert isinstance(head, LinearHead)
    fresh = end_to_end_trainer()
    probe = _linear_probe(fresh.model, fresh.train_dataset, fresh.args)
    assert torch.equal(head.to_module().weight, probe.weight)
    assert torch.equal(head.to_module().bias, probe.bias)
    assert not torch.equal(trained.weight, probe.weight)


def test_frozen_is_never_interpolated(few_shot):
    train = ImageDataset.from_lists(*bars(3, seed=1))
    model = EmbedFitModel.from_pretrained("dummy/tiny", labels=CLASSES, image_size=16)
    trainer = Trainer(model, TrainingArguments(training_mode="frozen"), train)
    assert list(few_shot.trained_weights(trainer, [0.0, 0.5])) == [None]


def test_distinct_weights_have_distinct_summary_rows(few_shot):
    args = few_shot.parse_arguments([*REQUIRED, "--interpolate", "1e-7", "1.000001e-7", "-0"])
    assert args.interpolate == [0.0, 1e-7, 1.000001e-7]
    assert math.copysign(1, args.interpolate[0]) == 1
    rows = few_shot.row_names(["frozen", "supcon"], args.interpolate)
    assert rows == ["frozen", "supcon", "supcon α=0.0", "supcon α=1e-07", "supcon α=1.000001e-07"]


def test_interpolation_needs_states_of_the_backbone(few_shot):
    module = nn.Linear(2, 2)
    state = few_shot.backbone_state(module, "cpu")
    partial = {"weight": state["weight"]}
    for initial, trained in ((partial, state), (state, partial), (state, {**state, "x": 1})):
        with pytest.raises(ConfigurationError, match="exactly the backbone's parameters"):
            few_shot.interpolate_backbone(module, initial, trained, 0.5)


def test_the_environment_stops_nnpack_warnings_where_it_cannot_run(
    few_shot, monkeypatch, tmp_path, capfd
):
    torch = few_shot.torch
    for name in ("HF_HOME", "TORCH_HOME", "TMPDIR"):  # setdefault then keeps these.
        monkeypatch.setenv(name, str(tmp_path))
    monkeypatch.setattr(torch.backends.nnpack, "is_available", lambda: False)
    enabled, threads = torch._C._get_nnpack_enabled(), torch.get_num_threads()
    try:
        args = few_shot.parse_arguments([*REQUIRED, "--threads", str(threads)])
        args.data_root = tmp_path
        few_shot.prepare_environment(args)
        assert not torch._C._get_nnpack_enabled()
        capfd.readouterr()
        torch.nn.Conv2d(3, 4, 3)(torch.rand(1, 3, 8, 8))
        assert "NNPACK" not in capfd.readouterr().err
    finally:
        torch.backends.nnpack.set_flags(enabled)


# --max-image-side: images decoded once and shrunk in memory.


@pytest.mark.parametrize(
    ("size", "side", "expected"),
    [
        ((640, 480), 224, (299, 224)),  # Shorter side capped, aspect ratio kept.
        ((300, 900), 224, (224, 672)),
        ((200, 150), 224, (200, 150)),  # Never enlarged.
        ((224, 300), 224, (224, 300)),
    ],
)
def test_downscale_caps_the_shorter_side_and_keeps_the_aspect_ratio(few_shot, size, side, expected):
    image = few_shot.downscale(Image.new("L", size, 90), side)
    assert (image.mode, image.size) == ("RGB", expected)


def test_downscale_decodes_files_like_the_library_and_keeps_small_images_exact(few_shot, tmp_path):
    pixels = np.random.default_rng(0).integers(0, 256, (20, 30, 3), dtype=np.uint8)
    path = tmp_path / "image.png"
    Image.fromarray(pixels).save(path)
    image = few_shot.downscale(path, 64)
    assert np.array_equal(np.asarray(image), pixels)
    shrunk = few_shot.downscale(path, 10)
    assert shrunk.size == (15, 10)
    # Antialiased: the shrunk image is close to an area average, not a sample of pixels.
    area = Image.fromarray(pixels).resize((15, 10), Image.Resampling.BOX)
    assert np.abs(np.asarray(shrunk, dtype=float) - np.asarray(area, dtype=float)).mean() < 40


def uniform(value, odd=None):
    """A 400x300 image of one grey, with optionally one pixel one level brighter."""
    image = Image.new("RGB", (400, 300), (value,) * 3)
    if odd is not None:
        image.putpixel(odd, (value + 1,) * 3)
    return image


def fake_loader(tmp_path):
    """Exact copies, and near copies that differ in one pixel: equal only once shrunk to 100."""
    big = tmp_path / "big.png"
    Image.new("RGB", (400, 300), (10, 20, 30)).save(big)
    copy = Image.new("RGB", (400, 300), (10, 20, 30))  # The pixels of the pool image.

    def loader(root, *, pool, evaluation, evaluate_on, seed):
        splits = {
            "pool": [(big, "a"), (Image.new("RGB", (50, 40), (1, 2, 3)), "b"), (uniform(90), "b")],
            evaluate_on: [
                (copy, "a"),
                (Image.new("RGB", (600, 200), (4, 5, 6)), "b"),
                (uniform(90, odd=(7, 7)), "b"),  # Near copy of a pool image.
                (uniform(150), "a"),
                (uniform(150, odd=(200, 100)), "a"),  # Near copy of the image before.
                (uniform(150), "a"),  # Exact repeat.
            ],
        }
        return splits, ["a", "b"]

    return loader, big


def splits_with(few_shot, monkeypatch, tmp_path, *options, drop_repeated=False):
    loader, big = fake_loader(tmp_path)
    monkeypatch.setitem(few_shot.LOADERS, "eurosat", loader)
    args = few_shot.parse_arguments([*REQUIRED, *options])
    return few_shot.load_splits(args, drop_repeated=drop_repeated), big


def test_the_near_copies_only_collide_once_shrunk(few_shot):
    for value, odd in ((90, (7, 7)), (150, (200, 100))):
        assert few_shot.digest(uniform(value)) != few_shot.digest(uniform(value, odd))
        shrunk = [few_shot.downscale(image, 100) for image in (uniform(value), uniform(value, odd))]
        assert few_shot.digest(shrunk[0]) == few_shot.digest(shrunk[1])


@pytest.mark.parametrize("side", [None, "100"])
def test_duplicates_and_repeats_are_judged_on_the_original_pixels(
    few_shot, monkeypatch, tmp_path, side
):
    options = [] if side is None else ["--max-image-side", side]
    (splits, _, duplicates, repeats), _ = splits_with(
        few_shot, monkeypatch, tmp_path, *options, drop_repeated=True
    )
    assert (duplicates, repeats) == (1, 1)  # The exact copy and the exact repeat only.
    assert len(splits["validation"]) == 4
    (_, _, _, repeats), _ = splits_with(few_shot, monkeypatch, tmp_path, *options)
    assert repeats == 0  # Not asked for.


def test_max_image_side_shrinks_every_loaded_image_once(few_shot, monkeypatch, tmp_path):
    (splits, labels, duplicates, _), _ = splits_with(
        few_shot, monkeypatch, tmp_path, "--max-image-side", "100"
    )
    assert duplicates == 1  # Judged on the original pixels, before shrinking.
    assert labels == ["a", "b"]
    sizes = {name: [image.size for image in split.images] for name, split in splits.items()}
    assert sizes == {
        "pool": [(133, 100), (50, 40), (133, 100)],
        "validation": [(300, 100), *[(133, 100)] * 4],
    }
    assert all(
        isinstance(image, Image.Image) and image.mode == "RGB"
        for split in splits.values()
        for image in split.images
    )


def test_without_max_image_side_the_loaded_items_are_unchanged(few_shot, monkeypatch, tmp_path):
    (splits, _, duplicates, _), big = splits_with(few_shot, monkeypatch, tmp_path)
    assert duplicates == 1
    assert splits["pool"].images[0] == big  # Still a path, decoded at every use.
    assert splits["validation"].images[0].size == (600, 200)


def test_max_image_side_is_part_of_the_configuration(few_shot):
    assert namespace(few_shot)["max_image_side"] is None
    assert namespace(few_shot, "--max-image-side", "224")["max_image_side"] == 224


def test_downscale_applies_the_exif_orientation_like_the_library(few_shot, tmp_path):
    path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: displayed rotated by 90 degrees.
    Image.new("RGB", (40, 20), (200, 30, 30)).save(path, exif=exif.tobytes())
    assert Image.open(path).size == (40, 20)  # As stored.
    assert few_shot.downscale(path, 100).size == (20, 40)  # As displayed.
    assert few_shot.downscale(path, 10).size == (10, 20)


@pytest.mark.parametrize(("side", "fails"), [(None, False), ("31", True), ("32", False)])
def test_max_image_side_must_reach_the_backbone_resize_edge(few_shot, side, fails, monkeypatch):
    built = []
    real = few_shot.EmbedFitModel.from_pretrained
    monkeypatch.setattr(
        few_shot.EmbedFitModel,
        "from_pretrained",
        lambda *a, **k: built.append(k) or real(*a, **k),
    )
    options = ["--backbone", "dummy/tiny", "--image-size", "32"]
    args = few_shot.parse_arguments(
        [*REQUIRED, *options, *([] if side is None else ["--max-image-side", side])]
    )
    if fails:
        with pytest.raises(SystemExit, match="--max-image-side 31 is below 32"):
            few_shot.check_image_side(args)
    else:
        few_shot.check_image_side(args)
    assert len(built) == (0 if side is None else 1)
    assert all(k["device"] == "cpu" for k in built)
