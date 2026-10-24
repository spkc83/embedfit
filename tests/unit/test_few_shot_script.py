"""benchmarks/few_shot.py: options, resuming, and WiSE-FT interpolation."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from embedfit import EmbedFitModel, FewShotEvaluator, ImageDataset
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
    "augmentation": "none",
    "max_steps": None,
    "projection_dim": None,
    "contrastive_views": 1,
    "temperature": 0.07,
    "device": "cpu",
    "interpolate": [],
}


def write_results(path, config):
    results = {"config": config, "environment": {"sessions": []}, "records": []}
    path.write_text(json.dumps(results))


@pytest.mark.parametrize(
    ("key", "default", "other"),
    [
        ("unfreeze_last_blocks", None, 2),
        ("augmentation", "none", "natural"),
        ("max_steps", None, 100),
        ("projection_dim", None, 128),
        ("contrastive_views", 1, 2),
        ("temperature", 0.07, 0.2),
        ("device", "cpu", "cuda"),
        ("interpolate", [], [0.5]),
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
    ["--unfreeze-last-blocks", "--max-steps", "--projection-dim", "--contrastive-views"],
)
@pytest.mark.parametrize("value", ["0", "-1", "x"])
def test_count_options_must_be_positive_ints(few_shot, option, value, capsys):
    with pytest.raises(SystemExit):
        few_shot.main([option, value])
    assert f"{option}: expected a positive int" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0", "-0.1", "inf", "nan", "x"])
def test_temperature_must_be_a_positive_number(few_shot, value, capsys):
    with pytest.raises(SystemExit):
        few_shot.main(["--temperature", value])
    assert "--temperature: expected a positive number" in capsys.readouterr().err


REQUIRED = ["--dataset", "eurosat", "--evaluate-on", "validation", "--output", "results.json"]


def test_an_unknown_augmentation_is_rejected(few_shot, capsys):
    with pytest.raises(SystemExit):
        few_shot.main([*REQUIRED, "--augmentation", "sepia"])
    assert "--augmentation: invalid choice" in capsys.readouterr().err


@pytest.mark.parametrize(
    "option",
    [["--augmentation", "natural"], ["--unfreeze-last-blocks", "1"], ["--max-steps", "50"]],
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
def test_supcon_options_need_the_supcon_method(few_shot, option, capsys):
    with pytest.raises(SystemExit):
        few_shot.main([*REQUIRED, "--methods", "frozen", "pair", *option])
    assert "applies to the supcon method only" in capsys.readouterr().err


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
    assert [{k: v for k, v in r.items() if k != "alpha"} for r in records] == expected


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


def test_an_interpolated_score_does_not_depend_on_the_other_weights(run):
    # end_to_end's linear head is warm-started, so it must start from the trained head every time.
    alone = run("--methods", "end_to_end", "--interpolate", "0.5", output="alone.json")
    together = run("--methods", "end_to_end", "--interpolate", "0", "0.5", output="both.json")
    assert values(alone["records"], "end_to_end", 0.5) == values(
        together["records"], "end_to_end", 0.5
    )
    assert values(alone["records"], "end_to_end", None) == values(
        together["records"], "end_to_end", None
    )
