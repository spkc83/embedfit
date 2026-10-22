"""benchmarks/few_shot.py: resuming from result files of earlier versions."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks" / "few_shot.py"


@pytest.fixture(scope="module")
def few_shot():
    spec = importlib.util.spec_from_file_location("few_shot_benchmark", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    ],
)
def test_a_file_from_before_an_option_resumes_with_its_default(
    few_shot, tmp_path, key, default, other
):
    output = tmp_path / "results.json"
    old = {"dataset": "eurosat", "gradient_accumulation_steps": 1, "dataset_revision": None}
    current = {
        **old,
        "unfreeze_last_blocks": None,
        "augmentation": "none",
        "max_steps": None,
        "projection_dim": None,
        "contrastive_views": 1,
        "temperature": 0.07,
        "device": "cpu",
    }
    write_results(output, old)
    assert few_shot.read_previous(output, current) is not None
    with pytest.raises(SystemExit, match="different configuration"):
        few_shot.read_previous(output, {**current, key: other})


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


def test_the_device_is_part_of_the_configuration_and_cudnn_is_not(few_shot):
    config = namespace(few_shot, "--device", "cuda", "--disable-cudnn")
    assert config["device"] == "cuda"
    assert "disable_cudnn" not in config
    assert namespace(few_shot, "--device", "cuda") == config
