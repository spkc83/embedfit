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


def test_a_file_from_before_unfreeze_last_blocks_resumes(few_shot, tmp_path):
    output = tmp_path / "results.json"
    old = {"dataset": "eurosat", "gradient_accumulation_steps": 1, "dataset_revision": None}
    write_results(output, old)
    assert few_shot.read_previous(output, {**old, "unfreeze_last_blocks": None}) is not None
    with pytest.raises(SystemExit, match="different configuration"):
        few_shot.read_previous(output, {**old, "unfreeze_last_blocks": 2})


@pytest.mark.parametrize("value", ["0", "-1", "x"])
def test_unfreeze_last_blocks_must_be_a_positive_int(few_shot, value, capsys):
    with pytest.raises(SystemExit):
        few_shot.main(["--unfreeze-last-blocks", value])
    assert "--unfreeze-last-blocks: expected a positive int" in capsys.readouterr().err
