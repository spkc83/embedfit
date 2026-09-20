"""Every example must run as documented, offline, and produce sensible results."""

import importlib.util
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
SCRIPTS = sorted(path.name for path in EXAMPLES.glob("0*.py"))


def load(name: str):
    sys.path.insert(0, str(EXAMPLES))  # The examples import their shared ``_common`` module.
    try:
        spec = importlib.util.spec_from_file_location(f"example_{name[:2]}", EXAMPLES / name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXAMPLES))


def test_every_example_is_listed_in_the_index():
    index = (EXAMPLES / "README.md").read_text()
    assert len(SCRIPTS) == 6
    assert all(name in index for name in SCRIPTS)


def test_frozen_baseline(capsys):
    metrics = load("01_frozen_baseline.py").main(["--quiet"])
    assert metrics["accuracy"] > 1 / 3 and "its probabilities" in capsys.readouterr().out


def test_supcon_beats_frozen():
    results = load("02_supcon_few_shot.py").main(["--quiet"])
    assert results["supcon"] > results["frozen"] + 0.2


def test_pair_training_with_relationships(capsys):
    metrics = load("03_pair_training_with_relationships.py").main(["--quiet"])
    assert metrics["accuracy"] > 0.9 and "same:group_id" in capsys.readouterr().out


def test_retrieval():
    results = load("04_retrieval.py").main(["--quiet"])
    assert results["precision_at_5"] > 0.9 and 0.9 < results["map_at_5"] <= 1.0


def test_synthetic_document_risk(capsys):
    results = load("05_synthetic_document_risk.py").main(["--quiet"])
    assert results["pr_auc"] > 0.9 and results["dollar_recall_at_10pct_review"] > 0.9
    assert results["test_review_rate_at_threshold"] < 0.25  # The threshold stays selective.
    assert results["top_item_neighbours_risk_share"] >= 0.8
    output = capsys.readouterr().out
    assert "shared group_id value(s) between 'train' and 'test'" in output
    assert "accuracy of never flagging anything: 0.9" in output


def test_few_shot_benchmark():
    module = load("06_few_shot_benchmark.py")
    summary = module.main(["--quiet", "--epochs", "8"], shots=(4, 8), seeds=(1, 2))
    assert {(row["method"], row["shots"]) for row in summary} == {
        (method, shots) for method in module.METHODS for shots in (4, 8)
    }
    by_key = {(row["method"], row["shots"]): row["mean"] for row in summary}
    assert by_key["supcon", 8] > by_key["frozen", 8]


@pytest.mark.parametrize("name", SCRIPTS)
def test_examples_use_only_generated_data(name):
    source = (EXAMPLES / name).read_text()
    assert "make_bar_images" in source or "synthetic_item" in source
    assert "open(" not in source and "http" not in source
