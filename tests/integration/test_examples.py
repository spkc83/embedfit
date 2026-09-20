"""Every example must run as documented, offline, and produce sensible results."""

import ast
import importlib.util
import re
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


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_synthetic_document_risk(capsys, seed):
    results = load("05_synthetic_document_risk.py").main(["--quiet", "--seed", str(seed)])
    output = capsys.readouterr().out
    # Properties of the workflow, not of one random draw.
    assert results["pr_auc"] > 0.9
    assert results["top_item_neighbours_risk_share"] >= 0.8
    assert 0 < results["test_review_rate_at_threshold"] < 0.5  # Selective, yet flags something.
    assert results["validation_fpr_at_threshold"] <= 0.05  # The target holds where it was set...
    assert 0 <= results["test_fpr_at_threshold"] <= 0.3  # ...and is reported where it was not.
    assert 0.5 < results["dollar_recall_at_10pct_review"] <= 1.0
    assert 0.8 < results["never_flag_accuracy"] < 0.95  # Rare positives: accuracy flatters.
    assert f"accuracy of never flagging anything: {results['never_flag_accuracy']:.2f}" in output
    assert f"test {results['test_fpr_at_threshold']:.3f}" in output
    assert "shared group_id value(s) between 'train' and 'test'" in output


def test_few_shot_benchmark():
    module = load("06_few_shot_benchmark.py")
    summary = module.main(["--quiet", "--epochs", "8"], shots=(4, 8), seeds=(1, 2))
    assert {(row["method"], row["shots"]) for row in summary} == {
        (method, shots) for method in module.METHODS for shots in (4, 8)
    }
    by_key = {(row["method"], row["shots"]): row["mean"] for row in summary}
    assert by_key["supcon", 8] > by_key["frozen", 8]


GENERATORS = {"make_bar_images", "synthetic_item"}
DATA_LOADERS = re.compile(
    r"^(open|.*\.open|.*\.load|.*\.loadtxt|.*\.read_\w+|.*\.read_bytes|.*\.read_text|"
    r".*load_dataset|.*urlopen|.*urlretrieve|.*\.get|.*\.fromfile|.*ImageFolder)$"
)


@pytest.mark.parametrize("name", [*SCRIPTS, "_common.py"])
def test_examples_never_read_data_from_files_or_the_network(name):
    tree = ast.parse((EXAMPLES / name).read_text())
    called = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    loaders = [call for call in called if DATA_LOADERS.match(call) and call != "defaults.get"]
    assert not loaders, f"{name} calls {loaders}"
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in ([ast.alias(node.module)] if isinstance(node, ast.ImportFrom) else node.names)
    }
    assert not imported & {"requests", "urllib", "datasets", "pandas", "torchvision", "csv"}


@pytest.mark.parametrize("name", SCRIPTS)
def test_every_example_generates_its_images(name):
    tree = ast.parse((EXAMPLES / name).read_text())
    called = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    assert called & GENERATORS  # An actual call, not a mention in a comment.
