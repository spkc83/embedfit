"""Structural guards for SECURITY_PRIVACY.md: what library code may print, log, and raise."""

import ast
from pathlib import Path

import pytest

from embedfit.data import LabelMapping
from embedfit.exceptions import ConfigurationError

SOURCES = sorted((Path(__file__).resolve().parents[2] / "src" / "embedfit").rglob("*.py"))
LOG_METHODS = {"debug", "info", "warning", "error", "exception", "critical"}


def calls(tree):
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_library_code_never_prints(path):
    names = [c.func.id for c in calls(ast.parse(path.read_text())) if isinstance(c.func, ast.Name)]
    assert "print" not in names


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_log_messages_are_constant_templates(path):
    """A constant %-template makes every logged value an explicit, reviewable argument."""
    for call in calls(ast.parse(path.read_text())):
        function = call.func
        is_log = (
            isinstance(function, ast.Attribute)
            and function.attr in LOG_METHODS
            and isinstance(function.value, ast.Name)
            and function.value.id == "logger"
        )
        if is_log:
            message = call.args[0]
            assert isinstance(message, ast.Constant) and isinstance(message.value, str), (
                f"{path.name}:{call.lineno} logs a computed message"
            )


def test_logged_arguments_are_counts_names_and_aggregates():
    """Every argument passed to a logger is reviewed here; new ones must be added on purpose."""
    allowed = {
        "len",
        "epoch",
        "best_epoch",
        "seed",
        "deterministic",
        "resolved",
        "shots",
        "method",
        "name",
        "exc",
        "split",
        "side",
        "realised",
        "fraction",
        "type",
        "train_losses",
        "eval_losses",
        "args",
        "self",
        "backbone",
        "model",
        "metric",
        "size",
        "count",
    }
    for path in SOURCES:
        for call in calls(ast.parse(path.read_text())):
            function = call.func
            if not (
                isinstance(function, ast.Attribute)
                and function.attr in LOG_METHODS
                and isinstance(function.value, ast.Name)
                and function.value.id == "logger"
            ):
                continue
            for argument in call.args[1:]:
                is_count = (
                    isinstance(argument, ast.Call)
                    and isinstance(argument.func, ast.Name)
                    and argument.func.id == "len"
                )
                is_shape = any(
                    isinstance(node, ast.Attribute) and node.attr == "shape"
                    for node in ast.walk(argument)
                )
                if is_count or is_shape:  # Lengths and array shapes describe structure only.
                    continue
                roots = {n.id for n in ast.walk(argument) if isinstance(n, ast.Name)}
                assert roots <= allowed | {"int", "float"}, (
                    f"{path.name}:{call.lineno} logs {sorted(roots - allowed)}"
                )


def test_unknown_label_error_names_the_type_not_the_value():
    mapping = LabelMapping(["legitimate", "risk"])
    with pytest.raises(ConfigurationError) as excinfo:
        mapping.encode(["legitimate", "acct-99887766"])
    message = str(excinfo.value)
    assert "99887766" not in message
    assert "type str" in message and "['legitimate', 'risk']" in message
