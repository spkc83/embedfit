"""Structural guards for SECURITY_PRIVACY.md: what library code may print, log, and raise.

The checks parse every module under ``src/embedfit``. They are deliberately strict: a new kind
of logged value, or an exception message that mentions a sample field, fails here and has to be
allowed on purpose.
"""

import ast
import logging
import traceback
from pathlib import Path

import pytest

from embedfit.data import ImageDataset, LabelMapping, to_rgb_tensor
from embedfit.exceptions import ConfigurationError, ImageInputError

SOURCES = sorted((Path(__file__).resolve().parents[2] / "src" / "embedfit").rglob("*.py"))
LOG_METHODS = {"debug", "info", "warning", "error", "exception", "critical", "log"}

# Scalars that code passes to loggers: counters, names of modes and splits, and aggregates.
LOGGED_NAMES = {
    "epoch",
    "best_epoch",
    "seed",
    "deterministic",
    "resolved",
    "method",
    "shots",
    "name",
    "exc",
    "split",
    "side",
    "realised",
    "probe_loss",
    "fraction",
    "count",
}
LOGGED_ATTRIBUTES = {
    "identifier",
    "embedding_dim",
    "training_mode",
    "num_epochs",
    "_metric",
    "augmentation",  # A registered pipeline name, like training_mode.
}
LOGGED_SEQUENCES = {"train_losses", "eval_losses"}  # Indexed for one aggregate value.
STRUCTURAL_ATTRIBUTES = {"shape", "ndim", "dtype", "size", "mode"}

# Names that hold sample content inside the library. In an exception message they may only be
# described structurally: type(x).__name__, len(x), x.shape, x.ndim, x.dtype.
SAMPLE_NAMES = {
    "sample",
    "record",
    "image",
    "images",
    "pixels",
    "pixel_values",
    "metadata",
    "embeddings",
    "features",
    "tokens",
}
SAMPLE_ATTRIBUTES = {"metadata", "image", "images", "sample_id", "group_id", "amount", "timestamp"}


def calls(tree):
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


def is_log_call(call):
    """Any ``<receiver>.info(...)``-style call, whatever the receiver is named."""
    function = call.func
    if not (isinstance(function, ast.Attribute) and function.attr in LOG_METHODS):
        return False
    receiver = ast.unparse(function.value)
    return "log" in receiver.lower()  # logger, log, logging.getLogger(...), self._logger, ...


def is_safe_log_argument(node):
    """Whether the WHOLE expression is a count, a name of a mode, or an aggregate."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Name):
        return node.id in LOGGED_NAMES
    if isinstance(node, ast.BinOp):  # epoch + 1, 100 * fraction
        return is_safe_log_argument(node.left) and is_safe_log_argument(node.right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return node.func.id == "len"  # The length of anything is a count.
    if isinstance(node, ast.Attribute):
        if node.attr == "__name__":  # type(x).__name__
            inner = node.value
            return (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id == "type"
            )
        return node.attr in LOGGED_ATTRIBUTES
    if isinstance(node, ast.Subscript):
        value = node.value
        if isinstance(value, ast.Attribute):
            return value.attr in STRUCTURAL_ATTRIBUTES  # matrix.shape[1]
        return isinstance(value, ast.Name) and value.id in LOGGED_SEQUENCES
    return False


def describes_structure_only(node):
    """type(x).__name__, len(x), x.shape, tuple(x.shape), x.ndim, x.dtype, x.shape[1] ..."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "len":
            return True
        if node.func.id in {"tuple", "list", "int", "sorted", "set"}:
            return all(describes_structure_only(argument) for argument in node.args)
    if isinstance(node, ast.Subscript):
        return describes_structure_only(node.value)
    if isinstance(node, ast.Attribute):
        if node.attr in STRUCTURAL_ATTRIBUTES or node.attr == "n_frames":
            return True
        if node.attr == "__name__":
            return True
    return False


def mentions_sample_content(node):
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in SAMPLE_NAMES:
            return True
        if isinstance(child, ast.Attribute) and child.attr in SAMPLE_ATTRIBUTES:
            return True
    return False


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_library_code_never_writes_to_the_console(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "builtins":
            pytest.fail(f"{path.name}:{node.lineno} imports from builtins")
    for call in calls(tree):
        source = ast.unparse(call.func)
        assert source != "print", f"{path.name}:{call.lineno} prints"
        assert not source.startswith(("sys.stdout", "sys.stderr")), f"{path.name}:{call.lineno}"
        assert source not in {"warnings.warn", "warn"}, (
            f"{path.name}:{call.lineno} warns; use the module logger with a constant message"
        )


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_log_messages_are_constant_templates_with_reviewed_arguments(path):
    for call in (c for c in calls(ast.parse(path.read_text())) if is_log_call(c)):
        where = f"{path.name}:{call.lineno}"
        message = call.args[0]
        assert isinstance(message, ast.Constant) and isinstance(message.value, str), (
            f"{where} logs a computed message"
        )
        assert not call.keywords or {k.arg for k in call.keywords} <= {"exc_info", "stacklevel"}
        for argument in call.args[1:]:
            assert is_safe_log_argument(argument), (
                f"{where} logs `{ast.unparse(argument)}`; only counts, mode names, and "
                "aggregates may be logged"
            )


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_exception_messages_describe_sample_content_only_by_structure(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        for formatted in (n for n in ast.walk(node.exc) if isinstance(n, ast.FormattedValue)):
            expression = formatted.value
            if mentions_sample_content(expression):
                assert describes_structure_only(expression), (
                    f"{path.name}:{node.lineno} puts `{ast.unparse(expression)}` in an "
                    "exception message; describe it by type, shape, or count instead"
                )


class TestTheAuditItself:
    """The guards above must reject the leaks they exist to prevent."""

    @pytest.mark.parametrize(
        "statement",
        [
            'logger.info("%s", self.metadata)',
            'logger.info("x %s", self.labels)',
            'logger.info("%s", sample.metadata)',
            'log.info("%s", sample.sample_id)',
            'logging.getLogger(__name__).info("%s", image)',
            'logger.info("%s", (self.metadata, x.shape))',
            'logger.info(f"label {label}")',
            'self._logger.warning("%s", args.output_dir)',
        ],
    )
    def test_unsafe_log_calls_are_recognised_and_rejected(self, statement):
        call = ast.parse(statement).body[0].value
        assert is_log_call(call)
        message, arguments = call.args[0], call.args[1:]
        constant = isinstance(message, ast.Constant)
        assert not (constant and all(is_safe_log_argument(a) for a in arguments))

    @pytest.mark.parametrize(
        "statement",
        [
            'logger.info("%d of %d", len(items), self.embedding_dim)',
            'logger.info("Epoch %d/%d: %.4f", epoch + 1, args.num_epochs, train_losses[-1])',
            'logger.debug("dim %d", matrix.shape[1])',
            'logger.info("fitting %s", type(self.model.head).__name__)',
        ],
    )
    def test_safe_log_calls_pass(self, statement):
        call = ast.parse(statement).body[0].value
        assert is_log_call(call) and all(is_safe_log_argument(a) for a in call.args[1:])

    @pytest.mark.parametrize(
        "expression, acceptable",
        [
            ("sample.metadata['acct']", False),
            ("record", False),
            ("image", False),
            ("self.metadata", False),
            ("type(image).__name__", True),
            ("tuple(pixel_values.shape)", True),
            ("len(images)", True),
            ("image.ndim", True),
        ],
    )
    def test_exception_rule(self, expression, acceptable):
        node = ast.parse(expression, mode="eval").body
        assert mentions_sample_content(node)
        assert describes_structure_only(node) is acceptable


def test_unknown_label_error_names_the_type_not_the_value():
    mapping = LabelMapping(["legitimate", "risk"])
    with pytest.raises(ConfigurationError) as excinfo:
        mapping.encode(["legitimate", "acct-99887766"])
    message = str(excinfo.value)
    assert "99887766" not in message
    assert "type str" in message and "['legitimate', 'risk']" in message


@pytest.mark.parametrize("content", [None, b"this is not an image"])
def test_image_file_paths_never_reach_a_traceback_or_a_log(tmp_path, caplog, content):
    """The cause of a file error names the path, so it must not be chained."""
    path = tmp_path / "acct-99887766-routing-0210.png"
    if content is not None:
        path.write_bytes(content)
    try:
        to_rgb_tensor(path)
    except ImageInputError as error:
        rendered = "".join(traceback.format_exception(error))
        with caplog.at_level(logging.ERROR):
            logging.getLogger("application").exception("could not encode an image")
    assert "99887766" not in rendered and "99887766" not in caplog.text
    assert "ImageInputError" in rendered


def test_skipping_unreadable_images_names_no_file_or_label(tmp_path, caplog):
    secret = tmp_path / "acct-99887766-routing-0210.png"
    secret.write_bytes(b"this is not an image")
    records = [
        {"image": secret, "label": "acct-99887766"},
        {"image": tmp_path / "acct-99887766-missing.png", "label": "acct-99887766"},
        {"image": b"acct-99887766", "label": "acct-99887766"},
    ]
    with caplog.at_level(logging.DEBUG):
        dataset = ImageDataset(records, on_corrupt_image="skip")
    assert len(dataset) == 0 and dataset.num_skipped == 3
    assert "99887766" not in caplog.text and "99887766" not in repr(dataset)
