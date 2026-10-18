"""Every Python block in README.md must run, in order, offline."""

import re
import warnings
from pathlib import Path

README = Path(__file__).resolve().parents[2] / "README.md"
FENCE = re.compile(r"^```python\n(.*?)^```", re.DOTALL | re.MULTILINE)
OFFLINE_SOURCES = re.compile(r'from_pretrained\(\s*("dummy/tiny"|"models/)')


def python_blocks() -> list[tuple[int, str]]:
    """Return each ```python block with the README line its code starts on."""
    text = README.read_text()
    return [
        (text.count("\n", 0, match.start(1)) + 1, match.group(1)) for match in FENCE.finditer(text)
    ]


def test_readme_runs_every_python_fence():
    # A fence opened as ```py, ```Python, or ```python  (trailing space) would be skipped.
    openers, inside = [], False
    for line in README.read_text().splitlines():
        if line.startswith("```"):
            if not inside:
                openers.append(line[3:].strip().lower())
            inside = not inside
    assert not inside
    assert sum(info.startswith("py") for info in openers) == len(python_blocks()) > 0


def test_readme_loads_only_offline_models():
    # A cached pretrained checkpoint would load without a network connection, so the socket
    # guard alone cannot tell that a block depends on downloaded weights.
    source = "\n".join(code for _, code in python_blocks())
    calls = source.count("from_pretrained(")
    assert calls > 0 and calls == len(OFFLINE_SOURCES.findall(source))
    assert "create_backbone(" not in source and "Backbone(" not in source


def test_readme_python_blocks_run_in_order(tmp_path, monkeypatch):
    blocks = python_blocks()
    monkeypatch.chdir(tmp_path)  # The README saves a model to a relative path.
    namespace: dict = {"__name__": "readme"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for line, code in blocks:
            # Leading newlines make tracebacks point at README line numbers.
            exec(compile("\n" * (line - 1) + code, str(README), "exec"), namespace)

    # DataLoader worker processes warn about fork(); every block that trains a representation
    # sets num_workers=0.
    assert not [w for w in caught if "use of fork()" in str(w.message)]
    assert namespace["model"].risk_score(namespace["test_images"]).shape == (32,)
    assert {row["method"] for row in namespace["summary"]} == {"frozen", "end_to_end", "supcon"}
