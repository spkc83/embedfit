"""Shared command-line options for the examples.

Every example runs offline in seconds on the ``dummy/tiny`` backbone, a small untrained CNN.
Pass ``--backbone torchvision/resnet18`` (with ``pip install 'embedfit[torchvision]'``) to run
the same code on a pretrained encoder; representation training then wants a much smaller
learning rate, e.g. ``--learning-rate 2e-5``.
"""

import argparse
import logging


def parse_args(
    description: str, argv: list[str] | None = None, **defaults: float | int
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--backbone", default="dummy/tiny", help="<provider>/<name> identifier")
    parser.add_argument("--epochs", type=int, default=defaults.get("epochs", 8))
    parser.add_argument("--learning-rate", type=float, default=defaults.get("learning_rate", 1e-2))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true", help="hide EmbedFit's log messages")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s")
    return args
