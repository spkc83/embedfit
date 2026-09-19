"""EmbedFit: few-shot adaptation of pretrained embedding models."""

import logging

from embedfit._version import __version__
from embedfit.exceptions import EmbedFitError

# Library code logs through the standard ``logging`` module and never configures handlers.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["EmbedFitError", "__version__"]
