"""Exception hierarchy for EmbedFit.

Messages raised by the library describe *structure* (types, shapes, counts, option names) and
never embed image contents or metadata values (see ``SECURITY_PRIVACY.md``).
"""


class EmbedFitError(Exception):
    """Base class for every error raised deliberately by EmbedFit."""


class ConfigurationError(EmbedFitError, ValueError):
    """An argument or configuration value is invalid or unsupported on this platform."""


class ImageInputError(EmbedFitError, ValueError):
    """An input image has an unsupported type, shape, or dtype."""


class BackboneProviderError(EmbedFitError, ValueError):
    """A backbone identifier is malformed, names an unknown provider, or cannot be built."""


class HeadNotFittedError(EmbedFitError, RuntimeError):
    """A prediction was requested before a classification head was fitted."""


class IndexNotFittedError(EmbedFitError, RuntimeError):
    """A search was requested before a retrieval index was fitted."""
