"""Exception hierarchy.

Every error the app raises on purpose derives from :class:`TTHeartError`, so the
CLI can present a clean message instead of a traceback while still letting
unexpected bugs surface normally.
"""

from __future__ import annotations


class TTHeartError(Exception):
    """Base class for all expected application errors."""


class ConfigError(TTHeartError):
    """Raised when config.yaml is malformed or contains unknown keys."""


class FlowParseError(TTHeartError):
    """Raised while loading/validating a flow definition."""


class ActionError(TTHeartError):
    """Raised when an action fails at runtime in a non-recoverable way."""


class WindowNotFoundError(TTHeartError):
    """Raised when no window matched the configured target spec."""


class TemplateNotFoundError(TTHeartError):
    """Raised when a template image cannot be resolved on disk."""


class StopRequested(TTHeartError):
    """Raised when the user pressed the stop key or Ctrl+C during a run."""


class FlowAborted(TTHeartError):
    """Raised by the ``stop`` action to end a flow early."""

    def __init__(self, message: str = "", *, success: bool = True) -> None:
        super().__init__(message)
        self.success = success
