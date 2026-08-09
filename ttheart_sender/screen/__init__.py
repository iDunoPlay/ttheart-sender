"""Screen capture and template matching."""

from __future__ import annotations

from .capture import ScreenCapture
from .matcher import MatchResult, TemplateMatcher
from .templates import Template, TemplateLibrary

__all__ = [
    "MatchResult",
    "ScreenCapture",
    "Template",
    "TemplateLibrary",
    "TemplateMatcher",
]
