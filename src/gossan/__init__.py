"""Gossan: surface clues to what's hidden underneath."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("gossan")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0"
