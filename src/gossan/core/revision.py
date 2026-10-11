"""Which version of Gossan's code made an output.

``gossan.__version__`` only changes when the package is reinstalled with a new number, so it says
little about a source checkout that has moved on. Provenance therefore also records the git
revision of the source tree, so an output can be tied to the exact code that produced it:

* ``7bb9c2d1e0``          a clean checkout at that commit
* ``7bb9c2d1e0+dirty``    tracked files differ from that commit (uncommitted changes)
* ``None``                not a git checkout (an installed wheel): set ``GOSSAN_REVISION`` when
                          building one, and that value is used instead.

The lookup runs ``git`` once per process and is cached.
"""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path

ENV_VAR = "GOSSAN_REVISION"
SOURCE_DIR = Path(__file__).resolve().parent


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        done = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip()


@lru_cache(maxsize=1)
def code_revision() -> str | None:
    """The revision stamp described in the module docstring; ``GOSSAN_REVISION`` wins if set."""
    override = os.environ.get(ENV_VAR, "").strip()
    if override:
        return override
    head = _git(["rev-parse", "--short=10", "HEAD"], SOURCE_DIR)
    if not head:
        return None
    # Only files git tracks: stray untracked files (scratch notes, outputs) do not change the code.
    dirty = _git(["status", "--porcelain", "--untracked-files=no", "--", str(SOURCE_DIR.parent)], SOURCE_DIR)
    return f"{head}+dirty" if dirty else head
