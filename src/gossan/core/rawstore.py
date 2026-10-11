"""Saving downloads under ``raw/`` without piling up identical copies.

``raw/`` keeps originals untouched, and every harvest used to add a new timestamped file even
when the source had not changed (about 90 MB after a handful of test runs on one area). A
download whose bytes are identical to a file already in the folder is now not written again: the
existing file is reused, and the retrieval is still logged to provenance (with
``"unchanged": true`` and the file's checksum), so the record of *when the data were checked*
stays complete while the disk holds each distinct version once.

Nothing is ever overwritten or deleted: a changed download is a new file, as before.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class StoredRaw:
    path: Path
    #: True when an identical file was already there and nothing new was written.
    unchanged: bool
    sha256: str


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def store_raw(folder: Path, stem: str, extension: str, content: bytes, retrieved_at: datetime) -> StoredRaw:
    """Save ``content`` as ``<stem>_<timestamp>.<extension>`` in ``folder`` unless an identical
    file named ``<stem>_*.<extension>`` exists (the newest such file is then returned).

    Candidate files are compared by size first, then by SHA-256, so a folder of large rasters
    costs a few stat calls, not a re-read of everything.
    """
    folder.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(content).hexdigest()
    for existing in sorted(folder.glob(f"{stem}_*.{extension}"), reverse=True):
        if existing.stat().st_size == len(content) and sha256_of(existing) == digest:
            return StoredRaw(existing, True, digest)
    path = folder / f"{stem}_{retrieved_at:%Y%m%dT%H%M%S%fZ}.{extension}"
    with path.open("xb") as f:  # "x": fail rather than overwrite an original
        f.write(content)
    return StoredRaw(path, False, digest)
