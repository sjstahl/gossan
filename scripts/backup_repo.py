"""Back up the Gossan repository as a full-history git bundle, outside the working tree.

A bundle is a single file holding every commit and branch; it survives the working folder (and
OneDrive) going wrong, and restores with plain git:

    git clone gossan-20261011-021500-7bb9c2d.bundle Gossan-restored

Usage:

    python scripts/backup_repo.py [DEST] [--keep N]

``DEST`` defaults to ``$GOSSAN_BACKUP_DIR`` or ``~/gossan-backups``. Each run writes
``gossan-<UTC date>-<UTC time>-<short hash>.bundle``, checks it with ``git bundle verify``, and keeps
only the newest ``--keep`` bundles (default 10). If the newest bundle already holds the current
HEAD and branches, nothing is written. Only committed work is in a bundle: commit first.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PREFIX = "gossan-"
SUFFIX = ".bundle"
DEFAULT_KEEP = 10


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)
    return done.stdout.strip()


def default_dest() -> Path:
    return Path(os.environ.get("GOSSAN_BACKUP_DIR") or Path.home() / "gossan-backups")


def refs_signature(repo: Path) -> str:
    """Every branch and tag with its commit: what a bundle of ``--all`` would hold."""
    return git(repo, "for-each-ref", "--format=%(objectname) %(refname)", "refs/heads", "refs/tags")


def bundle_signature(bundle: Path, repo: Path) -> str:
    """The branches and tags a bundle holds, in the same form as ``refs_signature``."""
    heads = git(repo, "bundle", "list-heads", str(bundle))
    lines = [ln for ln in heads.splitlines() if ln.split(" ", 1)[-1].startswith(("refs/heads/", "refs/tags/"))]
    return "\n".join(sorted(lines))


def make_backup(repo: Path, dest: Path, keep: int = DEFAULT_KEEP, now: datetime | None = None) -> Path | None:
    """Write a verified bundle of ``repo`` into ``dest``; None if the newest one is already current."""
    if keep < 1:
        raise ValueError("keep must be at least 1")
    dest.mkdir(parents=True, exist_ok=True)
    existing = sorted(dest.glob(f"{PREFIX}*{SUFFIX}"))
    wanted = "\n".join(sorted(refs_signature(repo).splitlines()))
    if existing and bundle_signature(existing[-1], repo) == wanted:
        return None
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d-%H%M%S")
    path = dest / f"{PREFIX}{stamp}-{git(repo, 'rev-parse', '--short', 'HEAD')}{SUFFIX}"
    git(repo, "bundle", "create", str(path), "--all")
    git(repo, "bundle", "verify", str(path))
    for old in sorted(dest.glob(f"{PREFIX}*{SUFFIX}"))[:-keep]:
        old.unlink()
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dest", nargs="?", type=Path, default=None, help="folder for bundles (default: $GOSSAN_BACKUP_DIR or ~/gossan-backups)")
    parser.add_argument("--keep", type=int, default=DEFAULT_KEEP, help=f"bundles to keep (default {DEFAULT_KEEP})")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent, help="repository to back up (default: this one)")
    args = parser.parse_args(argv)
    try:
        dirty = git(args.repo, "status", "--porcelain", "--untracked-files=no")
        path = make_backup(args.repo, args.dest or default_dest(), args.keep)
    except (OSError, subprocess.CalledProcessError, ValueError) as e:
        print(f"backup failed: {e}", file=sys.stderr)
        return 1
    print(f"wrote {path}" if path else "backup is already current; nothing written")
    if dirty:
        print("note: uncommitted changes are not in the bundle; commit first to include them", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
