"""The code-revision stamp recorded in provenance."""

import re
import subprocess
from pathlib import Path

import pytest

from gossan.core import revision
from gossan.core.provenance import ProvenanceLog


@pytest.fixture(autouse=True)
def fresh_cache(monkeypatch):
    monkeypatch.delenv(revision.ENV_VAR, raising=False)
    revision.code_revision.cache_clear()
    yield
    revision.code_revision.cache_clear()


def test_environment_override_wins(monkeypatch):
    monkeypatch.setenv(revision.ENV_VAR, "release-2026.10")
    assert revision.code_revision() == "release-2026.10"


def test_a_git_checkout_gives_a_short_hash_and_marks_dirty_tracked_files(tmp_path, monkeypatch):
    def git(*args):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=tmp_path, check=True, capture_output=True)

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "mod.py").write_text("x = 1\n")
    git("init", "-q")
    git("add", ".")
    git("commit", "-q", "-m", "init")
    monkeypatch.setattr(revision, "SOURCE_DIR", pkg)
    clean = revision.code_revision()
    assert re.fullmatch(r"[0-9a-f]{10}", clean)

    (tmp_path / "scratch.txt").write_text("untracked notes")  # untracked files do not count
    revision.code_revision.cache_clear()
    assert revision.code_revision() == clean

    (pkg / "mod.py").write_text("x = 2\n")  # a tracked file changed
    revision.code_revision.cache_clear()
    assert revision.code_revision() == f"{clean}+dirty"


def test_outside_git_the_revision_is_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(revision, "SOURCE_DIR", tmp_path)
    monkeypatch.setattr(revision, "_git", lambda args, cwd: None)
    assert revision.code_revision() is None


def test_records_carry_the_revision_and_old_logs_still_read(tmp_path, monkeypatch):
    monkeypatch.setenv(revision.ENV_VAR, "abc123")
    revision.code_revision.cache_clear()
    log = ProvenanceLog(tmp_path)
    log.log(dataset="d", source="s", identifier="i", url=None, license="l", operation="fetch")
    assert log.read()[0].gossan_revision == "abc123"
    # A log written before this field existed has no gossan_revision key at all.
    old = Path(tmp_path / "old")
    old.mkdir()
    (old / "provenance.jsonl").write_text(
        '{"dataset":"d","source":"s","identifier":"i","url":null,"retrieved_at":"2026-10-10T22:22:54Z",'
        '"license":"l","operation":"fetch","parameters":{},"gossan_version":"0.1.0"}\n'
    )
    assert ProvenanceLog(old).read()[0].gossan_version == "0.1.0"
