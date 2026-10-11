"""scripts/backup_repo.py: verified, rotating git bundles."""

import importlib.util
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "backup_repo.py"
spec = importlib.util.spec_from_file_location("backup_repo", SCRIPT)
backup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backup)


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    (r / "a.txt").write_text("one\n")
    git(r, "add", ".")
    git(r, "commit", "-q", "-m", "first")
    return r


def test_bundle_is_written_verified_and_restores(repo, tmp_path):
    dest = tmp_path / "backups"
    path = backup.make_backup(repo, dest)
    assert path.parent == dest and path.name.startswith("gossan-") and path.suffix == ".bundle"
    restored = tmp_path / "restored"
    subprocess.run(["git", "clone", "-q", str(path), str(restored)], check=True, capture_output=True)
    assert (restored / "a.txt").read_text() == "one\n"


def test_unchanged_repo_is_not_backed_up_twice(repo, tmp_path):
    dest = tmp_path / "backups"
    assert backup.make_backup(repo, dest) is not None
    assert backup.make_backup(repo, dest) is None
    assert len(list(dest.glob("*.bundle"))) == 1
    (repo / "a.txt").write_text("two\n")
    git(repo, "commit", "-qam", "second")
    assert backup.make_backup(repo, dest, now=datetime.now(UTC) + timedelta(seconds=5)) is not None
    git(repo, "branch", "side")  # a new branch at the same commit still counts as a change
    assert backup.make_backup(repo, dest, now=datetime.now(UTC) + timedelta(seconds=10)) is not None


def test_only_the_newest_are_kept(repo, tmp_path):
    dest = tmp_path / "backups"
    t0 = datetime(2026, 10, 11, 2, 0, 0, tzinfo=UTC)
    for i in range(4):
        (repo / "a.txt").write_text(f"v{i}\n")
        git(repo, "commit", "-qam", f"c{i}")
        backup.make_backup(repo, dest, keep=2, now=t0 + timedelta(minutes=i))
    names = sorted(p.name for p in dest.glob("*.bundle"))
    assert len(names) == 2 and names[0] < names[1] and "20261011-020200" in names[0]
    with pytest.raises(ValueError):
        backup.make_backup(repo, dest, keep=0)


def test_command_line_reports_and_warns_about_uncommitted_work(repo, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("GOSSAN_BACKUP_DIR", str(tmp_path / "env-backups"))
    (repo / "a.txt").write_text("edited\n")  # tracked, uncommitted
    assert backup.main(["--repo", str(repo)]) == 0
    out = capsys.readouterr()
    assert "wrote" in out.out and "uncommitted changes are not in the bundle" in out.err
    assert list((tmp_path / "env-backups").glob("*.bundle"))
    assert backup.main(["--repo", str(tmp_path)]) == 1  # not a git repository
    assert "backup failed" in capsys.readouterr().err
