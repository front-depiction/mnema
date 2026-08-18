"""mnema update: pulls the checkout the CLI runs from; fails closed elsewhere."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from mnema import cli


def _git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def _repo_with_commit(path: Path, marker: str) -> None:
    path.mkdir(parents=True)
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.email", "t@example.com")
    _git(path, "config", "user.name", "t")
    (path / marker).write_text(marker)
    _git(path, "add", "-A")
    _git(path, "commit", "-qm", marker)


def test_update_refuses_outside_a_git_checkout(tmp_path, monkeypatch):
    fake_pkg = tmp_path / "site" / "src" / "mnema"
    fake_pkg.mkdir(parents=True)
    monkeypatch.setattr(cli, "__file__", str(fake_pkg / "cli.py"))
    with pytest.raises(SystemExit) as ex:
        cli.cmd_update(SimpleNamespace())
    assert "not running from a git checkout" in str(ex.value)
    assert "uv tool install" in str(ex.value)         # the remedy is printed


def test_update_pulls_new_commits_and_reports_them(tmp_path, monkeypatch, capsys):
    origin = tmp_path / "origin"
    _repo_with_commit(origin, "one")
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(origin), str(clone))
    (clone / "src" / "mnema").mkdir(parents=True)
    monkeypatch.setattr(cli, "__file__", str(clone / "src" / "mnema" / "cli.py"))
    from mnema import serve as S
    monkeypatch.setattr(S, "running_daemons", lambda: [])

    cli.cmd_update(SimpleNamespace())
    assert "already up to date" in capsys.readouterr().out

    (origin / "two").write_text("two")
    _git(origin, "add", "-A")
    _git(origin, "commit", "-qm", "two")
    cli.cmd_update(SimpleNamespace())
    out = capsys.readouterr().out
    assert "updated" in out and "1 commits" in out and "two" in out
    assert (clone / "two").exists()                    # the pull really happened
    assert "dependencies may have changed" not in out  # pyproject untouched

    (origin / "pyproject.toml").write_text("[project]\nname='x'\n")
    _git(origin, "add", "-A")
    _git(origin, "commit", "-qm", "deps")
    cli.cmd_update(SimpleNamespace())
    assert "dependencies may have changed" in capsys.readouterr().out
