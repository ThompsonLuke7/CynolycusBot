from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest
import scripts.publish_theme_explorer as publisher

from scripts.publish_theme_explorer import (
    PublishConfig,
    PublishError,
    config_from_env,
    publish,
    validate_artifact,
)

pytestmark = pytest.mark.safe


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def _valid_html(generated_at: str = "2026-07-22T20:57:33") -> bytes:
    return (
        "<!DOCTYPE html><html><head><title>Theme Explorer — 3D</title></head>"
        "<body><script>const DATA = "
        f'{{"generated_at":"{generated_at}","nodes":[],"links":[]}};'
        "</script></body></html>"
    ).encode("utf-8")


def _init_remote(tmp_path: Path, index: bytes | None) -> Path:
    work = tmp_path / "seed"
    remote = tmp_path / "remote.git"
    work.mkdir()
    _git(work, "init", "-b", "main")
    _git(work, "config", "user.name", "Publisher Test")
    _git(work, "config", "user.email", "publisher-test@example.invalid")
    (work / "README.md").write_text("# Theme Explorer\n", encoding="utf-8")
    (work / "index.html").write_text("<html>root site</html>\n", encoding="utf-8")
    if index is not None:
        (work / "theme-explorer").mkdir()
        (work / "theme-explorer/index.html").write_bytes(index)
    _git(work, "add", "README.md", "index.html")
    if index is not None:
        _git(work, "add", "theme-explorer/index.html")
    _git(work, "commit", "-m", "initial")
    _git(tmp_path, "init", "--bare", str(remote))
    _git(work, "remote", "add", "origin", str(remote))
    _git(work, "push", "origin", "main")
    return remote


def _config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, remote: Path
) -> PublishConfig:
    key = tmp_path / "deploy-key"
    key.write_text("unused for local repository tests\n", encoding="utf-8")
    monkeypatch.setattr(publisher, "PUBLISH_REPO_URL", str(remote))
    return PublishConfig(
        enabled=True,
        repo_url=str(remote),
        deploy_key_path=key,
        git_name="Theme Explorer Publisher",
        git_email="theme-explorer-publisher@example.invalid",
    )


def test_config_disabled_requires_no_secret_material() -> None:
    config = config_from_env({})
    assert config.enabled is False
    assert config.repo_url == ""
    assert config.deploy_key_path is None


def test_config_enabled_requires_every_field(tmp_path: Path) -> None:
    with pytest.raises(PublishError, match="THEME_EXPLORER_PUBLISH_REPO"):
        config_from_env({"THEME_EXPLORER_PUBLISH_ENABLED": "1"})

    key = tmp_path / "deploy-key"
    key.write_text("key\n", encoding="utf-8")
    with pytest.raises(PublishError, match="THEME_EXPLORER_GIT_EMAIL"):
        config_from_env(
            {
                "THEME_EXPLORER_PUBLISH_ENABLED": "1",
                "THEME_EXPLORER_PUBLISH_REPO": "git@github.com:ThompsonLuke7/thompsonluke7.github.io.git",
                "THEME_EXPLORER_DEPLOY_KEY_PATH": str(key),
                "THEME_EXPLORER_GIT_NAME": "Theme Explorer Publisher",
            }
        )


def test_config_enabled_rejects_foreign_repository(tmp_path: Path) -> None:
    key = tmp_path / "deploy-key"
    key.write_text("key\n", encoding="utf-8")

    with pytest.raises(PublishError, match="ThompsonLuke7/thompsonluke7.github.io"):
        config_from_env(
            {
                "THEME_EXPLORER_PUBLISH_ENABLED": "1",
                "THEME_EXPLORER_PUBLISH_REPO": "git@github.com:example/other.git",
                "THEME_EXPLORER_DEPLOY_KEY_PATH": str(key),
                "THEME_EXPLORER_GIT_NAME": "Theme Explorer Publisher",
                "THEME_EXPLORER_GIT_EMAIL": "publisher@example.invalid",
            }
        )


def test_config_enabled_rejects_non_main_branch(tmp_path: Path) -> None:
    key = tmp_path / "deploy-key"
    key.write_text("key\n", encoding="utf-8")

    with pytest.raises(PublishError, match="must be main"):
        config_from_env(
            {
                "THEME_EXPLORER_PUBLISH_ENABLED": "1",
                "THEME_EXPLORER_PUBLISH_REPO": "git@github.com:ThompsonLuke7/thompsonluke7.github.io.git",
                "THEME_EXPLORER_PUBLISH_BRANCH": "preview",
                "THEME_EXPLORER_DEPLOY_KEY_PATH": str(key),
                "THEME_EXPLORER_GIT_NAME": "Theme Explorer Publisher",
                "THEME_EXPLORER_GIT_EMAIL": "publisher@example.invalid",
            }
        )


def test_module_offers_no_accepted_foreign_repository_config_path(tmp_path: Path) -> None:
    artifact = tmp_path / "theme_explorer.html"
    artifact.write_bytes(_valid_html())
    key = tmp_path / "deploy-key"
    key.write_text("key\n", encoding="utf-8")

    assert not hasattr(PublishConfig, "for_test")
    assert not hasattr(publisher, "_TEST_CONFIG_TOKEN")
    with pytest.raises(PublishError, match="publication repository must be"):
        publish(
            PublishConfig(
                enabled=True,
                repo_url="file:///tmp/foreign-theme-explorer.git",
                deploy_key_path=key,
                git_name="Theme Explorer Publisher",
                git_email="publisher@example.invalid",
                branch="main",
            ),
            artifact,
        )


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"<html><title>Wrong page</title></html>",
        b"<html><title>Theme Explorer</title><script>const DATA = {};</script>",
    ],
)
def test_validate_artifact_rejects_invalid_content(
    tmp_path: Path, content: bytes
) -> None:
    artifact = tmp_path / "theme_explorer.html"
    artifact.write_bytes(content)
    with pytest.raises(PublishError, match="artifact"):
        validate_artifact(artifact)


def test_validate_artifact_returns_exact_bytes(tmp_path: Path) -> None:
    artifact = tmp_path / "theme_explorer.html"
    expected = _valid_html()
    artifact.write_bytes(expected)
    assert validate_artifact(artifact) == expected


def test_validate_artifact_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(PublishError, match="artifact"):
        validate_artifact(tmp_path / "missing.html")


def test_publish_updates_only_index_html(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = _init_remote(tmp_path, _valid_html("2026-07-21T20:57:33"))
    artifact = tmp_path / "theme_explorer.html"
    expected = _valid_html()
    artifact.write_bytes(expected)

    result = publish(
        _config(monkeypatch, tmp_path, remote),
        artifact,
        now=datetime(2026, 7, 22, 23, 30, tzinfo=timezone.utc),
    )

    assert result == "published"
    checkout = tmp_path / "verify"
    _git(tmp_path, "clone", "--branch", "main", str(remote), str(checkout))
    assert (checkout / "theme-explorer/index.html").read_bytes() == expected
    assert (checkout / "index.html").read_text(encoding="utf-8") == "<html>root site</html>\n"
    assert (checkout / "README.md").read_text(encoding="utf-8") == "# Theme Explorer\n"
    changed = _git(checkout, "show", "--pretty=", "--name-only", "HEAD").stdout.splitlines()
    assert changed == ["theme-explorer/index.html"]
    message = _git(checkout, "log", "-1", "--pretty=%s").stdout.strip()
    assert message == "chore: refresh theme explorer 2026-07-22T23:30:00Z"


def test_publish_is_noop_when_index_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _valid_html()
    remote = _init_remote(tmp_path, expected)
    artifact = tmp_path / "theme_explorer.html"
    artifact.write_bytes(expected)

    before = _git(tmp_path, "--git-dir", str(remote), "rev-list", "--count", "main").stdout
    result = publish(_config(monkeypatch, tmp_path, remote), artifact)
    after = _git(tmp_path, "--git-dir", str(remote), "rev-list", "--count", "main").stdout

    assert result == "unchanged"
    assert after == before


@pytest.mark.parametrize("link_target", ["directory", "file"])
def test_publish_rejects_symlinked_destination_without_touching_external_sentinel(
    tmp_path: Path, link_target: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = _init_remote(tmp_path, None)
    seed = tmp_path / "seed"
    sentinel = tmp_path / "external-sentinel.html"
    original = b"must not be overwritten"
    sentinel.write_bytes(original)

    theme_dir = seed / "theme-explorer"
    if link_target == "directory":
        theme_dir.symlink_to(tmp_path, target_is_directory=True)
        staged_path = "theme-explorer"
    else:
        theme_dir.mkdir()
        (theme_dir / "index.html").symlink_to(sentinel)
        staged_path = "theme-explorer/index.html"
    _git(seed, "add", staged_path)
    _git(seed, "commit", "-m", f"add symlinked {link_target} destination")
    _git(seed, "push", "origin", "main")

    artifact = tmp_path / "theme_explorer.html"
    artifact.write_bytes(_valid_html())

    with pytest.raises(PublishError, match="symlink"):
        publish(_config(monkeypatch, tmp_path, remote), artifact)

    assert sentinel.read_bytes() == original


def test_publish_disabled_does_not_validate_or_clone(tmp_path: Path) -> None:
    config = PublishConfig(
        enabled=False,
        repo_url="",
        deploy_key_path=None,
        git_name="",
        git_email="",
        branch="main",
    )
    assert publish(config, tmp_path / "missing.html") == "disabled"


def test_git_failure_does_not_expose_deploy_key_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "theme_explorer.html"
    artifact.write_bytes(_valid_html())
    key = tmp_path / "private-deploy-key"
    key.write_text("not-a-real-key\n", encoding="utf-8")
    missing_remote = tmp_path / "missing-remote.git"
    monkeypatch.setattr(publisher, "PUBLISH_REPO_URL", str(missing_remote))
    config = PublishConfig(
        enabled=True,
        repo_url=str(missing_remote),
        deploy_key_path=key,
        git_name="Theme Explorer Publisher",
        git_email="theme-explorer-publisher@example.invalid",
    )

    with pytest.raises(PublishError, match="Git clone failed") as exc_info:
        publish(config, artifact)

    assert str(key) not in str(exc_info.value)
