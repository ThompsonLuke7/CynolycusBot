#!/usr/bin/env python3
"""Publish the generated Theme Explorer to its dedicated static repository."""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Mapping

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = REPO_ROOT / "themes/dynamic_theme/viz/theme_explorer.html"
PUBLISH_REPO_URL = "git@github.com:ThompsonLuke7/thompsonluke7.github.io.git"
PUBLISH_BRANCH = "main"

PublishResult = Literal["disabled", "unchanged", "published"]


class PublishError(RuntimeError):
    """A safe, actionable publication failure."""


@dataclass(frozen=True)
class PublishConfig:
    enabled: bool
    repo_url: str
    deploy_key_path: Path | None
    git_name: str
    git_email: str
    branch: str = PUBLISH_BRANCH


def _enabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def config_from_env(environ: Mapping[str, str]) -> PublishConfig:
    enabled = _enabled(environ.get("THEME_EXPLORER_PUBLISH_ENABLED"))
    if not enabled:
        return PublishConfig(False, "", None, "", "")

    values = {
        "THEME_EXPLORER_PUBLISH_REPO": environ.get(
            "THEME_EXPLORER_PUBLISH_REPO", ""
        ).strip(),
        "THEME_EXPLORER_DEPLOY_KEY_PATH": environ.get(
            "THEME_EXPLORER_DEPLOY_KEY_PATH", ""
        ).strip(),
        "THEME_EXPLORER_GIT_NAME": environ.get("THEME_EXPLORER_GIT_NAME", "").strip(),
        "THEME_EXPLORER_GIT_EMAIL": environ.get("THEME_EXPLORER_GIT_EMAIL", "").strip(),
    }
    for name, value in values.items():
        if not value:
            raise PublishError(f"{name} is required when Theme Explorer publication is enabled")

    key_path = Path(values["THEME_EXPLORER_DEPLOY_KEY_PATH"]).expanduser()
    if not key_path.is_file() or not os.access(key_path, os.R_OK):
        raise PublishError("THEME_EXPLORER_DEPLOY_KEY_PATH is not a readable file")
    if values["THEME_EXPLORER_PUBLISH_REPO"] != PUBLISH_REPO_URL:
        raise PublishError(
            "THEME_EXPLORER_PUBLISH_REPO must be "
            "git@github.com:ThompsonLuke7/thompsonluke7.github.io.git"
        )

    branch = environ.get("THEME_EXPLORER_PUBLISH_BRANCH", PUBLISH_BRANCH).strip()
    if branch != PUBLISH_BRANCH:
        raise PublishError("THEME_EXPLORER_PUBLISH_BRANCH must be main")

    return PublishConfig(
        enabled=True,
        repo_url=values["THEME_EXPLORER_PUBLISH_REPO"],
        deploy_key_path=key_path,
        git_name=values["THEME_EXPLORER_GIT_NAME"],
        git_email=values["THEME_EXPLORER_GIT_EMAIL"],
        branch=PUBLISH_BRANCH,
    )


def validate_artifact(path: Path) -> bytes:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise PublishError(f"Theme Explorer artifact is unavailable: {path}") from exc

    required = (
        b"<title>Theme Explorer",
        b"const DATA = ",
        b"</html>",
    )
    if not content or any(marker not in content for marker in required):
        raise PublishError(f"Theme Explorer artifact failed structural validation: {path}")
    return content


def _git_env(key_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["GIT_SSH_COMMAND"] = (
        "ssh "
        f"-i {shlex.quote(str(key_path))} "
        "-o IdentitiesOnly=yes "
        "-o StrictHostKeyChecking=accept-new"
    )
    return env


def _run_git(
    args: list[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    operation: str,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=dict(env),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise PublishError(f"Git {operation} failed with exit code {result.returncode}")
    return result


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _destination_in_checkout(checkout: Path, temporary_root: Path) -> Path:
    """Return the only permitted destination after rejecting symlink escapes."""
    try:
        resolved_temporary_root = temporary_root.resolve(strict=True)
        resolved_checkout = checkout.resolve(strict=True)
    except OSError as exc:
        raise PublishError("Temporary Theme Explorer checkout is unavailable") from exc
    if checkout.is_symlink() or not _is_within(resolved_checkout, resolved_temporary_root):
        raise PublishError("Temporary Theme Explorer checkout escapes its temporary directory")

    theme_directory = checkout / "docs"
    destination = theme_directory / "index.html"
    for path in (theme_directory, destination):
        if path.is_symlink():
            raise PublishError(f"Theme Explorer destination contains a symlink: {path.name}")

    try:
        resolved_destination = destination.resolve(strict=False)
    except OSError as exc:
        raise PublishError("Theme Explorer destination cannot be resolved safely") from exc
    if not _is_within(resolved_destination, resolved_checkout):
        raise PublishError("Theme Explorer destination escapes the temporary checkout")
    return destination


def _validate_publish_config(config: PublishConfig) -> None:
    if config.branch != PUBLISH_BRANCH:
        raise PublishError("Theme Explorer publication branch must be main")
    if config.repo_url != PUBLISH_REPO_URL:
        raise PublishError(
            "Theme Explorer publication repository must be "
            "git@github.com:ThompsonLuke7/thompsonluke7.github.io.git"
        )


def publish(
    config: PublishConfig,
    artifact_path: Path = DEFAULT_ARTIFACT,
    *,
    now: datetime | None = None,
) -> PublishResult:
    if not config.enabled:
        return "disabled"
    if config.deploy_key_path is None:
        raise PublishError("Enabled publication has no deploy key")
    _validate_publish_config(config)

    artifact = validate_artifact(artifact_path)
    git_env = _git_env(config.deploy_key_path)
    timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    timestamp_text = timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")

    with tempfile.TemporaryDirectory(prefix="theme-explorer-publish-") as tmp:
        tmp_path = Path(tmp)
        checkout = tmp_path / "repository"
        _run_git(
            [
                "clone",
                "--branch",
                config.branch,
                "--single-branch",
                config.repo_url,
                str(checkout),
            ],
            cwd=tmp_path,
            env=git_env,
            operation="clone",
        )

        destination = _destination_in_checkout(checkout, tmp_path)
        if destination.exists() and destination.read_bytes() == artifact:
            return "unchanged"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination = _destination_in_checkout(checkout, tmp_path)
        destination.write_bytes(artifact)

        _run_git(
            ["config", "user.name", config.git_name],
            cwd=checkout,
            env=git_env,
            operation="identity configuration",
        )
        _run_git(
            ["config", "user.email", config.git_email],
            cwd=checkout,
            env=git_env,
            operation="identity configuration",
        )
        _run_git(
            ["add", "--", "theme-explorer/index.html"],
            cwd=checkout,
            env=git_env,
            operation="staging",
        )
        _run_git(
            ["commit", "-m", f"chore: refresh theme explorer {timestamp_text}"],
            cwd=checkout,
            env=git_env,
            operation="commit",
        )
        _run_git(
            ["push", "origin", f"HEAD:{config.branch}"],
            cwd=checkout,
            env=git_env,
            operation="push",
        )
    return "published"


def main() -> int:
    load_dotenv(REPO_ROOT / ".env", override=False)
    try:
        config = config_from_env(os.environ)
        result = publish(config)
    except PublishError as exc:
        print(f"Theme Explorer publication failed: {exc}", file=sys.stderr)
        return 1
    print(f"Theme Explorer publication: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
