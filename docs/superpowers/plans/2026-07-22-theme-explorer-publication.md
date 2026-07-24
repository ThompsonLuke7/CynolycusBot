# Theme Explorer Public Publication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the locally generated Theme Explorer into the existing public GitHub Pages website repository after each successful refresh and expose it through a normal new-tab website link.

**Architecture:** CynolycusBot remains the sole generator and sends only the allowlisted `theme_explorer.html` artifact to `ThompsonLuke7/thompsonluke7.github.io` as `theme-explorer/index.html`. A tested Python publisher uses a repository-scoped SSH deploy key and a temporary clone; the repository's existing GitHub Pages configuration serves the nested page.

**Tech Stack:** Python 3.11+, pytest, Git CLI, SSH deploy keys, Bash, GitHub CLI, GitHub Pages, static HTML/JavaScript

## Global Constraints

- Publish only `themes/dynamic_theme/viz/theme_explorer.html`; never recursively copy a CynolycusBot directory.
- Never publish datasets, credentials, logs, Python code, model artifacts, broker state, or private configuration.
- The publisher must not use `git add -A`, force-push, rewrite public history, print credentials, or modify the CynolycusBot worktree.
- Publication is disabled by default and requires complete local configuration before any clone or push.
- Theme generation remains local; GitHub Pages serves the already-generated static artifact and does not reconstruct theme data.
- Publication failures are explicit and nonzero but remain non-trading-critical to the nightly data workflow.
- Public runtime behavior continues to use the existing pinned Three.js and 3d-force-graph unpkg dependencies.
- Automated tests use temporary local Git repositories and never contact the real public repository.

---

## File Structure

### CynolycusBot

- Create `scripts/publish_theme_explorer.py`: validate one generated artifact, clone the configured public repository into a temporary directory, update only `theme-explorer/index.html`, commit when changed, and push normally.
- Create `themes/dynamic_theme/tests/test_publish_theme_explorer.py`: unit and local-Git integration coverage for configuration, validation, publishing, and no-op behavior.
- Modify `scripts/nightly_market_data.sh`: capture the explorer build result and invoke the publisher only after a successful build.
- Create `UI/tests/test_nightly_theme_explorer_publish_hook.py`: regression coverage for the shell orchestration boundary.
- Modify `LIVING_SUMMARY.md`: record implementation and verification without secrets.

### `ThompsonLuke7/thompsonluke7.github.io`

- Create `theme-explorer/index.html`: exact generated Theme Explorer artifact.
- Preserve `README.md`, the root site, Pages configuration, workflows, assets, and every other repository path.

---

### Task 1: Build the Allowlisted Local Publisher

**Files:**

- Create: `scripts/publish_theme_explorer.py`
- Create: `themes/dynamic_theme/tests/test_publish_theme_explorer.py`

**Interfaces:**

- Consumes: `themes/dynamic_theme/viz/theme_explorer.html` and environment keys `THEME_EXPLORER_PUBLISH_ENABLED`, `THEME_EXPLORER_PUBLISH_REPO`, `THEME_EXPLORER_DEPLOY_KEY_PATH`, `THEME_EXPLORER_GIT_NAME`, and `THEME_EXPLORER_GIT_EMAIL`.
- Produces: `PublishConfig`, `PublishError`, `config_from_env(environ)`, `validate_artifact(path)`, `publish(config, artifact_path, now=None) -> Literal["disabled", "unchanged", "published"]`, and a CLI returning zero for disabled/unchanged/published or one for a safe publication error. The only remote path it owns is `theme-explorer/index.html`.

- [ ] **Step 1: Write the failing publisher tests**

Create `themes/dynamic_theme/tests/test_publish_theme_explorer.py`:

```python
from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

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


def _config(tmp_path: Path, remote: Path) -> PublishConfig:
    key = tmp_path / "deploy-key"
    key.write_text("unused for local repository tests\n", encoding="utf-8")
    return PublishConfig(
        enabled=True,
        repo_url=str(remote),
        deploy_key_path=key,
        git_name="Theme Explorer Publisher",
        git_email="theme-explorer-publisher@example.invalid",
        branch="main",
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


def test_publish_updates_only_index_html(tmp_path: Path) -> None:
    remote = _init_remote(tmp_path, _valid_html("2026-07-21T20:57:33"))
    artifact = tmp_path / "theme_explorer.html"
    expected = _valid_html()
    artifact.write_bytes(expected)

    result = publish(
        _config(tmp_path, remote),
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


def test_publish_is_noop_when_index_is_unchanged(tmp_path: Path) -> None:
    expected = _valid_html()
    remote = _init_remote(tmp_path, expected)
    artifact = tmp_path / "theme_explorer.html"
    artifact.write_bytes(expected)

    before = _git(tmp_path, "--git-dir", str(remote), "rev-list", "--count", "main").stdout
    result = publish(_config(tmp_path, remote), artifact)
    after = _git(tmp_path, "--git-dir", str(remote), "rev-list", "--count", "main").stdout

    assert result == "unchanged"
    assert after == before


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


def test_git_failure_does_not_expose_deploy_key_path(tmp_path: Path) -> None:
    artifact = tmp_path / "theme_explorer.html"
    artifact.write_bytes(_valid_html())
    key = tmp_path / "private-deploy-key"
    key.write_text("not-a-real-key\n", encoding="utf-8")
    config = PublishConfig(
        enabled=True,
        repo_url=str(tmp_path / "missing-remote.git"),
        deploy_key_path=key,
        git_name="Theme Explorer Publisher",
        git_email="theme-explorer-publisher@example.invalid",
        branch="main",
    )

    with pytest.raises(PublishError, match="Git clone failed") as exc_info:
        publish(config, artifact)

    assert str(key) not in str(exc_info.value)
```

- [ ] **Step 2: Run the tests and verify they fail because the publisher does not exist**

Run:

```bash
./.venv/bin/python -m pytest themes/dynamic_theme/tests/test_publish_theme_explorer.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'scripts.publish_theme_explorer'`.

- [ ] **Step 3: Implement the minimal publisher**

Create `scripts/publish_theme_explorer.py`:

```python
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
    branch: str = "main"


def _enabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def config_from_env(environ: Mapping[str, str]) -> PublishConfig:
    enabled = _enabled(environ.get("THEME_EXPLORER_PUBLISH_ENABLED"))
    if not enabled:
        return PublishConfig(False, "", None, "", "")

    values = {
        "THEME_EXPLORER_PUBLISH_REPO": environ.get("THEME_EXPLORER_PUBLISH_REPO", "").strip(),
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

    return PublishConfig(
        enabled=True,
        repo_url=values["THEME_EXPLORER_PUBLISH_REPO"],
        deploy_key_path=key_path,
        git_name=values["THEME_EXPLORER_GIT_NAME"],
        git_email=values["THEME_EXPLORER_GIT_EMAIL"],
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

        destination = checkout / "theme-explorer/index.html"
        if destination.exists() and destination.read_bytes() == artifact:
            return "unchanged"
        destination.parent.mkdir(parents=True, exist_ok=True)
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
```

- [ ] **Step 4: Run the publisher tests**

Run:

```bash
./.venv/bin/python -m pytest themes/dynamic_theme/tests/test_publish_theme_explorer.py -q
```

Expected: `11 passed`.

- [ ] **Step 5: Run static checks and the disabled CLI smoke test**

Run:

```bash
./.venv/bin/python -m py_compile scripts/publish_theme_explorer.py themes/dynamic_theme/tests/test_publish_theme_explorer.py
THEME_EXPLORER_PUBLISH_ENABLED=0 ./.venv/bin/python scripts/publish_theme_explorer.py
```

Expected: compilation succeeds and the CLI prints `Theme Explorer publication: disabled` without network access.

- [ ] **Step 6: Commit the publisher**

```bash
git add scripts/publish_theme_explorer.py themes/dynamic_theme/tests/test_publish_theme_explorer.py
git commit -m "feat: add theme explorer publisher"
```

Expected: the commit contains only the publisher and its tests.

---

### Task 2: Gate Nightly Publication on a Successful Explorer Build

**Files:**

- Modify: `scripts/nightly_market_data.sh:214-225`
- Create: `UI/tests/test_nightly_theme_explorer_publish_hook.py`

**Interfaces:**

- Consumes: Task 1 CLI `./.venv/bin/python scripts/publish_theme_explorer.py`, which is safe when disabled.
- Produces: logged `explorer_build_exit` and `explorer_publish_exit` values; publication runs only when the builder returns zero.

- [ ] **Step 1: Write the failing nightly orchestration test**

Create `UI/tests/test_nightly_theme_explorer_publish_hook.py`:

```python
from pathlib import Path

import pytest

pytestmark = pytest.mark.safe

REPO_ROOT = Path(__file__).resolve().parents[2]


def _theme_block() -> str:
    source = (REPO_ROOT / "scripts/nightly_market_data.sh").read_text(encoding="utf-8")
    start = source.index('echo "[$(ts)] themes — pending ticker enrichment')
    end = source.index("  # 9) Earnings enrichment", start)
    return source[start:end]


def test_nightly_captures_explorer_build_status_before_publication() -> None:
    block = _theme_block()
    builder = '"$PYTHON" -u -m themes.dynamic_theme.viz.build_theme_explorer'
    publisher = '"$PYTHON" -u scripts/publish_theme_explorer.py'

    assert builder in block
    assert "explorer_build_exit=$?" in block
    assert publisher in block
    assert block.index("explorer_build_exit=$?") < block.index(publisher)
    assert 'if [ "$explorer_build_exit" -eq 0 ]; then' in block


def test_publication_failure_is_logged_but_not_promoted_to_trading_status() -> None:
    block = _theme_block()
    assert "explorer_publish_exit=$?" in block
    assert 'echo "[$(ts)] theme explorer publication exit=$explorer_publish_exit"' in block
    assert 'STATUS="$explorer_publish_exit"' not in block
```

- [ ] **Step 2: Run the tests and verify the hook is absent**

Run:

```bash
./.venv/bin/python -m pytest UI/tests/test_nightly_theme_explorer_publish_hook.py -q
```

Expected: both tests fail because the current shell block neither captures the builder status nor calls the publisher.

- [ ] **Step 3: Replace the explorer rebuild block with a success-gated build and publish sequence**

In `scripts/nightly_market_data.sh`, replace the current `if [ "$emerging_theme_exit" -eq 0 ]` block with:

```bash
  if [ "$emerging_theme_exit" -eq 0 ]; then
    echo "[$(ts)] themes — rebuilding explorer"
    timeout --signal=TERM --kill-after=30s 1200s \
      "$PYTHON" -u -m themes.dynamic_theme.viz.build_theme_explorer
    explorer_build_exit=$?
    echo "[$(ts)] theme explorer build exit=$explorer_build_exit"

    if [ "$explorer_build_exit" -eq 0 ]; then
      echo "[$(ts)] themes — publishing explorer"
      timeout --signal=TERM --kill-after=30s 300s \
        "$PYTHON" -u scripts/publish_theme_explorer.py
      explorer_publish_exit=$?
      echo "[$(ts)] theme explorer publication exit=$explorer_publish_exit"
    else
      echo "[$(ts)] theme explorer publication skipped: build failed"
    fi
  fi
```

Do not assign `explorer_build_exit` or `explorer_publish_exit` to the trading-data `STATUS`; the emerging-theme and publication path remains best effort.

- [ ] **Step 4: Run the nightly hook tests and shell syntax check**

Run:

```bash
./.venv/bin/python -m pytest UI/tests/test_nightly_theme_explorer_publish_hook.py -q
bash -n scripts/nightly_market_data.sh
```

Expected: `2 passed` and `bash -n` exits zero.

- [ ] **Step 5: Re-run the publisher tests with the hook tests**

Run:

```bash
./.venv/bin/python -m pytest themes/dynamic_theme/tests/test_publish_theme_explorer.py UI/tests/test_nightly_theme_explorer_publish_hook.py -q
```

Expected: `13 passed`.

- [ ] **Step 6: Commit the nightly hook**

```bash
git add scripts/nightly_market_data.sh UI/tests/test_nightly_theme_explorer_publish_hook.py
git commit -m "feat: publish refreshed theme explorer nightly"
```

Expected: the commit contains only the nightly hook and its regression test.

---

### Task 3: Configure the Existing Pages Repository and Prove End-to-End Refresh

**Files:**

- Create outside repositories: `/home/luket/.ssh/cynolycus_theme_explorer_ed25519`
- Modify ignored local file: `.env`
- Modify: `LIVING_SUMMARY.md`
- Create remotely: `ThompsonLuke7/thompsonluke7.github.io/theme-explorer/index.html`

**Interfaces:**

- Consumes: Tasks 1–2, authenticated GitHub access, the existing public
  `ThompsonLuke7/thompsonluke7.github.io` repository, and local theme inputs.
- Produces: unattended local write access limited to the public repository, a successful publisher-created refresh commit, a successful Pages deployment, and the website link handoff.

- [ ] **Step 1: Confirm the existing repository boundary before mutation**

Use the GitHub connector or authenticated GitHub CLI to verify:

```bash
gh repo view ThompsonLuke7/thompsonluke7.github.io --json nameWithOwner,visibility,defaultBranchRef
gh api repos/ThompsonLuke7/thompsonluke7.github.io/contents/README.md --jq .path
```

Expected: the repository is `PUBLIC`, its default branch is `main`, and
`README.md` exists. Confirm that `theme-explorer/index.html` does not yet exist.
Do not replace or delete `README.md`, root `index.html`, workflows, Pages
configuration, assets, or any other existing path.

- [ ] **Step 2: Generate a dedicated passwordless Ed25519 deploy key**

Confirm the exact key paths do not already exist:

```bash
test ! -e /home/luket/.ssh/cynolycus_theme_explorer_ed25519
test ! -e /home/luket/.ssh/cynolycus_theme_explorer_ed25519.pub
```

Expected: both checks exit zero. Then run with approval to write outside the repository:

```bash
ssh-keygen -t ed25519 -C "CynolycusBot Theme Explorer publisher" -N "" -f /home/luket/.ssh/cynolycus_theme_explorer_ed25519
chmod 600 /home/luket/.ssh/cynolycus_theme_explorer_ed25519
chmod 644 /home/luket/.ssh/cynolycus_theme_explorer_ed25519.pub
```

Expected: exactly one private/public key pair is created at the specified paths. Never print the private key.

- [ ] **Step 3: Register only the public key with write access to the destination repository**

Run:

```bash
gh repo deploy-key add /home/luket/.ssh/cynolycus_theme_explorer_ed25519.pub --allow-write --title "CynolycusBot Theme Explorer publisher" --repo ThompsonLuke7/thompsonluke7.github.io
gh repo deploy-key list --repo ThompsonLuke7/thompsonluke7.github.io --json title,readOnly --jq '.[] | select(.title == "CynolycusBot Theme Explorer publisher")'
```

Expected:

```json
{"readOnly":false,"title":"CynolycusBot Theme Explorer publisher"}
```

- [ ] **Step 4: Write only the publisher settings into the ignored `.env`**

Use the installed `python-dotenv` CLI so existing secret values are neither printed nor rewritten manually:

```bash
./.venv/bin/dotenv -f .env set THEME_EXPLORER_PUBLISH_REPO git@github.com:ThompsonLuke7/thompsonluke7.github.io.git
./.venv/bin/dotenv -f .env set THEME_EXPLORER_DEPLOY_KEY_PATH /home/luket/.ssh/cynolycus_theme_explorer_ed25519
./.venv/bin/dotenv -f .env set THEME_EXPLORER_GIT_NAME "CynolycusBot Theme Explorer Publisher"
./.venv/bin/dotenv -f .env set THEME_EXPLORER_GIT_EMAIL ThompsonLuke7@users.noreply.github.com
./.venv/bin/dotenv -f .env set THEME_EXPLORER_PUBLISH_ENABLED 1
```

Verify only the key names, not the complete `.env`:

```bash
./.venv/bin/dotenv -f .env get THEME_EXPLORER_PUBLISH_ENABLED
./.venv/bin/dotenv -f .env get THEME_EXPLORER_PUBLISH_REPO
./.venv/bin/dotenv -f .env get THEME_EXPLORER_DEPLOY_KEY_PATH
```

Expected: `1`, the destination SSH URL, and the dedicated key path.

- [ ] **Step 5: Rebuild the explorer and publish through the deploy key**

Run:

```bash
./.venv/bin/python -m themes.dynamic_theme.viz.build_theme_explorer
./.venv/bin/python scripts/publish_theme_explorer.py
```

Expected: the builder reports current theme/link/ticker counts and the publisher prints `Theme Explorer publication: published`.

- [ ] **Step 6: Confirm the remote commit changed only the nested explorer file**

Run:

```bash
gh api repos/ThompsonLuke7/thompsonluke7.github.io/commits/main --jq '.commit.message'
gh api repos/ThompsonLuke7/thompsonluke7.github.io/commits/main --jq '.files[].filename'
```

Expected: a message beginning with `chore: refresh theme explorer` and exactly:

```text
theme-explorer/index.html
```

- [ ] **Step 7: Verify Pages serves the publisher commit**

Run:

```bash
curl -sS --retry 6 --retry-delay 10 https://thompsonluke7.github.io/theme-explorer/ | rg -m1 '"generated_at":"'
```

Expected: the public response contains the newly embedded generation timestamp.

- [ ] **Step 8: Verify the unchanged-artifact no-op path**

Run without rebuilding:

```bash
./.venv/bin/python scripts/publish_theme_explorer.py
```

Expected: `Theme Explorer publication: unchanged` and no new public commit or Pages run.

- [ ] **Step 9: Run the complete focused verification**

Run:

```bash
./.venv/bin/python -m pytest themes/dynamic_theme/tests/test_publish_theme_explorer.py UI/tests/test_nightly_theme_explorer_publish_hook.py -q
./.venv/bin/python -m py_compile scripts/publish_theme_explorer.py themes/dynamic_theme/tests/test_publish_theme_explorer.py UI/tests/test_nightly_theme_explorer_publish_hook.py
bash -n scripts/nightly_market_data.sh
git diff --check
```

Expected: `13 passed`; compilation, shell syntax, and whitespace checks all succeed.

- [ ] **Step 10: Append the durable handoff and commit only that update if it is not already part of another active session**

Capture the real timestamp first:

```bash
date '+%Y-%m-%d %H:%M %Z'
```

Append a maximum-three-line entry to `LIVING_SUMMARY.md`. The first line uses
the exact timestamp returned above inside braces, followed by
`{agent: Codex} {theme explorer public publishing}`. Use these exact two
narrative lines:

```text
Published Theme Explorer under the existing GitHub Pages website with a repository-scoped deploy-key publisher; nightly publication now runs only after a successful local explorer build.
Focused publisher/nightly tests, shell syntax, first deploy, public URL, and unchanged no-op verified; website link target is https://thompsonluke7.github.io/theme-explorer/.
```

Preserve unrelated working-tree changes and stage only `LIVING_SUMMARY.md` if
it contains no concurrent uncommitted entries; otherwise leave the appended
handoff uncommitted and report that fact.

- [ ] **Step 11: Provide the website button**

Return this exact integration snippet:

```html
<a
  href="https://thompsonluke7.github.io/theme-explorer/"
  target="_blank"
  rel="noopener noreferrer"
  class="theme-explorer-button"
>
  Open Theme Explorer
</a>
```

State that no website JavaScript, iframe, CORS configuration, or data synchronization is required.

---

## Final Completion Gate

Before claiming completion, verify all of the following:

- The public repository visibility is `PUBLIC`.
- The publisher-created application surface is limited to
  `theme-explorer/index.html`; existing website files remain unchanged.
- The deploy key is write-enabled only on
  `ThompsonLuke7/thompsonluke7.github.io`.
- No private key, token, `.env`, dataset, log, model, or Python file appears in the public repository.
- A successful local rebuild produces a publisher commit changing only
  `theme-explorer/index.html`.
- A failed or disabled publisher does not modify the destination.
- GitHub Pages serves the latest embedded `generated_at`.
- The second unchanged publish creates no commit.
- Focused tests, Python compilation, shell syntax, and `git diff --check` pass.
- Existing unrelated CynolycusBot working-tree changes remain untouched.
