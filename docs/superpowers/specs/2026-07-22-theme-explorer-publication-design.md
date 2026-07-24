# Theme Explorer Public Publication Design

## Goal

Publish the generated Theme Explorer as a standalone public page inside the
existing GitHub Pages website repository, keep it synchronized with successful
local refreshes, and let another website page open it in a new browser tab
through a normal link.

The publication path must expose only the intentionally public browser artifact.
It must not copy source datasets, credentials, logs, Python pipeline code, model
artifacts, or private configuration into the public repository.

## Current State

`themes/dynamic_theme/viz/build_theme_explorer.py` reads the latest dynamic-theme
registries, relationships, ticker memberships, pending themes, and selected
market-cap data. It writes
`themes/dynamic_theme/viz/theme_explorer.html`.

The generated file is approximately 1.23 MB. Its graph data is embedded inline,
so no application server or Python runtime is required. It is not completely
offline: at runtime it loads pinned versions of Three.js and 3d-force-graph from
unpkg.

`scripts/nightly_market_data.sh` already rebuilds the explorer after a successful
weekday emerging-theme refresh. The build runs on the local CynolycusBot machine
because the source data is local and is not suitable for reconstruction in a
public GitHub Actions runner.

## Chosen Architecture

Use two repositories with one-way publication:

```text
CynolycusBot private/local pipeline
    |
    | successful data refresh
    v
build_theme_explorer.py
    |
    | allowlisted artifact only
    v
theme_explorer.html
    |
    | repository-scoped authenticated push
    v
ThompsonLuke7/thompsonluke7.github.io (public)
    |
    | existing GitHub Pages publication
    v
https://thompsonluke7.github.io/theme-explorer/
    ^
    | ordinary link opened in a new tab
main website
```

The public repository is a deployment target, not a second implementation of
the theme pipeline. CynolycusBot remains the single owner of explorer generation.

## Public Repository

Use the existing public repository `ThompsonLuke7/thompsonluke7.github.io` with
`main` as its default branch. The repository is already connected to GitHub
Pages. Preserve all existing and future website files; the publisher owns only:

```text
theme-explorer/index.html
```

`theme-explorer/index.html` is the published copy of
`themes/dynamic_theme/viz/theme_explorer.html`. The publisher does not create,
replace, stage, or delete the repository's root `index.html`, README, Pages
configuration, workflows, assets, or other website paths.

The default public URL is
`https://thompsonluke7.github.io/theme-explorer/`. A custom domain can be added
later without changing the publication design.

## Publisher in CynolycusBot

Add `scripts/publish_theme_explorer.py` as a focused publisher owned by
CynolycusBot. Python is used so validation, subprocess boundaries, temporary
directory cleanup, and failure behavior can be covered by the existing test
stack. The publisher performs the following steps:

1. Require publication to be explicitly enabled through local configuration.
2. Resolve exactly one source file:
   `themes/dynamic_theme/viz/theme_explorer.html`.
3. Reject a missing, empty, or structurally invalid artifact. At minimum, the
   document must contain the Theme Explorer title, embedded `DATA` payload, and
   closing HTML element.
4. Clone the public repository into a temporary directory using its dedicated
   write credential.
5. Copy the allowlisted source file to `theme-explorer/index.html`, creating
   only the `theme-explorer` directory when it does not yet exist.
6. Exit successfully without a commit when the destination is byte-for-byte
   unchanged.
7. Stage only `theme-explorer/index.html`, create a timestamped refresh commit,
   and push normally to `main`.
8. Remove the temporary clone automatically.

The publisher must never recursively copy a CynolycusBot directory, run
`git add -A`, force-push, rewrite public history, print credentials, or operate
inside the dirty CynolycusBot working tree. A failed or rejected push leaves both
repositories unchanged and returns a nonzero status.

## Authentication and Configuration

Use a dedicated GitHub deploy key with write access to only
`ThompsonLuke7/thompsonluke7.github.io`. Store its private key outside both repositories.
The matching public key is registered only on the destination repository.

Local, ignored configuration supplies:

- `THEME_EXPLORER_PUBLISH_ENABLED`, defaulting to `0` until setup is complete;
- `THEME_EXPLORER_PUBLISH_REPO`, set to the destination SSH repository URL;
- `THEME_EXPLORER_DEPLOY_KEY_PATH`, set to the dedicated private-key path;
- `THEME_EXPLORER_GIT_NAME` and `THEME_EXPLORER_GIT_EMAIL`, set to the
  publisher commit identity.

The Python publisher loads these values from the process environment and the
existing ignored `.env` file. Enabling publication without a repository URL,
readable deploy key, or commit identity is a configuration error and fails
before cloning.

No token or private key is committed to either repository. The publisher passes
the selected identity to Git only for the temporary clone and push. This keeps a
compromised publication credential from writing to CynolycusBot or unrelated
repositories.

## Refresh and Publication Flow

In `scripts/nightly_market_data.sh`, capture the explorer builder's exit code
instead of treating it as an unchecked command. Run the publisher only when:

- emerging-theme enrichment succeeded;
- explorer generation succeeded; and
- publication is enabled.

The normal weekday sequence becomes:

```text
emerging-theme refresh
    -> rebuild explorer
        -> validate generated artifact
            -> publish public theme-explorer/index.html
                -> GitHub Pages deploy
```

The publisher's output and exit code are written to the existing nightly log.
Publication is non-trading-critical: a failure must not undo theme data or block
the remaining earnings enrichment. It must nevertheless be reported explicitly
as a publication failure, including enough context to retry without exposing the
credential.

The same publisher can be run manually after a weekly taxonomy refresh or a
visual/template change. Manual publication uses the identical validation and
push path through:

```bash
./.venv/bin/python scripts/publish_theme_explorer.py
```

There is no separate upload procedure.

## Website Integration

The main website does not fetch, inject, or embed the remote HTML. It links to
the Pages URL:

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

The website may style this anchor as a button. Because the explorer owns its own
page and origin, the parent website needs no CORS configuration, iframe sizing,
application JavaScript, or awareness of the explorer's data format.

## Failure Handling and Recovery

- If theme enrichment fails, retain the last published explorer and skip build
  and publication.
- If explorer generation fails, retain the last published explorer and do not
  push a partial file.
- If validation fails, refuse publication.
- If authentication, clone, commit, or push fails, leave the existing Pages
  deployment untouched and return a nonzero publisher status.
- If GitHub Pages publication fails after the source push, the committed artifact
  remains available for diagnosis and a normal retry.
- If a bad but valid artifact is published, revert the corresponding commit in
  the public repository. Normal Git history is the rollback mechanism.
- The next successful nightly or manual run retries from a fresh clone, so no
  persistent publication checkout needs repair.

## Testing and Verification

Add focused automated coverage in CynolycusBot for:

- missing and structurally invalid source artifacts;
- exact mapping from the allowlisted source to `theme-explorer/index.html`;
- unchanged-content no-op behavior;
- staging only `theme-explorer/index.html`;
- subprocess failure propagation without credential output;
- nightly publication occurring only after a successful build;
- publication being skipped when disabled or when upstream generation fails.

Use temporary local Git repositories in tests; automated tests must never access
or modify the real public repository.

Before enabling unattended publication:

1. Build the current explorer locally and verify its embedded generation time,
   theme count, link count, ticker count, search, graph navigation, and external
   library loading.
2. Run the publisher once manually and inspect the exact public commit.
3. Verify the existing Pages publication serves the expected explorer URL.
4. Open the Pages URL from the website button and confirm it opens in a new tab.
5. Run a second unchanged-artifact publication test and confirm it creates no
   commit.
6. Run the relevant unit tests and a shell syntax check for the modified nightly
   orchestration.

## Scope Boundaries

This work does not:

- move theme computation into the public repository;
- publish parquet, CSV, JSON, model, log, or credential files;
- expose a live trading API or broker state;
- add user authentication;
- embed the explorer in the main website;
- create a second source of truth for explorer code;
- change theme-generation logic or trading behavior;
- vendor the two existing CDN JavaScript dependencies.

The publication surface is intentionally limited to one replaceable artifact,
`theme-explorer/index.html`, inside the existing website repository.
