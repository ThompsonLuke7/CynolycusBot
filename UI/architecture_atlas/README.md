# Cynolycus System Atlas

The System Atlas is a read-only, zoomable architecture map for explaining the
repository from its six major domains down to models, feature families,
signals, policies, execution boundaries, and research evidence.

It has two views generated from one curated manifest:

- **Public** is presentation-safe and is the only build intended for hosting.
- **Local Full** adds repository evidence and internal ownership context. It is
  for local use only and must never be deployed.

Neither build imports trading modules, reads market data, calls a broker, or
uses a runtime API.

## Build and open

From the repository root:

```bash
./.venv/bin/python scripts/build_architecture_atlas.py --check
./.venv/bin/python scripts/build_architecture_atlas.py
```

The generated files are:

- `UI/architecture_atlas/dist/public/index.html`
- `UI/architecture_atlas/dist/local/index.html`

Both work when opened directly with `file://`. For browser testing through
HTTP, run:

```bash
./.venv/bin/python -m http.server 8080 --directory UI/architecture_atlas/dist/public
```

Then open `http://127.0.0.1:8080/`. Replace `public` with `local` only for a
private local review.

## Explore the atlas

- Select a node to inspect it.
- Press Enter or double-click to focus its next sublayer.
- Use Back, Escape, breadcrumbs, or browser history to move upward.
- Search jumps to any component, including deeply nested nodes.
- Filters isolate data, feature, signal, policy, execution, audit, research,
  and control relationships.
- Selecting a node reveals only its connected flow labels and fades unrelated
  components. Cross-domain links live in the Connected Systems dock instead
  of competing with the active diagram.
- At viewport widths of 2560 px or greater, the atlas automatically uses its
  large-display presentation scale. The `Aa` header control overrides the
  automatic choice and remembers that choice in the browser.
- Narrow screens use the complete outline view instead of hiding content.

Depth is variable. A branch only has as many levels as the architecture needs;
there is no requirement for every node to reach four levels.

## Update architecture content

Edit `source/architecture.json`, keeping every path repository-relative and
every claim tied to `evidence`. Then run:

```bash
./.venv/bin/python scripts/build_architecture_atlas.py --check
./.venv/bin/python -m pytest UI/tests/test_architecture_atlas_build.py UI/tests/test_architecture_atlas_content.py -q
./.venv/bin/python scripts/build_architecture_atlas.py
```

The builder rejects malformed graphs, unresolved required evidence, unsafe
Public fields, credential-like material, absolute paths, unexpected static
files, and Local-only data crossing into Public.

The generated `dist/` directory is intentionally ignored. Commit the manifest,
application sources, tests, vendored runtime, fonts, and all license files.

## Deployment boundary

Only `dist/public/` may be sent to Sites. Never upload `dist/local/`. Public
contains no Local dataset, so the Local toggle is automatically unavailable in
the hosted app.

The tracked `hosting/` directory is a minimal vinext wrapper for Sites. For a
release, copy `hosting/` into a clean staging repository, copy the validated
contents of `dist/public/` into its `public/` directory, and copy the root
`.openai/hosting.json` into the staging repository. Commit and push that exact
staging state to the Sites source repository before saving and deploying a
version. Do not place Local Full anywhere in the staging repository.
