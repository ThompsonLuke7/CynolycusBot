# Interactive Architecture Atlas Implementation Plan

**Goal:** Implement and publish the approved variable-depth Cynolycus System
Atlas with deterministic Public and Local Full static builds.

**Architecture:** A curated JSON manifest is validated against repository
evidence by a Python builder. The builder produces two self-contained static
artifacts from one HTML/CSS/JavaScript application: Public contains only
allowlisted presentation data; Local contains both datasets and enables the
comparison toggle.

**Tech Stack:** Python 3.12, pytest, static HTML/CSS/JavaScript, pinned vendored
Cytoscape.js, Figma for the visual contract, and Sites for the Public build.

## Global Constraints

- Do not change trading, model, feature, order, broker, or dashboard behavior.
- Do not read market datasets or import/execute trading modules during builds.
- Do not include secrets, account identifiers, order/position payloads,
  absolute workstation paths, or Local data in the Public artifact.
- Do not add a JavaScript build toolchain or runtime API.
- Generated `UI/architecture_atlas/dist/` outputs remain ignored.
- Preserve unrelated working-tree changes.

## Task 1: Build and Validation Engine

**Files:**

- Create `scripts/build_architecture_atlas.py`
- Create `UI/tests/test_architecture_atlas_build.py`
- Create `UI/tests/test_architecture_atlas_content.py`

Implement schema, graph, evidence, redaction, deterministic ordering, metadata,
and atomic-output validation using tests first. The builder accepts repository,
manifest, static-source, and output paths so tests use isolated temporary
repositories. It injects classic `window.ATLAS_DATA` before application code so
both artifacts work through `file://`.

Verification:

```bash
./.venv/bin/python -m pytest UI/tests/test_architecture_atlas_build.py UI/tests/test_architecture_atlas_content.py -q
```

## Task 2: Curated Architecture Manifest

**Files:**

- Create `UI/architecture_atlas/source/architecture.json`

Map the six root domains, all documented strategy/context modules, shared
engineering, research/evaluation, runtime boundaries, and dominant cross-links.
Momentum Expansion and Multi-Ticker Swing receive complete input → feature →
model/rank → signal → policy → execution → audit paths. Every entry includes
presentation copy, maturity/mode, evidence, and a curated position.

Verification:

```bash
./.venv/bin/python scripts/build_architecture_atlas.py --check
```

## Task 3: Static System Atlas

**Files:**

- Create `UI/architecture_atlas/static/index.html`
- Create `UI/architecture_atlas/static/atlas.css`
- Create `UI/architecture_atlas/static/atlas.js`
- Vendor `UI/architecture_atlas/static/vendor/cytoscape.min.js`
- Vendor its license and approved font assets/licenses

Implement:

- cinematic root composition and edge-flow styling;
- variable-depth focus scopes and portal dependencies;
- inspector, breadcrumb, Back/Forward, reset, search, filters, minimap, and
  copy-link behavior;
- Public/Local dataset switching with path preservation;
- keyboard navigation, focus states, reduced motion, and branded failures;
- tablet inspector and complete narrow-screen outline fallback.

Verification:

- build both artifacts;
- serve each with a local static server;
- inspect desktop and narrow-screen layouts;
- open each generated `index.html` directly.

## Task 4: Local Use and Documentation

**Files:**

- Create `UI/architecture_atlas/README.md`
- Modify `.gitignore`
- Optionally add a Hub link only if it can be done without changing dashboard
  process ownership or requiring the atlas server.

Document build, check, direct-open, local-server, dataset, evidence, update, and
security workflows. Ignore generated output but keep all sources, vendor assets,
and licenses tracked.

Verification:

```bash
git check-ignore UI/architecture_atlas/dist/public/index.html
```

## Task 5: Figma Visual Contract

Create one Figma design file from the implemented application containing:

1. root overview;
2. Specialist Strategies focus;
3. selected strategy feature/policy inspector;
4. narrow-screen outline fallback.

Record color/type/spacing tokens, node variants, inspector states, focus states,
portal nodes, and motion timing. The JSON manifest remains the semantic source.

## Task 6: Targeted Verification

Run:

```bash
./.venv/bin/python -m pytest UI/tests/test_architecture_atlas_build.py UI/tests/test_architecture_atlas_content.py -q
./.venv/bin/python -m py_compile scripts/build_architecture_atlas.py
./.venv/bin/python scripts/build_architecture_atlas.py --check
git diff --check
```

Inspect artifact inventories and scan Public for Local fields, absolute paths,
credential-like assignments, external URLs, and runtime requests. Verify
performance-size targets and representative semantic routes.

## Task 7: Public Deployment

Read `.openai/hosting.json` if present. Create one Sites project only when none
exists, push the exact validated `dist/public` source state, save a version, and
deploy that saved version. Never upload `dist/local`.

Verify the production URL loads the root graph, nested navigation, search,
filters, browser history, keyboard navigation, reduced motion, and responsive
outline without external runtime requests.

## Task 8: Handoff

Append the durable result, validation, production URL, files changed, and next
maintenance command to `LIVING_SUMMARY.md`. Report any incomplete requirement,
manual-only browser check, Sites limitation, or visual discrepancy explicitly.
