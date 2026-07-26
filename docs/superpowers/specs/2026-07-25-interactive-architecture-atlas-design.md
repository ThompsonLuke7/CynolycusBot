# Interactive Architecture Atlas Design

## Goal

Build a polished, static, interactive representation of CynolycusBot that lets
an unfamiliar viewer understand the system at a glance and recursively zoom
into its architecture, feature pipelines, specialist models, order policies,
execution boundaries, research evidence, and operational controls.

The result is one application codebase with two data products:

- **Public:** presentation-safe architecture, explanations, maturity, feature
  families, policy concepts, and safe repository references.
- **Local Full:** everything in Public plus exact internal paths, runtime
  ownership, schedules, state and ledger relationships, recovery paths, and
  broker-boundary details.

Both products exclude secrets, credentials, account identifiers, current
positions, raw private data, and absolute workstation paths.

## Audience and Success Criteria

The primary audience is a technical or academically curious viewer who has not
worked in the repository. The same page must remain useful to the author as a
technical orientation tool.

The atlas succeeds when a first-time viewer can answer:

1. What enters the system?
2. How is data made causal and model-ready?
3. Which specialist strategies exist, and how mature are they?
4. Which features and signals feed each strategy?
5. How does each strategy convert a score or rule into an order policy?
6. Which paths are research-only, paper-only, or broker-integrated?
7. How are readiness, execution, reconciliation, audit, state, and dashboards
   shared?
8. Where in the repository is each claim grounded?

“Entire repository” means complete coverage of architecture-relevant modules
and dependencies, not a visualization of every experiment output, cached bar,
log, image, or individual source file. Legacy and experimental branches remain
discoverable but can be collapsed.

## Chosen Experience

Use a **cinematic, zoomable 2D System Atlas** rather than a directory tree,
linear slide deck, or decorative constellation.

The root view presents six semantic domains:

1. Market and Context Inputs
2. Feature and Data Fabric
3. Context Intelligence
4. Specialist Strategies
5. Policy and Execution
6. Research and Control

Animated routes communicate the dominant flow:

```text
inputs
  -> time alignment, features, readiness
  -> contextual and strategy intelligence
  -> strategy-specific policy and shared execution
  -> broker boundary, audit, state, and evaluation
```

Clicking a node selects it and opens its inspector. Activating an expandable
node performs a camera transition into its children. Neighboring or external
dependencies remain visible as subdued “portal” nodes so the viewer does not
lose context.

### Variable Depth

There is no global “level 1 of 4” and no requirement that every branch have the
same depth.

- The breadcrumb displays the current semantic path, for example:
  `System / Specialist Strategies / Momentum Expansion / Feature Engineering`.
- A node with children can zoom deeper.
- A leaf opens its inspector without another camera transition.
- Back, Escape, and any breadcrumb ancestor return upward.
- Branches normally use two to five meaningful layers. Validation warns when a
  branch exceeds five layers because deeper nesting becomes difficult to
  explain, but it does not fabricate layers to make branches equal.

Examples:

```text
System
  / Specialist Strategies
    / Momentum Expansion
      / Feature Engineering
        / Trend and momentum
```

```text
System
  / Policy and Execution
    / Broker Reconciliation
```

## Information Architecture

### Major Architecture Branches

The curated map must cover:

- external market, broker, options, news, event, regulatory, macro, social, and
  stored-artifact inputs;
- shared calendars, universes, completed bars, caches, data readiness, job
  guards, feature construction, and audit utilities;
- news, catalysts, scheduled events, forward guidance, dynamic themes, legacy
  themes, and social attention;
- SPY Intraday, Multi-Ticker Swing, Momentum Expansion, HTF Swing, Meta Ranker,
  Dealer Positioning, Dealer Ranker, and Intraday Structure;
- strategy-specific and shared entry gates, sizing, routing, stop, target,
  scale-out, trailing, time-exit, reconciliation, and audit behavior;
- feature, label, training, out-of-fold, frozen-test, backtest, live inference,
  dashboard, scheduler, persistence, and recovery boundaries.

Each active strategy receives a consistent conceptual substructure:

```text
Inputs
  -> Feature Engineering
  -> Model / Rules / Ranking
  -> Signal Selection
  -> Order Policy
  -> Execution and Reconciliation
  -> Audit and Evaluation
```

The contents remain strategy-specific. The atlas must not imply that SPY,
Swing, the 4-hour rankers, Dealer Ranker, and Intraday Structure use the same
model or order policy.

### Feature Views

Feature Engineering expands into explainable families before individual
features:

- trend and momentum;
- volatility and ATR;
- volume, dollar liquidity, and tradability;
- relative strength and cross-sectional ranks;
- distance from highs, structure, and location;
- regime and benchmark context;
- lagged daily and weekly context;
- earnings, news, events, themes, social, and dealer context;
- identity or metadata features where relevant.

The inspector shows which downstream models consume a family, its decision
timeframe, whether it is shared or strategy-specific, and its causal-availability
rule. Individual feature names are shown only where they aid explanation.

### Relationship Types

Every edge has one declared type:

- `data`
- `feature`
- `signal`
- `policy`
- `execution`
- `audit`
- `research`
- `control`

Color, animation, legend, and filtering use these types consistently. Edge
direction always reflects information or control flow, not folder containment.

## Visual System

The approved direction is a high-technology “systems atlas,” not the existing
dashboard theme and not a generic card grid.

### Design Tokens

- Background: `#030711`
- Raised surface: `#081421`
- Structural line: `#1D3B56`
- Primary text: `#EAF8FF`
- Muted text: `#7190A8`
- Data/input blue: `#5498FF`
- Shared-engineering green: `#52EBA0`
- Context-intelligence violet: `#AF79FF`
- Strategy gold: `#FFC95C`
- Policy/execution rose: `#FF6FA9`
- Active/focus cyan: `#46F3FF`

Use Space Grotesk for display text and JetBrains Mono for telemetry labels,
vendored as WOFF2 with their license files. System fallbacks remain available.

### Layout and Components

- 68–72 px translucent command header with brand, breadcrumb, search, dataset
  selector, and help.
- 48–52 px floating tool rail for overview, search, filters, minimap, and
  settings.
- Full remaining viewport for the graph.
- 340–380 px frosted inspector drawer, closed by default.
- Bottom-left validation/freshness readout.
- Bottom-right minimap when the active graph exceeds the viewport.
- Compact portal nodes for dependencies outside the active branch.

Nodes use restrained illumination, precise borders, small telemetry labels,
and clear hierarchy. Persistent bloom, excessive glass, and illegible neon are
out of scope.

### Motion

- Node hover/focus: 160–200 ms.
- Inspector open/close: 240–320 ms.
- Branch camera transition: 420–560 ms with a smooth ease-in-out curve.
- Route-flow animation: subtle 7–9 second loop.
- Dataset switch: 200–300 ms crossfade without resetting the selected path when
  the node exists in both datasets.

`prefers-reduced-motion` disables camera interpolation, route animation,
parallax, and decorative rotation while preserving all navigation.

### Responsive Behavior

Desktop is the primary canvas experience. Tablet retains pan, zoom, breadcrumb,
and a bottom-sheet inspector. On narrow mobile screens, the graph becomes a
searchable hierarchical outline with the same data and inspectors rather than
shrinking the canvas into unreadability.

## Interaction Model

- Single click or keyboard focus selects a node and opens the inspector.
- Double-click, Enter on an expandable node, or the inspector’s “Enter domain”
  action zooms into children.
- Mouse wheel/pinch pans and zooms within conservative limits.
- Search returns nodes across the full dataset and can jump directly to a node,
  reconstructing its breadcrumb.
- Filters can hide edge types, maturity states, or research/operational layers.
- “Reset view” returns to the root without changing the dataset.
- Browser history records semantic paths so Back and Forward work.
- Copy Link creates a route to the current public-safe node. Local-only routes
  remain local and are never generated as public URLs.
- Source links in Local Full use repository-relative paths; public source links
  are included only when intentionally allowlisted.

## Hybrid Source of Truth

The graph is manually curated for meaning and automatically validated against
the repository. Static analysis assists verification; it does not decide the
architecture.

Use one tracked source manifest:

`UI/architecture_atlas/source/architecture.json`

### Node Shape

Each node defines:

```json
{
  "id": "strategy.momentum.features",
  "parent_id": "strategy.momentum",
  "kind": "feature",
  "visibility": "public",
  "edge_color_role": "feature",
  "position": {"x": 420, "y": 180},
  "public": {
    "label": "Feature Engineering",
    "summary": "Causal 4-hour and lagged higher-timeframe context.",
    "maturity": "research-ready",
    "mode": "paper-capable",
    "repo_paths": ["strategies/momentum_expansion/features"]
  },
  "local": {
    "details": "Exact matrix, readiness, and runtime ownership.",
    "repo_paths": ["Data/shared/bars/4h"],
    "runtime_owner": "nightly data-readiness pipeline"
  },
  "evidence": {
    "required_paths": ["strategies/momentum_expansion"],
    "symbols": []
  }
}
```

Fields under `public` can enter both builds. Fields under `local` can enter only
the Local Full build. A node with `visibility: "local"` is omitted entirely
from the Public build.

Edges define `id`, `source`, `target`, `type`, `visibility`, optional public and
local labels, layout hints, and evidence.

### Automated Validation

`scripts/build_architecture_atlas.py` must:

1. validate the manifest schema and schema version;
2. reject duplicate IDs, missing parents, cycles, unreachable nodes, invalid
   edge endpoints, unknown enum values, and malformed positions;
3. confirm required repository-relative paths exist;
4. validate declared Python class/function symbols through `ast` where
   specified;
5. validate explicitly declared Python import relationships through `ast`;
6. validate declared shell/config/text evidence through exact bounded patterns;
7. warn, rather than infer, when a dynamic dependency cannot be proven;
8. produce deterministic node and edge ordering;
9. record build time, Git revision, source-manifest hash, and validation counts;
10. build into a temporary directory and atomically replace a successful
    output;
11. fail without altering the last valid outputs.

The validator never imports trading modules, runs pipelines, loads market data,
starts dashboards, calls APIs, or submits orders.

## Public and Local Build Boundary

There is one application source, but two static build artifacts:

```text
UI/architecture_atlas/dist/public/
UI/architecture_atlas/dist/local/
```

The Local build embeds both Public and Local Full datasets and displays the
toggle. The Public build embeds only the Public dataset and hides the toggle.
Hiding Local Full through CSS is not an acceptable security boundary.

Both builds use classic relative script assets and embedded JavaScript data so
they work from a static host and by directly opening `index.html`; they do not
fetch local JSON through `file://`.

The Public builder uses an allowlist of fields, then rejects:

- absolute POSIX or Windows workstation paths;
- account identifiers or position/order payloads;
- credential values or assignments;
- local runtime-state contents;
- any `local` field or local-only node;
- any reference to a non-allowlisted output file.

The application is read-only. It has no market-data API, broker connection,
credentials, order controls, or runtime dashboard state.

## Technical Architecture

Use static HTML, CSS, and browser JavaScript without React, Vite, npm, or a
runtime backend. Vendor a pinned Cytoscape.js release and license for graph
rendering, pan/zoom, selection, and camera fitting. Use curated preset positions
per parent scope so the layout remains intentional and presentation-quality.

Proposed source structure:

```text
UI/architecture_atlas/
  README.md
  source/
    architecture.json
  static/
    index.html
    atlas.css
    atlas.js
    vendor/
      cytoscape.min.js
      LICENSE-cytoscape.txt
    fonts/
      SpaceGrotesk.woff2
      JetBrainsMono.woff2
      LICENSE-fonts.txt
  dist/
    public/
    local/
scripts/
  build_architecture_atlas.py
UI/tests/
  test_architecture_atlas_build.py
  test_architecture_atlas_content.py
```

Generated `dist` artifacts are reproducible and must not be edited manually.
They are ignored by Git. Deployment builds Public from tracked source and
passes that exact validated directory to Sites; the source manifest and static
application files are always tracked.

The first implementation release must map all major domains and every active,
legacy, or experimental module named in the repository status documentation.
At least Momentum Expansion and Multi-Ticker Swing must be fully traversable
through features, model/ranking, signal selection, order policy, execution,
audit, and evidence before the interaction pattern is replicated across the
remaining strategies.

## Figma, Lovable, and Sites

### Figma

Figma is the visual-contract tool, not the architecture source of truth. Create
four key frames from the approved design:

1. root overview;
2. Specialist Strategies after camera zoom;
3. selected module with feature and policy detail;
4. narrow-screen outline fallback.

The Figma file defines tokens, node variants, portal nodes, inspector states,
focus/hover states, dataset selector, validation readout, and motion timing.
Production labels and relationships continue to come from the JSON manifest.

### Lovable

Lovable is optional and disposable. It may be used to compare a motion or
responsive interaction quickly, but generated React/Vite code must not become a
second production implementation or source of architecture data. If it does
not materially improve a specific interaction over the browser prototype and
Figma frames, skip it.

### Sites

Sites is the public deployment target for
`UI/architecture_atlas/dist/public/`. Only a validated saved Public build may be
deployed. The Local build is never uploaded. A Sites project and hosting config
are created during deployment work; none exists today.

Default templates may supply deployment scaffolding, but the final visual
system remains the custom System Atlas design.

## Failure Handling

- Invalid source data fails the build and preserves the last valid artifact.
- A missing Local dataset hides the selector and leaves Public fully usable.
- A malformed or unsupported data schema shows a branded error panel with the
  expected and received versions.
- An unresolved required path or symbol fails validation; explicitly optional
  legacy evidence produces a visible warning in the validation report.
- A canvas-rendering failure falls back to the hierarchical outline.
- A missing font uses local system fallbacks without blocking the graph.
- Public redaction failure blocks deployment.
- Sites deployment failure leaves the previous production deployment intact.

## Accessibility

- Every node is keyboard focusable and exposes label, kind, maturity, and
  expandability.
- All mouse actions have keyboard equivalents.
- Focus state is distinct from hover and selection.
- Color is never the only carrier of edge type, maturity, or mode.
- Contrast targets WCAG 2.2 AA for text and controls.
- Reduced-motion behavior is automatic.
- The outline fallback provides a complete non-canvas reading order.
- Tooltips are supplemental; essential information lives in the inspector.

## Testing and Verification

Automated pytest coverage must include:

- valid deterministic Public and Local builds;
- schema-version rejection;
- duplicate, cycle, orphan, and invalid-edge detection;
- missing required path and symbol detection;
- variable-depth branches and leaf behavior encoded correctly;
- Public allowlist and Local-field exclusion;
- rejection of absolute paths and credential-like assignments in Public output;
- Local build containing both datasets and Public build containing only Public;
- output preservation after a failed build;
- complete reachability of required major domains and strategy modules;
- exact source-revision and validation metadata.

Verification also includes:

- `python -m py_compile` for the builder and tests;
- a local static-server smoke test for both outputs;
- direct `file://` opening of both outputs;
- browser checks for root navigation, at least two nested branches, Back/Forward,
  search, filters, dataset switching, copy link, and outline fallback;
- desktop, tablet, and narrow-screen inspection;
- keyboard-only and reduced-motion inspection;
- public-artifact text and file inventory review before Sites deployment.

Performance targets for the first release:

- under 2.5 MB uncompressed for the Public build;
- no external runtime requests;
- no more than 60 nodes and 100 edges rendered in one focused scope;
- usable first render within 1.5 seconds on a contemporary desktop;
- smooth pan, hover, selection, and camera transitions at the scoped graph size.

## Scope Boundaries

This design does not:

- visualize every file or generated artifact;
- infer architecture automatically from imports;
- replace source documentation, tests, dashboards, or the capstone figure;
- expose live data, account state, orders, credentials, or controls;
- change feature engineering, models, order policies, or trading behavior;
- require a JavaScript build toolchain;
- make Figma, Lovable, or Sites the semantic source of truth;
- deploy the Local Full build publicly.

## Acceptance Criteria

The feature is complete when:

1. Public and Local builds are produced deterministically from one validated
   manifest and one static application.
2. The root view explains the six major domains and dominant end-to-end flow.
3. Every documented strategy and context module is discoverable.
4. Strategy branches distinguish their own inputs, features, intelligence,
   policies, execution paths, maturity, mode, audit, and evidence.
5. Variable-depth navigation, breadcrumbs, search, filters, inspector, browser
   history, keyboard access, reduced motion, and mobile outline all work.
6. Public output contains no Local dataset or prohibited operational material.
7. Local Full clearly adds useful technical detail without including secrets.
8. Validation proves the curated nodes and edges remain grounded in current
   repository paths and declared evidence.
9. The Public artifact is successfully deployed through Sites and the Local
   artifact opens directly in a browser.
