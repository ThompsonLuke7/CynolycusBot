(function () {
  "use strict";

  const EDGE_TYPES = ["data", "feature", "signal", "policy", "execution", "audit", "research", "control"];
  const COLORS = {
    data: "#5498ff",
    feature: "#52eba0",
    signal: "#af79ff",
    policy: "#ffc95c",
    execution: "#ff6fa9",
    audit: "#46f3ff",
    research: "#6db8aa",
    control: "#7aa4c4",
    source: "#5498ff",
    model: "#ffc95c",
    system: "#46f3ff",
    ui: "#46f3ff"
  };
  const NODE_BACKGROUNDS = {
    data: "#0a1d32",
    feature: "#0b2524",
    signal: "#1c1730",
    policy: "#2a2112",
    execution: "#281522",
    audit: "#09252a",
    research: "#102525",
    control: "#0d1d2b"
  };

  const els = {
    cy: document.getElementById("cy"),
    fallback: document.getElementById("graph-fallback"),
    breadcrumbs: document.getElementById("breadcrumbs"),
    scopeLabel: document.getElementById("scope-label"),
    back: document.getElementById("back-button"),
    inspector: document.getElementById("inspector"),
    inspectorClose: document.getElementById("inspector-close"),
    inspectorKicker: document.getElementById("inspector-kicker"),
    inspectorTitle: document.getElementById("inspector-title"),
    inspectorSummary: document.getElementById("inspector-summary"),
    inspectorKind: document.getElementById("inspector-kind"),
    inspectorMaturity: document.getElementById("inspector-maturity"),
    inspectorMode: document.getElementById("inspector-mode"),
    inspectorNavigation: document.getElementById("inspector-navigation"),
    inspectorPaths: document.getElementById("inspector-paths"),
    inspectorConnections: document.getElementById("inspector-connections"),
    localBlock: document.getElementById("local-detail-block"),
    inspectorDetails: document.getElementById("inspector-details"),
    inspectorOwner: document.getElementById("inspector-owner"),
    enterDomain: document.getElementById("enter-domain"),
    copyLink: document.getElementById("copy-link"),
    datasetSwitch: document.getElementById("dataset-switch"),
    searchPanel: document.getElementById("search-panel"),
    searchInput: document.getElementById("search-input"),
    searchResults: document.getElementById("search-results"),
    filtersPanel: document.getElementById("filters-panel"),
    edgeFilters: document.getElementById("edge-filters"),
    showResearch: document.getElementById("show-research"),
    outlinePanel: document.getElementById("outline-panel"),
    outlineTree: document.getElementById("outline-tree"),
    helpPanel: document.getElementById("help-panel"),
    largeTextToggle: document.getElementById("large-text-toggle"),
    validationState: document.getElementById("validation-state"),
    validationDetail: document.getElementById("validation-detail"),
    validationReadout: document.querySelector(".validation-readout"),
    minimap: document.getElementById("minimap"),
    minimapNodes: document.getElementById("minimap-nodes"),
    contextDock: document.getElementById("context-dock"),
    contextDockItems: document.getElementById("context-dock-items"),
    toast: document.getElementById("toast")
  };

  const state = {
    bundle: window.ATLAS_DATA || null,
    datasetName: "public",
    dataset: null,
    nodesById: new Map(),
    childrenByParent: new Map(),
    edges: [],
    scopeId: "system",
    selectedId: null,
    edgeTypes: new Set(EDGE_TYPES),
    showResearch: true,
    cy: null,
    visibleIds: [],
    routeAnimation: null
  };

  const LARGE_DISPLAY_QUERY = "(min-width: 2560px) and (min-height: 1080px)";
  const LARGE_DISPLAY_STORAGE_KEY = "cynolycus-atlas-large-display";

  function defaultLargeDisplay() {
    return window.matchMedia(LARGE_DISPLAY_QUERY).matches;
  }

  function readLargeDisplayPreference() {
    try {
      const preference = window.localStorage.getItem(LARGE_DISPLAY_STORAGE_KEY);
      if (preference === "on") return true;
      if (preference === "off") return false;
    } catch (_) {
      // file:// and privacy-restricted browsers may not expose localStorage.
    }
    return defaultLargeDisplay();
  }

  function graphScale() {
    return document.body.classList.contains("large-display") ? 1.24 : 1;
  }

  function graphTextScale() {
    return document.body.classList.contains("large-display") ? 1.38 : 1;
  }

  function graphSpacingScale() {
    return document.body.classList.contains("large-display") ? 1.18 : 1;
  }

  function graphFitPadding() {
    return document.body.classList.contains("large-display") ? 96 : 72;
  }

  function graphPosition(node) {
    if (node.id === state.scopeId) return {x: 500, y: 330};
    const position = node.position || {x: 500, y: 330};
    const spacing = graphSpacingScale();
    return {
      x: 500 + (Number(position.x) - 500) * spacing,
      y: 330 + (Number(position.y) - 330) * spacing
    };
  }

  function refreshGraphViewport() {
    if (!state.cy) return;
    window.requestAnimationFrame(function () {
      state.cy.resize();
      state.cy.fit(state.cy.elements(), graphFitPadding());
    });
  }

  function applyLargeDisplay(enabled, persist) {
    document.body.classList.toggle("large-display", enabled);
    els.largeTextToggle.setAttribute("aria-pressed", String(enabled));
    els.largeTextToggle.setAttribute(
      "aria-label",
      enabled ? "Use standard display text" : "Use large display text"
    );
    els.largeTextToggle.title = enabled ? "Use standard display text" : "Use large display text";
    if (persist) {
      try {
        window.localStorage.setItem(LARGE_DISPLAY_STORAGE_KEY, enabled ? "on" : "off");
      } catch (_) {
        // The visual change still applies for the current page.
      }
    }
    if (state.cy) {
      state.cy.style(cyStyle());
      state.cy.maxZoom(enabled ? 2.2 : 1.7);
      state.cy.nodes().forEach(function (cyNode) {
        const node = state.nodesById.get(cyNode.id());
        if (node) cyNode.position(graphPosition(node));
      });
      refreshGraphViewport();
    }
  }

  function fail(message) {
    els.fallback.hidden = false;
    els.validationReadout.classList.add("error");
    els.validationState.innerHTML = "<span aria-hidden=\"true\">●</span> MANIFEST ERROR";
    els.validationDetail.textContent = message;
    console.error("Architecture Atlas:", message);
  }

  function nodePublic(node) {
    return node && node.public ? node.public : {};
  }

  function nodeLabel(node) {
    return nodePublic(node).label || node.id;
  }

  function titleCase(value) {
    return String(value || "unknown")
      .replace(/[-_]/g, " ")
      .replace(/\b\w/g, function (ch) { return ch.toUpperCase(); });
  }

  function buildIndexes() {
    state.nodesById = new Map();
    state.childrenByParent = new Map();
    (state.dataset.nodes || []).forEach(function (node) {
      state.nodesById.set(node.id, node);
      const parent = node.parent_id === null ? "__root__" : node.parent_id;
      if (!state.childrenByParent.has(parent)) state.childrenByParent.set(parent, []);
      state.childrenByParent.get(parent).push(node);
    });
    state.childrenByParent.forEach(function (nodes) {
      nodes.sort(function (a, b) { return nodeLabel(a).localeCompare(nodeLabel(b)); });
    });
    state.edges = (state.dataset.edges || []).slice();
  }

  function hasChildren(id) {
    return (state.childrenByParent.get(id) || []).length > 0;
  }

  function parentOf(id) {
    const node = state.nodesById.get(id);
    return node ? node.parent_id : null;
  }

  function ancestors(id) {
    const out = [];
    let current = state.nodesById.get(id);
    const guard = new Set();
    while (current && !guard.has(current.id)) {
      guard.add(current.id);
      out.unshift(current);
      current = current.parent_id ? state.nodesById.get(current.parent_id) : null;
    }
    return out;
  }

  function parseHash() {
    const raw = location.hash.replace(/^#\/?/, "");
    if (!raw) return {scope: "system", selected: null};
    const parts = raw.split("?");
    const scope = decodeURIComponent(parts[0] || "system");
    const params = new URLSearchParams(parts[1] || "");
    return {scope: scope, selected: params.get("selected")};
  }

  function setRoute(scopeId, selectedId, replace) {
    const path = "#/" + encodeURIComponent(scopeId) +
      (selectedId && selectedId !== scopeId ? "?selected=" + encodeURIComponent(selectedId) : "");
    if (replace) history.replaceState(null, "", path);
    else if (location.hash !== path) history.pushState(null, "", path);
  }

  function routeFromLocation() {
    const route = parseHash();
    state.scopeId = state.nodesById.has(route.scope) && hasChildren(route.scope) ? route.scope : "system";
    state.selectedId = state.nodesById.has(route.selected) ? route.selected : null;
    renderScope();
    if (state.selectedId) selectNode(state.selectedId, false);
  }

  function currentScopeElements() {
    const focus = state.nodesById.get(state.scopeId);
    const children = (state.childrenByParent.get(state.scopeId) || []).filter(function (node) {
      return state.showResearch || !["research", "audit"].includes(node.kind);
    });
    const nodes = [focus].concat(children).filter(Boolean);
    const ids = new Set(nodes.map(function (node) { return node.id; }));
    const portals = [];

    state.edges.forEach(function (edge) {
      if (!state.edgeTypes.has(edge.type)) return;
      const sourceInside = ids.has(edge.source);
      const targetInside = ids.has(edge.target);
      if (sourceInside === targetInside) return;
      const outsideId = sourceInside ? edge.target : edge.source;
      const outsideNode = state.nodesById.get(outsideId);
      if (!outsideNode || ids.has(outsideId) || portals.some(function (node) { return node.id === outsideId; })) return;
      if (!state.showResearch && ["research", "audit"].includes(outsideNode.kind)) return;
      portals.push(outsideNode);
    });

    const edges = state.edges.filter(function (edge) {
      return state.edgeTypes.has(edge.type) && ids.has(edge.source) && ids.has(edge.target);
    });
    return {nodes: nodes, edges: edges, portals: portals.slice(0, 5)};
  }

  function graphElements(scope) {
    const children = new Set((state.childrenByParent.get(state.scopeId) || []).map(function (n) { return n.id; }));
    const nodeElements = scope.nodes.map(function (node) {
      const position = graphPosition(node);
      return {
        group: "nodes",
        data: {
          id: node.id,
          label: nodeLabel(node),
          kind: node.kind,
          role: node.edge_color_role || node.kind,
          maturity: nodePublic(node).maturity || "",
          mode: nodePublic(node).mode || "",
          focus: node.id === state.scopeId ? "yes" : "no",
          expandable: hasChildren(node.id) ? "yes" : "no",
          child: children.has(node.id) ? "yes" : "no"
        },
        position: {x: Number(position.x), y: Number(position.y)}
      };
    });

    const edgeElements = scope.edges.map(function (edge) {
      const edgeData = edge.public || {};
      return {
        group: "edges",
        data: {
          id: edge.id,
          source: edge.source,
          target: edge.target,
          type: edge.type,
          label: edgeData.label || edge.type
        }
      };
    });
    return nodeElements.concat(edgeElements);
  }

  function cyStyle() {
    const nodeScale = graphScale();
    const textScale = graphTextScale();
    const edgeScale = document.body.classList.contains("large-display") ? 1.18 : 1;
    return [
      {
        selector: "node",
        style: {
          "width": 176 * nodeScale,
          "height": 96 * nodeScale,
          "shape": "cutrectangle",
          "background-color": function (ele) {
            return NODE_BACKGROUNDS[ele.data("role")] || NODE_BACKGROUNDS.control;
          },
          "background-opacity": 0.94,
          "border-width": 1.5 * edgeScale,
          "border-color": function (ele) { return COLORS[ele.data("role")] || "#46f3ff"; },
          "label": "data(label)",
          "color": "#eaf8ff",
          "font-family": "Space Grotesk",
          "font-size": 12 * textScale,
          "font-weight": 700,
          "text-wrap": "wrap",
          "text-overflow-wrap": "anywhere",
          "text-max-width": 142 * nodeScale,
          "text-valign": "center",
          "text-halign": "center",
          "overlay-opacity": 0,
          "underlay-color": function (ele) { return COLORS[ele.data("role")] || "#46f3ff"; },
          "underlay-opacity": 0,
          "underlay-padding": 0,
          "shadow-blur": 22 * edgeScale,
          "shadow-color": "#000",
          "shadow-opacity": 0.38,
          "shadow-offset-y": 10 * edgeScale,
          "transition-property": "background-blacken, border-width, opacity, underlay-opacity",
          "transition-duration": "150ms"
        }
      },
      {
        selector: "node[role = 'data']",
        style: {"shape": "barrel"}
      },
      {
        selector: "node[role = 'signal']",
        style: {"shape": "hexagon"}
      },
      {
        selector: "node[role = 'policy']",
        style: {"shape": "tag"}
      },
      {
        selector: "node[role = 'execution']",
        style: {"shape": "roundrectangle"}
      },
      {
        selector: "node[focus = 'yes']",
        style: {
          "width": 170 * nodeScale,
          "height": 170 * nodeScale,
          "shape": "hexagon",
          "background-color": "#0b2838",
          "border-color": "#46f3ff",
          "border-width": 2.2 * edgeScale,
          "font-size": 14 * textScale,
          "text-max-width": 124 * nodeScale,
          "underlay-opacity": 0,
          "shadow-blur": 30 * edgeScale,
          "shadow-color": "#000",
          "shadow-opacity": 0.42,
          "shadow-offset-y": 12 * edgeScale
        }
      },
      {
        selector: "node[expandable = 'yes'][focus != 'yes']",
        style: {
          "border-width": 2.3 * edgeScale
        }
      },
      {
        selector: "node.hovered",
        style: {
          "background-blacken": -0.13
        }
      },
      {
        selector: "node:selected",
        style: {
          "border-color": "#46f3ff",
          "border-width": 3.2 * edgeScale,
          "underlay-opacity": 0,
          "shadow-blur": 28 * edgeScale,
          "shadow-color": "#46f3ff",
          "shadow-opacity": 0.25
        }
      },
      {
        selector: "node.dimmed",
        style: {
          "opacity": 0.24,
          "text-opacity": 0.4,
          "underlay-opacity": 0
        }
      },
      {
        selector: "edge",
        style: {
          "width": 1.35 * edgeScale,
          "curve-style": "unbundled-bezier",
          "control-point-distances": 24,
          "control-point-weights": 0.5,
          "line-color": function (ele) { return COLORS[ele.data("type")] || "#2b5572"; },
          "target-arrow-color": function (ele) { return COLORS[ele.data("type")] || "#2b5572"; },
          "target-arrow-shape": "triangle",
          "arrow-scale": 0.65 * edgeScale,
          "line-style": "dashed",
          "line-dash-pattern": [5 * edgeScale, 11 * edgeScale],
          "opacity": 0.42,
          "label": "data(label)",
          "font-family": "JetBrains Mono",
          "font-size": 7 * textScale,
          "color": "#68889f",
          "text-background-color": "#030711",
          "text-background-opacity": 0.92,
          "text-background-padding": 4 * edgeScale,
          "text-rotation": "autorotate",
          "text-margin-y": -9 * edgeScale,
          "text-opacity": 0,
          "overlay-opacity": 0,
          "transition-property": "opacity, width, text-opacity",
          "transition-duration": "150ms"
        }
      },
      {
        selector: "edge[type = 'research'], edge[type = 'audit']",
        style: {"line-style": "dotted"}
      },
      {
        selector: "edge.active-flow",
        style: {
          "width": 2.2 * edgeScale,
          "opacity": 0.92,
          "text-opacity": 0.9,
          "z-index": 8
        }
      },
      {
        selector: "edge.dimmed",
        style: {
          "opacity": 0.08,
          "text-opacity": 0
        }
      }
    ];
  }

  function stopRouteAnimation() {
    if (state.routeAnimation) cancelAnimationFrame(state.routeAnimation);
    state.routeAnimation = null;
  }

  function startRouteAnimation() {
    stopRouteAnimation();
    if (!state.cy || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    let offset = 0;
    function tick() {
      offset = (offset - 0.22) % 16;
      try {
        state.cy.edges().style("line-dash-offset", offset);
      } catch (_) {
        stopRouteAnimation();
        return;
      }
      state.routeAnimation = requestAnimationFrame(tick);
    }
    state.routeAnimation = requestAnimationFrame(tick);
  }

  function initCy() {
    if (typeof window.cytoscape !== "function") {
      fail("The vendored graph renderer did not load.");
      return false;
    }
    els.cy.tabIndex = 0;
    state.cy = window.cytoscape({
      container: els.cy,
      elements: [],
      style: cyStyle(),
      layout: {name: "preset"},
      minZoom: 0.48,
      maxZoom: graphScale() > 1 ? 2.2 : 1.7,
      wheelSensitivity: 0.2,
      boxSelectionEnabled: false,
      autoungrabify: true
    });

    state.cy.on("tap", "node", function (event) {
      selectNode(event.target.id(), true);
    });
    state.cy.on("dbltap", "node", function (event) {
      enterNode(event.target.id());
    });
    state.cy.on("tap", function (event) {
      if (event.target === state.cy) closeInspector();
    });
    state.cy.on("mouseover", "node", function (event) {
      event.target.addClass("hovered");
    });
    state.cy.on("mouseout", "node", function (event) {
      event.target.removeClass("hovered");
    });
    return true;
  }

  function renderScope() {
    const scopeNode = state.nodesById.get(state.scopeId);
    if (!scopeNode) {
      state.scopeId = "system";
      return renderScope();
    }
    const scope = currentScopeElements();
    state.visibleIds = scope.nodes.map(function (node) { return node.id; });
    state.selectedId = null;
    closeInspector();
    renderBreadcrumbs();
    renderMinimap(scope.nodes);
    renderContextDock(scope.portals);
    els.scopeLabel.textContent = nodeLabel(scopeNode) + " · " +
      (state.childrenByParent.get(state.scopeId) || []).length + " children";
    els.back.hidden = !parentOf(state.scopeId);

    if (!state.cy && !initCy()) return;
    stopRouteAnimation();
    state.cy.elements().remove();
    state.cy.add(graphElements(scope));
    state.cy.layout({name: "preset", fit: false}).run();
    const duration = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 500;
    state.cy.animate(
      {fit: {eles: state.cy.elements(), padding: graphFitPadding()}},
      {duration: duration, easing: "ease-in-out-cubic"}
    );
    startRouteAnimation();
    renderOutline();
  }

  function renderBreadcrumbs() {
    els.breadcrumbs.replaceChildren();
    ancestors(state.scopeId).forEach(function (node, index, list) {
      if (index) {
        const divider = document.createElement("span");
        divider.className = "breadcrumb-divider";
        divider.textContent = "/";
        els.breadcrumbs.appendChild(divider);
      }
      const button = document.createElement("button");
      button.type = "button";
      button.className = "breadcrumb-button";
      button.textContent = nodeLabel(node);
      if (index === list.length - 1) button.setAttribute("aria-current", "page");
      button.addEventListener("click", function () {
        navigateToScope(node.id);
      });
      els.breadcrumbs.appendChild(button);
    });
  }

  function renderMinimap(nodes) {
    els.minimapNodes.replaceChildren();
    const positions = nodes.map(function (node) {
      const pos = graphPosition(node);
      return {node: node, x: Number(pos.x), y: Number(pos.y)};
    });
    positions.forEach(function (item) {
      const dot = document.createElement("i");
      dot.className = "minimap-node";
      dot.style.left = Math.max(2, Math.min(98, item.x / 10)) + "%";
      dot.style.top = Math.max(4, Math.min(96, item.y / 6.6)) + "%";
      dot.style.color = COLORS[item.node.edge_color_role] || COLORS[item.node.kind] || COLORS.control;
      dot.style.background = "currentColor";
      els.minimapNodes.appendChild(dot);
    });
    els.minimap.hidden = nodes.length < 7;
  }

  function renderContextDock(portals) {
    els.contextDockItems.replaceChildren();
    els.contextDock.hidden = !portals.length;
    document.body.classList.toggle("context-dock-open", Boolean(portals.length));
    portals.forEach(function (node) {
      const button = document.createElement("button");
      const role = node.edge_color_role || node.kind || "control";
      button.type = "button";
      button.className = "context-dock-button";
      button.style.setProperty("--dock-accent", COLORS[role] || COLORS.control);
      button.innerHTML = "<span></span><small></small>";
      button.querySelector("span").textContent = nodeLabel(node);
      button.querySelector("small").textContent = titleCase(node.kind);
      button.addEventListener("click", function () {
        navigateToScope(node.parent_id || node.id, node.id);
      });
      els.contextDockItems.appendChild(button);
    });
  }

  function highlightFlows(id) {
    if (!state.cy) return;
    state.cy.nodes().removeClass("dimmed");
    state.cy.edges().removeClass("dimmed active-flow");
    if (!id) return;
    const selected = state.cy.getElementById(id);
    if (!selected || !selected.length) return;
    const connectedEdges = selected.connectedEdges();
    const connectedNodes = connectedEdges.connectedNodes().union(selected);
    state.cy.nodes().difference(connectedNodes).addClass("dimmed");
    state.cy.edges().difference(connectedEdges).addClass("dimmed");
    connectedEdges.addClass("active-flow");
  }

  function selectNode(id, updateRoute) {
    const node = state.nodesById.get(id);
    if (!node) return;
    state.selectedId = id;
    if (state.cy) {
      state.cy.$(":selected").unselect();
      const cyNode = state.cy.getElementById(id);
      if (cyNode && cyNode.length) cyNode.select();
      highlightFlows(id);
    }
    openInspector(node);
    if (updateRoute) setRoute(state.scopeId, id, true);
  }

  function openInspector(node) {
    if (window.innerWidth > 760 && els.outlinePanel.classList.contains("open")) closeOutline();
    const pub = nodePublic(node);
    const local = state.datasetName === "local" ? (node.local || null) : null;
    els.inspectorKicker.textContent = hasChildren(node.id) ? "EXPANDABLE DOMAIN" : "ARCHITECTURE COMPONENT";
    els.inspectorTitle.textContent = pub.label || node.id;
    els.inspectorSummary.textContent = pub.summary || "No summary available.";
    els.inspectorKind.textContent = titleCase(node.kind);
    els.inspectorMaturity.textContent = titleCase(pub.maturity);
    els.inspectorMode.textContent = titleCase(pub.mode);
    els.inspectorNavigation.textContent = hasChildren(node.id) ? "Zoom deeper" : "Detail leaf";
    els.enterDomain.hidden = !hasChildren(node.id);
    els.enterDomain.dataset.nodeId = node.id;

    els.localBlock.hidden = !local;
    if (local) {
      els.inspectorDetails.textContent = local.details || "No additional local detail.";
      els.inspectorOwner.textContent = local.runtime_owner || "Not declared";
    }

    const paths = []
      .concat(pub.repo_paths || [])
      .concat(local && local.repo_paths ? local.repo_paths : []);
    els.inspectorPaths.replaceChildren();
    Array.from(new Set(paths)).forEach(function (path) {
      const li = document.createElement("li");
      li.textContent = path;
      els.inspectorPaths.appendChild(li);
    });
    if (!paths.length) {
      const li = document.createElement("li");
      li.textContent = "Presentation-only node";
      els.inspectorPaths.appendChild(li);
    }

    els.inspectorConnections.replaceChildren();
    state.edges
      .filter(function (edge) { return edge.source === node.id || edge.target === node.id; })
      .slice(0, 12)
      .forEach(function (edge) {
        const inbound = edge.target === node.id;
        const other = state.nodesById.get(inbound ? edge.source : edge.target);
        const li = document.createElement("li");
        const small = document.createElement("small");
        small.textContent = (inbound ? "Inbound " : "Outbound ") + edge.type;
        li.appendChild(small);
        li.appendChild(document.createTextNode((inbound ? "← " : "→ ") + (other ? nodeLabel(other) : "Unknown")));
        els.inspectorConnections.appendChild(li);
      });

    els.inspector.classList.add("open");
    els.inspector.setAttribute("aria-hidden", "false");
    document.body.classList.add("side-panel-open");
    refreshGraphViewport();
  }

  function closeInspector() {
    els.inspector.classList.remove("open");
    els.inspector.setAttribute("aria-hidden", "true");
    state.selectedId = null;
    if (state.cy) state.cy.$(":selected").unselect();
    highlightFlows(null);
    if (!els.outlinePanel.classList.contains("open")) {
      document.body.classList.remove("side-panel-open");
      refreshGraphViewport();
    }
  }

  function enterNode(id) {
    const node = state.nodesById.get(id);
    if (!node) return;
    if (hasChildren(id)) {
      navigateToScope(id);
      return;
    }
    const parent = node.parent_id;
    if (parent && parent !== state.scopeId) {
      navigateToScope(parent, id);
    } else {
      selectNode(id, true);
    }
  }

  function navigateToScope(id, selected) {
    if (!state.nodesById.has(id)) return;
    const scope = hasChildren(id) ? id : (parentOf(id) || "system");
    state.scopeId = scope;
    setRoute(scope, selected || (scope !== id ? id : null), false);
    renderScope();
    if (selected || scope !== id) selectNode(selected || id, false);
  }

  function goBack() {
    const parent = parentOf(state.scopeId);
    if (parent) navigateToScope(parent, state.scopeId);
  }

  function renderEdgeFilters() {
    els.edgeFilters.replaceChildren();
    EDGE_TYPES.forEach(function (type) {
      const label = document.createElement("label");
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = state.edgeTypes.has(type);
      checkbox.dataset.edgeType = type;
      checkbox.addEventListener("change", function () {
        if (checkbox.checked) state.edgeTypes.add(type);
        else state.edgeTypes.delete(type);
        renderScope();
      });
      const swatch = document.createElement("span");
      swatch.textContent = titleCase(type);
      swatch.style.color = COLORS[type];
      label.append(checkbox, swatch);
      els.edgeFilters.appendChild(label);
    });
  }

  function showPanel(panel) {
    [els.searchPanel, els.filtersPanel, els.helpPanel].forEach(function (item) {
      if (item !== panel) item.hidden = true;
    });
    panel.hidden = false;
    if (panel === els.searchPanel) {
      els.searchInput.focus();
      updateSearch();
    }
  }

  function closePanels() {
    [els.searchPanel, els.filtersPanel, els.helpPanel].forEach(function (panel) {
      panel.hidden = true;
    });
  }

  function updateSearch() {
    const query = els.searchInput.value.trim().toLowerCase();
    const results = Array.from(state.nodesById.values())
      .filter(function (node) {
        const pub = nodePublic(node);
        const haystack = [node.id, pub.label, pub.summary, pub.maturity, pub.mode].join(" ").toLowerCase();
        return !query || haystack.includes(query);
      })
      .sort(function (a, b) {
        if (!query) return nodeLabel(a).localeCompare(nodeLabel(b));
        const aLabel = nodeLabel(a).toLowerCase();
        const bLabel = nodeLabel(b).toLowerCase();
        return Number(bLabel.startsWith(query)) - Number(aLabel.startsWith(query)) || aLabel.localeCompare(bLabel);
      })
      .slice(0, 18);

    els.searchResults.replaceChildren();
    results.forEach(function (node) {
      const li = document.createElement("li");
      li.className = "search-result";
      const button = document.createElement("button");
      button.type = "button";
      button.innerHTML = "<strong></strong><small></small>";
      button.querySelector("strong").textContent = nodeLabel(node);
      button.querySelector("small").textContent = titleCase(node.kind) + " · " + titleCase(nodePublic(node).maturity);
      button.addEventListener("click", function () {
        closePanels();
        navigateToScope(node.parent_id || node.id, node.id);
      });
      li.appendChild(button);
      els.searchResults.appendChild(li);
    });
  }

  function outlineBranch(parentId) {
    const list = document.createElement("ul");
    (state.childrenByParent.get(parentId) || []).forEach(function (node) {
      if (!state.showResearch && ["research", "audit"].includes(node.kind)) return;
      const li = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.className = "outline-node";
      button.textContent = nodeLabel(node);
      const tag = document.createElement("small");
      tag.textContent = titleCase(node.kind);
      button.appendChild(tag);
      button.addEventListener("click", function () {
        if (window.innerWidth > 760) closeOutline();
        navigateToScope(node.parent_id || node.id, node.id);
      });
      li.appendChild(button);
      if (hasChildren(node.id)) li.appendChild(outlineBranch(node.id));
      list.appendChild(li);
    });
    return list;
  }

  function renderOutline() {
    els.outlineTree.replaceChildren();
    const root = state.nodesById.get("system");
    if (!root) return;
    const top = document.createElement("ul");
    const li = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "outline-node";
    button.textContent = nodeLabel(root);
    button.addEventListener("click", function () { navigateToScope("system"); });
    li.append(button, outlineBranch("system"));
    top.appendChild(li);
    els.outlineTree.appendChild(top);
  }

  function openOutline() {
    if (window.innerWidth > 760 && els.inspector.classList.contains("open")) closeInspector();
    els.outlinePanel.classList.add("open");
    els.outlinePanel.setAttribute("aria-hidden", "false");
    if (window.innerWidth > 760) {
      document.body.classList.add("side-panel-open");
      refreshGraphViewport();
    }
  }

  function closeOutline() {
    if (window.innerWidth <= 760) return;
    els.outlinePanel.classList.remove("open");
    els.outlinePanel.setAttribute("aria-hidden", "true");
    if (!els.inspector.classList.contains("open")) {
      document.body.classList.remove("side-panel-open");
      refreshGraphViewport();
    }
  }

  function toast(message) {
    els.toast.textContent = message;
    els.toast.hidden = false;
    clearTimeout(toast.timer);
    toast.timer = setTimeout(function () { els.toast.hidden = true; }, 2200);
  }

  function copyCurrentLink() {
    const value = location.href;
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(value).then(function () { toast("Architecture link copied"); });
      return;
    }
    const input = document.createElement("textarea");
    input.value = value;
    input.style.position = "fixed";
    input.style.opacity = "0";
    document.body.appendChild(input);
    input.select();
    try {
      document.execCommand("copy");
      toast("Architecture link copied");
    } catch (_) {
      toast("Copy unavailable — use the browser address bar");
    }
    input.remove();
  }

  function switchDataset(name) {
    if (!state.bundle.datasets[name]) return;
    const previousScope = state.scopeId;
    const previousSelected = state.selectedId;
    state.datasetName = name;
    state.dataset = state.bundle.datasets[name];
    buildIndexes();
    if (!state.nodesById.has(previousScope)) state.scopeId = "system";
    else state.scopeId = previousScope;
    state.selectedId = state.nodesById.has(previousSelected) ? previousSelected : null;
    els.datasetSwitch.querySelectorAll("button").forEach(function (button) {
      button.setAttribute("aria-pressed", String(button.dataset.dataset === name));
    });
    renderScope();
    if (state.selectedId) selectNode(state.selectedId, false);
    updateValidationReadout();
    toast(name === "local" ? "Local Full context enabled" : "Public-safe context enabled");
  }

  function updateValidationReadout() {
    const metadata = state.bundle.metadata || {};
    const counts = metadata.validation_counts || {};
    els.validationReadout.classList.remove("error");
    els.validationState.innerHTML = "<span aria-hidden=\"true\">●</span> MANIFEST VALID";
    els.validationDetail.textContent =
      (counts.nodes || state.dataset.nodes.length) + " nodes · " +
      (counts.edges || state.dataset.edges.length) + " edges · " +
      state.datasetName.toUpperCase() + " · " +
      (metadata.source_revision || "local");
  }

  function bindEvents() {
    document.getElementById("home-button").addEventListener("click", function () { navigateToScope("system"); });
    document.getElementById("fit-button").addEventListener("click", function () {
      if (state.cy) state.cy.fit(state.cy.elements(), graphFitPadding());
    });
    document.getElementById("search-open").addEventListener("click", function () { showPanel(els.searchPanel); });
    document.getElementById("search-rail").addEventListener("click", function () { showPanel(els.searchPanel); });
    document.getElementById("filters-open").addEventListener("click", function () { showPanel(els.filtersPanel); });
    document.getElementById("outline-open").addEventListener("click", openOutline);
    document.getElementById("help-open").addEventListener("click", function () { showPanel(els.helpPanel); });
    els.largeTextToggle.addEventListener("click", function () {
      const enabled = !document.body.classList.contains("large-display");
      applyLargeDisplay(enabled, true);
      toast(enabled ? "Large display text enabled" : "Standard display text enabled");
    });
    els.back.addEventListener("click", goBack);
    els.inspectorClose.addEventListener("click", closeInspector);
    els.enterDomain.addEventListener("click", function () { enterNode(els.enterDomain.dataset.nodeId); });
    els.copyLink.addEventListener("click", copyCurrentLink);
    els.searchInput.addEventListener("input", updateSearch);
    els.showResearch.addEventListener("change", function () {
      state.showResearch = els.showResearch.checked;
      renderScope();
    });
    document.getElementById("filters-reset").addEventListener("click", function () {
      state.edgeTypes = new Set(EDGE_TYPES);
      state.showResearch = true;
      els.showResearch.checked = true;
      renderEdgeFilters();
      renderScope();
    });
    els.datasetSwitch.addEventListener("click", function (event) {
      const button = event.target.closest("button[data-dataset]");
      if (button) switchDataset(button.dataset.dataset);
    });
    document.querySelectorAll("[data-close-panel]").forEach(function (button) {
      button.addEventListener("click", function () {
        const panel = document.getElementById(button.dataset.closePanel);
        if (panel === els.outlinePanel) closeOutline();
        else panel.hidden = true;
      });
    });
    document.querySelectorAll("[data-open-outline]").forEach(function (button) {
      button.addEventListener("click", openOutline);
    });
    window.addEventListener("hashchange", routeFromLocation);
    window.addEventListener("popstate", routeFromLocation);
    window.addEventListener("resize", function () {
      if (state.cy) state.cy.resize();
      if (window.innerWidth <= 760) openOutline();
    });
    document.addEventListener("keydown", function (event) {
      if (event.key === "/" && !["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) {
        event.preventDefault();
        showPanel(els.searchPanel);
      } else if (event.key === "Escape") {
        if (!els.searchPanel.hidden || !els.filtersPanel.hidden || !els.helpPanel.hidden) closePanels();
        else if (els.inspector.classList.contains("open")) closeInspector();
        else if (els.outlinePanel.classList.contains("open") && window.innerWidth > 760) closeOutline();
        else goBack();
      } else if (event.key === "Enter" && document.activeElement === els.cy) {
        if (state.selectedId) enterNode(state.selectedId);
      }
    });
  }

  function validateBundle(bundle) {
    if (!bundle || bundle.schema_version !== 1) throw new Error("Unsupported or missing atlas schema.");
    if (!bundle.datasets || !bundle.datasets.public) throw new Error("Public architecture dataset is missing.");
    if (!Array.isArray(bundle.datasets.public.nodes) || !Array.isArray(bundle.datasets.public.edges)) {
      throw new Error("Architecture nodes or edges are malformed.");
    }
    if (!bundle.datasets.public.nodes.some(function (node) { return node.id === "system"; })) {
      throw new Error("The system root is missing.");
    }
  }

  function boot() {
    try {
      validateBundle(state.bundle);
    } catch (error) {
      fail(error.message);
      openOutline();
      return;
    }
    state.dataset = state.bundle.datasets.public;
    applyLargeDisplay(readLargeDisplayPreference(), false);
    buildIndexes();
    if (!state.bundle.datasets.local) {
      els.datasetSwitch.hidden = true;
    }
    renderEdgeFilters();
    bindEvents();
    updateValidationReadout();
    const route = parseHash();
    if (!location.hash) setRoute("system", null, true);
    state.scopeId = state.nodesById.has(route.scope) && hasChildren(route.scope) ? route.scope : "system";
    state.selectedId = state.nodesById.has(route.selected) ? route.selected : null;
    renderScope();
    if (state.selectedId) selectNode(state.selectedId, false);
    if (window.innerWidth <= 760) openOutline();
  }

  boot();
})();
