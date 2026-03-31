/**
 * graph.js — Transaction Cosmos (force-graph network visualisation).
 *
 * Uses the `force-graph` library loaded via CDN.
 * Nodes represent accounts/merchants; links represent transactions.
 * Fraud links glow red; normal links are indigo.
 */

let _graph = null;

export function initGraph(containerId, graphData) {
  const container = document.getElementById(containerId);
  if (!container) return;

  // Destroy previous instance if re-initialising
  if (_graph) {
    _graph._destructor?.();
    container.innerHTML = "";
  }

  const { nodes, links } = graphData;
  if (!nodes?.length) {
    container.innerHTML =
      '<p class="text-slate-500 text-sm text-center pt-16">No graph data available.</p>';
    return;
  }

  // Colour helpers — NPC nodes are greyed out; CSV nodes use vivid colours
  const nodeColor = (n) => {
    if (!n.in_csv) return "rgba(100,116,139,0.35)";  // NPC: muted slate
    if (n.fraud) return "#ef4444";                    // fraud: red
    return n.type === "merchant" ? "#8b5cf6" : "#6366f1"; // merchant: purple, account: indigo
  };
  const nodeRelSize = (n) => (n.in_csv ? 5 : 3);
  const linkColor = (l) => {
    if (l.npc) return "rgba(100,116,139,0.08)";            // NPC link: nearly invisible
    return l.fraud ? "rgba(239,68,68,0.7)" : "rgba(99,102,241,0.3)";
  };
  const linkWidth = (l) => {
    if (l.npc) return 0.4;
    return l.fraud ? 2 : 0.8;
  };

  _graph = ForceGraph()(container)
    .backgroundColor("transparent")
    .nodeId("id")
    .nodeLabel((n) => n.in_csv ? `${n.id} [${n.type}]` : `${n.id} [background]`)
    .nodeColor(nodeColor)
    .nodeRelSize(4)
    .nodeVal((n) => (n.in_csv ? 5 : 2))
    .linkColor(linkColor)
    .linkWidth(linkWidth)
    .linkDirectionalArrowLength((l) => (l.npc ? 0 : 4))
    .linkDirectionalArrowRelPos(1)
    .linkLabel((l) => l.npc ? "" : `$${l.amount?.toLocaleString() || 0}`)
    .graphData({ nodes, links })
    .width(container.clientWidth || 800)
    .height(container.clientHeight || 500)
    .onNodeClick((node) => {
      _graph.centerAt(node.x, node.y, 600);
      _graph.zoom(4, 600);
    });

}

export function destroyGraph() {
  if (_graph) {
    _graph._destructor?.();
    _graph = null;
  }
}
