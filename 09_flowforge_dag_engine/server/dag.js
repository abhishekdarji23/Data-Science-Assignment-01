/**
 * FlowForge core: a small DAG (directed acyclic graph) engine.
 *
 * - Kahn's algorithm does two jobs at once: it detects cycles (if we run out
 *   of zero-indegree nodes before every node is visited, a cycle exists),
 *   and it groups nodes into "levels" — everything in a level has all its
 *   dependencies already satisfied by the previous levels, so a real
 *   scheduler could run an entire level in parallel.
 * - "Running" the graph simulates that: each level executes concurrently,
 *   a node that's set to fail marks everything downstream of it as
 *   skipped rather than run, and the engine waits for each level to
 *   finish before starting the next.
 */

function topologicalLevels(nodes) {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const indegree = new Map(nodes.map((n) => [n.id, 0]));
  const dependents = new Map(nodes.map((n) => [n.id, []]));

  for (const node of nodes) {
    for (const depId of node.dependsOn || []) {
      if (!byId.has(depId)) {
        return { error: `Node "${node.id}" depends on unknown node "${depId}"` };
      }
      indegree.set(node.id, indegree.get(node.id) + 1);
      dependents.get(depId).push(node.id);
    }
  }

  let frontier = nodes.filter((n) => indegree.get(n.id) === 0).map((n) => n.id);
  const levels = [];
  const visited = new Set();

  while (frontier.length > 0) {
    levels.push(frontier);
    const nextFrontier = [];
    for (const id of frontier) {
      visited.add(id);
      for (const depId of dependents.get(id)) {
        indegree.set(depId, indegree.get(depId) - 1);
        if (indegree.get(depId) === 0) nextFrontier.push(depId);
      }
    }
    frontier = nextFrontier;
  }

  if (visited.size !== nodes.length) {
    const stuck = nodes.map((n) => n.id).filter((id) => !visited.has(id));
    return { error: `Cycle detected — these nodes never reached zero dependencies: ${stuck.join(", ")}` };
  }

  return { levels, order: levels.flat() };
}

async function runGraph(nodes, { onUpdate } = {}) {
  const { levels, error } = topologicalLevels(nodes);
  if (error) throw new Error(error);

  const byId = new Map(nodes.map((n) => [n.id, { ...n, status: "pending" }]));
  const failed = new Set();
  const emit = () => onUpdate && onUpdate([...byId.values()]);

  emit();

  for (const level of levels) {
    // if any dependency already failed or was skipped, skip this node instead of running it
    const toRun = [];
    for (const id of level) {
      const node = byId.get(id);
      const blocked = (node.dependsOn || []).some((depId) => failed.has(depId));
      if (blocked) {
        node.status = "skipped";
        failed.add(id);
      } else {
        node.status = "running";
        toRun.push(node);
      }
    }
    emit();

    await Promise.all(
      toRun.map(
        (node) =>
          new Promise((resolve) => {
            setTimeout(() => {
              node.status = node.shouldFail ? "failed" : "success";
              if (node.status === "failed") failed.add(node.id);
              resolve();
            }, node.duration || 300);
          })
      )
    );
    emit();
  }

  return [...byId.values()];
}

module.exports = { topologicalLevels, runGraph };
