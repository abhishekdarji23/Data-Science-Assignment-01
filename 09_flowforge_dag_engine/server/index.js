const express = require("express");
const cors = require("cors");
const { topologicalLevels, runGraph } = require("./dag");

const PORT = process.env.PORT || 8009;
const app = express();
app.use(cors());
app.use(express.json());

// a small default pipeline: two independent branches (data + linting) that
// converge on deploy, so there's something to look at on first load
let graph = {
  nodes: [
    { id: "fetch_data", label: "Fetch data", dependsOn: [], duration: 400 },
    { id: "clean_data", label: "Clean data", dependsOn: ["fetch_data"], duration: 500 },
    { id: "train_model", label: "Train model", dependsOn: ["clean_data"], duration: 900 },
    { id: "evaluate_model", label: "Evaluate model", dependsOn: ["train_model"], duration: 400 },
    { id: "lint", label: "Lint", dependsOn: [], duration: 300 },
    { id: "build", label: "Build", dependsOn: ["lint"], duration: 500 },
    { id: "deploy", label: "Deploy", dependsOn: ["evaluate_model", "build"], duration: 600 },
  ],
};

let statuses = graph.nodes.map((n) => ({ ...n, status: "pending" }));
let running = false;

app.get("/api/graph", (req, res) => {
  res.json(graph);
});

app.post("/api/graph", (req, res) => {
  const nodes = req.body.nodes;
  if (!Array.isArray(nodes) || nodes.length === 0) {
    return res.status(400).json({ error: "Body must be { nodes: [...] } with at least one node." });
  }
  const result = topologicalLevels(nodes);
  if (result.error) {
    return res.status(400).json({ error: result.error });
  }
  graph = { nodes };
  statuses = nodes.map((n) => ({ ...n, status: "pending" }));
  res.json({ valid: true, levels: result.levels, order: result.order });
});

app.post("/api/run", async (req, res) => {
  if (running) {
    return res.status(409).json({ error: "A run is already in progress." });
  }
  const check = topologicalLevels(graph.nodes);
  if (check.error) {
    return res.status(400).json({ error: check.error });
  }

  running = true;
  res.json({ started: true, levels: check.levels });

  try {
    await runGraph(graph.nodes, {
      onUpdate: (nodeStates) => {
        statuses = nodeStates;
      },
    });
  } finally {
    running = false;
  }
});

app.get("/api/status", (req, res) => {
  res.json({ running, nodes: statuses });
});

app.listen(PORT, () => {
  console.log(`FlowForge server listening on http://localhost:${PORT}`);
});
