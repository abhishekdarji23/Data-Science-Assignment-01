const API_URL = "http://localhost:8009";

const graphInput = document.getElementById("graphInput");
const messageEl = document.getElementById("message");
const statusBody = document.querySelector("#statusTable tbody");
const validateBtn = document.getElementById("validateBtn");
const runBtn = document.getElementById("runBtn");

let pollTimer = null;

function renderStatuses(nodes) {
  statusBody.innerHTML = "";
  for (const n of nodes) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${n.label || n.id}</td>
      <td>${(n.dependsOn || []).join(", ") || "—"}</td>
      <td><span class="badge badge-${n.status}">${n.status}</span></td>`;
    statusBody.appendChild(tr);
  }
}

async function loadGraph() {
  const res = await fetch(`${API_URL}/api/graph`);
  const data = await res.json();
  graphInput.value = JSON.stringify(data, null, 2);
  renderStatuses(data.nodes.map((n) => ({ ...n, status: "pending" })));
}

async function validate() {
  let parsed;
  try {
    parsed = JSON.parse(graphInput.value);
  } catch (e) {
    messageEl.textContent = `Invalid JSON: ${e.message}`;
    return;
  }
  const res = await fetch(`${API_URL}/api/graph`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(parsed),
  });
  const data = await res.json();
  if (!res.ok) {
    messageEl.textContent = `Invalid graph: ${data.error}`;
    return;
  }
  messageEl.textContent = `Valid. Execution levels (parallel groups):\n${data.levels
    .map((lvl, i) => `  level ${i}: ${lvl.join(", ")}`)
    .join("\n")}`;
  renderStatuses(parsed.nodes.map((n) => ({ ...n, status: "pending" })));
}

async function run() {
  await validate();
  const res = await fetch(`${API_URL}/api/run`, { method: "POST" });
  const data = await res.json();
  if (!res.ok) {
    messageEl.textContent = `Could not start run: ${data.error}`;
    return;
  }
  poll();
}

function poll() {
  clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    const res = await fetch(`${API_URL}/api/status`);
    const data = await res.json();
    renderStatuses(data.nodes);
    if (!data.running) clearInterval(pollTimer);
  }, 400);
}

validateBtn.addEventListener("click", validate);
runBtn.addEventListener("click", run);
window.addEventListener("DOMContentLoaded", loadGraph);
