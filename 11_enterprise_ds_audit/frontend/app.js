const API_URL = "http://localhost:8011";

const DIMENSION_LABELS = {
  leakage: "Leakage",
  missing_data: "Missing Data",
  distribution_drift: "Distribution Drift",
  label_quality: "Label Quality",
  split_integrity: "Split Integrity",
  reproducibility: "Reproducibility",
};

async function load() {
  const res = await fetch(`${API_URL}/api/audit`);
  const data = await res.json();

  document.getElementById("overall").innerHTML = `
    <span class="score-badge status-${data.overall_status}">${data.overall_score}/100</span>
    <span>Overall status: <strong>${data.overall_status}</strong></span>
  `;

  const grid = document.getElementById("grid");
  grid.innerHTML = "";
  for (const [key, dim] of Object.entries(data.dimensions)) {
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `
      <h3>${DIMENSION_LABELS[key] || key} <span class="score-badge status-${dim.status}" style="font-size: 0.9rem; padding: 2px 10px;">${dim.score}</span></h3>
      <ul>${dim.findings.map((f) => `<li>${f}</li>`).join("")}</ul>
    `;
    grid.appendChild(card);
  }
}

load();
