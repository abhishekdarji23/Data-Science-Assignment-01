const CLUSTER_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22", "#34495e"];

async function loadCustomersAndDrawScatter() {
  const res = await fetch("/api/customers");
  if (!res.ok) return;
  const rows = await res.json();

  const byCluster = {};
  rows.forEach((r) => {
    const c = r.cluster;
    if (!byCluster[c]) byCluster[c] = [];
    byCluster[c].push({ x: r["Annual Income (k$)"], y: r["Spending Score (1-100)"] });
  });

  const clusterIds = Object.keys(byCluster).sort((a, b) => a - b);
  const datasets = clusterIds.map((c, i) => ({
    label: `Segment ${c}`,
    data: byCluster[c],
    backgroundColor: CLUSTER_COLORS[i % CLUSTER_COLORS.length],
    pointRadius: 4,
  }));

  new Chart(document.getElementById("scatter"), {
    type: "scatter",
    data: { datasets },
    options: {
      scales: {
        x: { title: { display: true, text: "Annual Income (k$)" } },
        y: { title: { display: true, text: "Spending Score (1-100)" } },
      },
      plugins: { legend: { display: true, position: "bottom" } },
    },
  });
}

async function loadClusterProfiles() {
  const res = await fetch("/api/clusters");
  if (!res.ok) {
    document.getElementById("clusters-box").textContent = "Model not trained yet — run `python src/train.py`.";
    return;
  }
  const profiles = await res.json();
  const rows = profiles
    .map(
      (p) => `<tr>
        <td>${p.cluster}</td>
        <td>${p.label}</td>
        <td>${p.size} (${p.pct_of_customers}%)</td>
        <td>${p.avg_age}</td>
        <td>$${p.avg_income_k}k</td>
        <td>${p.avg_spending_score}</td>
      </tr>`
    )
    .join("");
  document.getElementById("clusters-box").innerHTML = `
    <table>
      <thead><tr><th>#</th><th>Segment</th><th>Size</th><th>Avg age</th><th>Avg income</th><th>Avg spending</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

async function loadMetrics() {
  const res = await fetch("/api/metrics");
  if (!res.ok) {
    document.getElementById("metrics-box").textContent = "Model not trained yet — run `python src/train.py`.";
    return;
  }
  const m = await res.json();
  const rows = [
    ["Model", m.model],
    ["Customers", m.n_customers],
    ["Chosen k (segments)", m.chosen_k],
    ["Silhouette score (chosen k)", m.chosen_k_silhouette],
    ["k values tried", m.k_candidates_tried.join(", ")],
  ]
    .map(([k, v]) => `<tr><th>${k}</th><td>${v}</td></tr>`)
    .join("");
  document.getElementById("metrics-box").innerHTML = `<table>${rows}</table>`;
}

document.getElementById("predict-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const payload = {
    age: parseFloat(document.getElementById("age").value),
    annual_income_k: parseFloat(document.getElementById("income").value),
    spending_score: parseFloat(document.getElementById("spending").value),
  };
  const res = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    alert("Prediction failed: " + (err.detail || res.statusText));
    return;
  }
  const data = await res.json();
  document.getElementById("predict-result").style.display = "flex";
  document.getElementById("predict-value").textContent = `Segment ${data.cluster} — ${data.segment_label}`;
});

loadCustomersAndDrawScatter();
loadClusterProfiles();
loadMetrics();
