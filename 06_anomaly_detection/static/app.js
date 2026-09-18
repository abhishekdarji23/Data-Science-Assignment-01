const METHOD_LABELS = {
  isolation_forest: "Isolation Forest",
  local_outlier_factor: "Local Outlier Factor (LOF)",
  z_score_baseline: "Z-score (statistical baseline)",
};

async function loadMetrics() {
  const res = await fetch("/api/metrics");
  if (!res.ok) {
    document.getElementById("methods-box").textContent = "Model not trained yet — run `python src/train.py`.";
    return;
  }
  const m = await res.json();
  const rows = Object.entries(m.methods_compared)
    .sort((a, b) => b[1].pr_auc - a[1].pr_auc)
    .map(([method, stats], i) => {
      const cls = method === m.best_method ? "best-method" : "";
      const star = method === m.best_method ? " ⭐ best" : "";
      return `<tr class="${cls}">
        <td>${METHOD_LABELS[method] || method}${star}</td>
        <td>${stats.pr_auc}</td>
        <td>${(stats.precision_at_k * 100).toFixed(1)}%</td>
        <td>${(stats.recall_at_k * 100).toFixed(1)}%</td>
        <td>${(stats.f1_at_k * 100).toFixed(1)}%</td>
      </tr>`;
    })
    .join("");
  document.getElementById("methods-box").innerHTML = `
    <p class="hint">${m.n_transactions.toLocaleString()} transactions, ${m.n_anomalies_labeled} labeled anomalies (${(m.contamination_rate * 100).toFixed(1)}%). Production model: <strong>${METHOD_LABELS[m.production_method]}</strong>.</p>
    <table>
      <thead><tr><th>Method</th><th>PR-AUC</th><th>Precision@k</th><th>Recall@k</th><th>F1@k</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

async function loadAnomalies() {
  const res = await fetch("/api/anomalies?limit=15");
  if (!res.ok) {
    document.getElementById("anomalies-box").textContent = "Model not trained yet — run `python src/train.py`.";
    return;
  }
  const rows = await res.json();
  const trs = rows
    .map(
      (r) => `<tr>
        <td>${r.transaction_id}</td>
        <td>$${r.amount_usd}</td>
        <td>${r.hour_of_day}:00</td>
        <td>${r.account_age_days}d</td>
        <td>${r.transactions_last_hour}</td>
        <td>${r.distance_from_home_km} km</td>
        <td>${r.anomaly_score.toFixed(3)}</td>
        <td>${r.is_anomaly === 1 ? '<span class="badge flagged">true anomaly</span>' : ""}</td>
      </tr>`
    )
    .join("");
  document.getElementById("anomalies-box").innerHTML = `
    <table>
      <thead><tr><th>ID</th><th>Amount</th><th>Hour</th><th>Acct age</th><th>Tx/hr</th><th>Distance</th><th>Score</th><th>Ground truth</th></tr></thead>
      <tbody>${trs}</tbody>
    </table>`;
}

document.getElementById("score-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const payload = {
    amount_usd: parseFloat(document.getElementById("amount").value),
    hour_of_day: parseInt(document.getElementById("hour").value, 10),
    account_age_days: parseFloat(document.getElementById("account_age").value),
    transactions_last_hour: parseInt(document.getElementById("tx_last_hour").value, 10),
    distance_from_home_km: parseFloat(document.getElementById("distance").value),
  };
  const res = await fetch("/api/score", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    alert("Scoring failed: " + (err.detail || res.statusText));
    return;
  }
  const data = await res.json();
  const box = document.getElementById("score-result");
  box.style.display = "block";
  box.className = data.flagged_as_anomaly ? "risk-high" : "risk-low";
  box.innerHTML = `
    <div class="risk-label">Risk score (0-100)</div>
    <div class="risk-value">${data.risk_score}</div>
    <div class="risk-detail">${data.flagged_as_anomaly ? "🚩 Flagged as anomaly" : "✅ Looks normal"} — Isolation Forest raw score: ${data.raw_isolation_forest_score}</div>
  `;
});

loadMetrics();
loadAnomalies();
