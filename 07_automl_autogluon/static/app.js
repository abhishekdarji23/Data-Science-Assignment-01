const MODEL_LABELS = {
  logistic_regression: "Logistic Regression",
  random_forest: "Random Forest",
  gradient_boosting: "Gradient Boosting",
  k_nearest_neighbors: "K-Nearest Neighbors",
  decision_tree: "Decision Tree",
  stacked_ensemble: "Stacked Ensemble (meta-learner)",
};

async function loadLeaderboard() {
  const res = await fetch("/api/leaderboard");
  if (!res.ok) {
    document.getElementById("leaderboard-box").textContent = "Model not trained yet — run `python src/train.py`.";
    return;
  }
  const data = await res.json();
  const entries = Object.entries(data.leaderboard).sort((a, b) => b[1].roc_auc - a[1].roc_auc);
  const maxAuc = Math.max(...entries.map(([, m]) => m.roc_auc));

  const bars = entries
    .map(([name, m]) => {
      const pct = (m.roc_auc / maxAuc) * 100;
      const cls = name === "stacked_ensemble" ? "ensemble" : "";
      const star = name === data.best_model ? " ⭐" : "";
      return `<div class="bar-row">
        <div class="bar-label">${MODEL_LABELS[name] || name}${star}</div>
        <div class="bar-track"><div class="bar-fill ${cls}" style="width:${pct}%"></div></div>
        <div class="bar-value">${m.roc_auc}</div>
      </div>`;
    })
    .join("");

  const rows = entries
    .map(([name, m]) => {
      const cls = name === data.best_model ? "best-row" : "";
      return `<tr class="${cls}"><td>${MODEL_LABELS[name] || name}</td><td>${m.roc_auc}</td><td>${(m.accuracy * 100).toFixed(1)}%</td><td>${(m.precision * 100).toFixed(1)}%</td><td>${(m.recall * 100).toFixed(1)}%</td><td>${(m.f1 * 100).toFixed(1)}%</td></tr>`;
    })
    .join("");

  document.getElementById("leaderboard-box").innerHTML = `
    <p class="hint">${data.n_train} train / ${data.n_test} test customers, ${(data.churn_rate * 100).toFixed(1)}% churn rate.</p>
    ${bars}
    <table style="margin-top:14px;">
      <thead><tr><th>Model</th><th>ROC-AUC</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

document.getElementById("score-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const payload = {
    tenure_months: parseFloat(document.getElementById("tenure").value),
    monthly_charges: parseFloat(document.getElementById("monthly").value),
    total_charges: parseFloat(document.getElementById("total").value),
    contract_type: document.getElementById("contract").value,
    internet_service: document.getElementById("internet").value,
    tech_support: document.getElementById("techsupport").value,
    payment_method: document.getElementById("payment").value,
    num_support_calls: parseInt(document.getElementById("calls").value, 10),
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
  const box = document.getElementById("predict-result");
  box.style.display = "block";
  box.className = `risk-${data.risk_level}`;

  const modelRows = Object.entries(data.base_model_predictions)
    .map(([name, p]) => `<tr><td>${MODEL_LABELS[name] || name}</td><td>${(p * 100).toFixed(1)}%</td></tr>`)
    .join("");

  box.innerHTML = `
    <div class="hint">Stacked ensemble prediction</div>
    <div class="risk-value">${(data.churn_probability * 100).toFixed(1)}% churn probability — ${data.risk_level} risk</div>
    <table class="risk-detail-table">
      <thead><tr><th>Individual base model opinion</th><th>Probability</th></tr></thead>
      <tbody>${modelRows}</tbody>
    </table>`;
});

loadLeaderboard();
