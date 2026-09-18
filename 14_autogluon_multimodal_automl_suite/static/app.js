const MODEL_LABELS = {
  tabular_only: "Tabular only",
  text_only: "Text only",
  fusion: "Fusion (both)",
};

async function loadLeaderboard() {
  const res = await fetch("/api/leaderboard");
  if (!res.ok) {
    document.getElementById("leaderboard-box").textContent = "Model not trained yet — run `python src/train.py`.";
    return;
  }
  const data = await res.json();
  const entries = Object.entries(data.leaderboard);
  const maxR2 = Math.max(...entries.map(([, m]) => m.r2));

  const bars = entries
    .map(([name, m]) => {
      const pct = (m.r2 / maxR2) * 100;
      const cls = name === "fusion" ? "fusion" : "";
      const star = name === data.best_model ? " ⭐" : "";
      return `<div class="bar-row">
        <div class="bar-label">${MODEL_LABELS[name]}${star}</div>
        <div class="bar-track"><div class="bar-fill ${cls}" style="width:${pct}%"></div></div>
        <div class="bar-value">R²=${m.r2}</div>
      </div>`;
    })
    .join("");

  const rows = entries
    .map(([name, m]) => {
      const cls = name === data.best_model ? "best-row" : "";
      return `<tr class="${cls}"><td>${MODEL_LABELS[name]}</td><td>${m.r2}</td><td>$${m.mae_usd}</td></tr>`;
    })
    .join("");

  document.getElementById("leaderboard-box").innerHTML = `
    <p class="hint">${data.n_train} train / ${data.n_test} test listings. Fusion beats both single modalities: <strong>${data.fusion_beats_both_singles ? "yes" : "no"}</strong>.</p>
    ${bars}
    <table style="margin-top:14px;">
      <thead><tr><th>Model</th><th>R²</th><th>MAE</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

document.getElementById("predict-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const payload = {
    bedrooms: parseInt(document.getElementById("bedrooms").value, 10),
    bathrooms: parseInt(document.getElementById("bathrooms").value, 10),
    accommodates: parseInt(document.getElementById("accommodates").value, 10),
    room_type: document.getElementById("room_type").value,
    neighborhood: document.getElementById("neighborhood").value,
    description: document.getElementById("description").value,
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
  document.getElementById("predict-result").innerHTML = `
    <div class="result-grid">
      <div class="result-card">
        <div class="r-label">Tabular only</div>
        <div class="r-value">$${data.tabular_only_prediction}</div>
      </div>
      <div class="result-card">
        <div class="r-label">Text only</div>
        <div class="r-value">$${data.text_only_prediction}</div>
      </div>
      <div class="result-card fusion">
        <div class="r-label">Fusion (both)</div>
        <div class="r-value">$${data.fusion_prediction}</div>
      </div>
    </div>`;
});

loadLeaderboard();
