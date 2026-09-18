const METHOD_LABELS = {
  naive: "Naive persistence",
  moving_average: "Moving average (5d)",
  exponential_smoothing: "Exponential smoothing",
  arima: "ARIMA",
  gradient_boosting: "Gradient boosting",
};

async function loadLeaderboard() {
  const res = await fetch("/api/leaderboard");
  if (!res.ok) {
    document.getElementById("leaderboard-box").textContent = "Model not trained yet — run `python src/train.py`.";
    return;
  }
  const data = await res.json();
  const entries = Object.entries(data.leaderboard).sort((a, b) => a[1].mae_usd - b[1].mae_usd);

  const rows = entries
    .map(([name, m]) => {
      const cls = name === data.best_method ? "best-row" : "";
      const beats = data.beats_naive[name];
      const badge = name === "naive"
        ? ""
        : `<span class="beats-badge ${beats ? "yes" : "no"}">${beats ? "beats naive" : "does not beat naive"}</span>`;
      return `<tr class="${cls}"><td>${METHOD_LABELS[name]}</td><td>$${m.mae_usd}</td><td>$${m.rmse_usd}</td><td>${m.mape_pct}%</td><td>${badge}</td></tr>`;
    })
    .join("");

  document.getElementById("leaderboard-box").innerHTML = `
    <p class="hint">${data.n_train} train days, ${data.n_test} test days. ARIMA order: (${data.arima_order.join(",")}). Best by MAE: <strong>${METHOD_LABELS[data.best_method]}</strong>.</p>
    <table>
      <thead><tr><th>Method</th><th>MAE</th><th>RMSE</th><th>MAPE</th><th></th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

async function loadChart() {
  const res = await fetch("/api/chart");
  if (!res.ok) return;
  const rows = await res.json();
  drawChart(document.getElementById("chart-canvas"), rows);
}

function drawChart(canvas, rows) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height, pad = 45;
  ctx.clearRect(0, 0, W, H);

  const actual = rows.map((r) => r.actual);
  const naive = rows.map((r) => r.naive_pred);
  const all = actual.concat(naive);
  const yMin = Math.min(...all) * 0.98, yMax = Math.max(...all) * 1.02;
  const sx = (i) => pad + (i / (rows.length - 1)) * (W - 2 * pad);
  const sy = (y) => H - pad - ((y - yMin) / (yMax - yMin)) * (H - 2 * pad);

  ctx.strokeStyle = "#ccc";
  ctx.beginPath();
  ctx.moveTo(pad, H - pad); ctx.lineTo(W - pad, H - pad);
  ctx.moveTo(pad, H - pad); ctx.lineTo(pad, pad);
  ctx.stroke();

  function drawLine(values, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    values.forEach((v, i) => {
      const px = sx(i), py = sy(v);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.stroke();
  }
  drawLine(actual, "#16213e");
  drawLine(naive, "#d32f2f");

  ctx.fillStyle = "#333";
  ctx.font = "11px sans-serif";
  ctx.fillText("● actual", pad, 16);
  ctx.fillStyle = "#d32f2f";
  ctx.fillText("● naive forecast", pad + 70, 16);
  ctx.fillStyle = "#16213e";
}

document.getElementById("forecast-btn").addEventListener("click", async () => {
  const btn = document.getElementById("forecast-btn");
  btn.disabled = true;
  btn.textContent = "Computing…";
  try {
    const res = await fetch("/api/forecast", { method: "POST" });
    const data = await res.json();
    const cards = Object.entries(data.forecasts)
      .map(([name, price]) => {
        const delta = price - data.last_close;
        const dir = delta >= 0 ? "up" : "down";
        return `<div class="forecast-card">
          <div class="f-label">${METHOD_LABELS[name]}</div>
          <div class="f-value">$${price}</div>
          <div class="f-delta ${dir}">${delta >= 0 ? "+" : ""}${delta.toFixed(2)}</div>
        </div>`;
      })
      .join("");
    document.getElementById("forecast-result").innerHTML = `
      <p class="hint">As of ${data.as_of_date}, last close $${data.last_close}. Forecasts for the next trading day:</p>
      <div class="forecast-grid">${cards}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Get tomorrow's forecast from all 5 methods";
  }
});

loadLeaderboard();
loadChart();
