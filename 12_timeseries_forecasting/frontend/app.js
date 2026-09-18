const API_URL = "http://localhost:8012";

const tabsEl = document.getElementById("tabs");
const contentEl = document.getElementById("content");

const TABS = [
  { id: "forecast", label: "Series & Forecast" },
  { id: "acfpacf", label: "ACF / PACF" },
  { id: "backtest", label: "Backtest" },
];

// ---------- tiny SVG chart helpers (no library) ----------

function scaleFns(values, width, height, padding = 30) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const n = values.length;
  const x = (i) => padding + (i / Math.max(n - 1, 1)) * (width - 2 * padding);
  const y = (v) => height - padding - ((v - min) / range) * (height - 2 * padding);
  return { x, y, min, max };
}

function lineChartSVG({ history, forecast, width = 840, height = 320 }) {
  const allValues = [...history.map((p) => p.value), ...forecast.map((p) => p.upper), ...forecast.map((p) => p.lower)];
  const n = history.length + forecast.length;
  const { x, y } = scaleFns(allValues, width, height);

  const histPoints = history.map((p, i) => `${x(i)},${y(p.value)}`).join(" ");
  const fcPoints = forecast.map((p, i) => `${x(history.length + i)},${y(p.forecast)}`).join(" ");

  const bandTop = forecast.map((p, i) => `${x(history.length + i)},${y(p.upper)}`).join(" ");
  const bandBottom = forecast
    .map((p, i) => `${x(history.length + i)},${y(p.lower)}`)
    .reverse()
    .join(" ");

  return `
    <svg viewBox="0 0 ${width} ${height}">
      <polygon points="${bandTop} ${bandBottom}" fill="#3d5a80" opacity="0.15" />
      <polyline points="${histPoints}" fill="none" stroke="#1c2733" stroke-width="1.5" />
      <polyline points="${fcPoints}" fill="none" stroke="#c1440e" stroke-width="1.5" stroke-dasharray="4,3" />
      <line x1="${x(history.length - 1)}" y1="0" x2="${x(history.length - 1)}" y2="${height}" stroke="#999" stroke-dasharray="2,2" />
    </svg>
    <p class="note">Solid line: history · dashed red: forecast · shaded band: 95% interval · vertical line: forecast start</p>
  `;
}

function barChartSVG(values, { width = 840, height = 220, sigBand = null } = {}) {
  const padding = 30;
  const maxAbs = Math.max(...values.map(Math.abs), 0.1);
  const barWidth = (width - 2 * padding) / values.length;
  const zeroY = height / 2;

  const bars = values
    .map((v, i) => {
      const barHeight = (Math.abs(v) / maxAbs) * (height / 2 - padding / 2);
      const x = padding + i * barWidth;
      const y = v >= 0 ? zeroY - barHeight : zeroY;
      return `<rect x="${x}" y="${y}" width="${barWidth * 0.7}" height="${barHeight}" fill="${v >= 0 ? '#3d5a80' : '#c1440e'}" />`;
    })
    .join("");

  let sigLines = "";
  if (sigBand) {
    const upperY = zeroY - (sigBand / maxAbs) * (height / 2 - padding / 2);
    const lowerY = zeroY + (sigBand / maxAbs) * (height / 2 - padding / 2);
    sigLines = `
      <line x1="${padding}" y1="${upperY}" x2="${width - padding}" y2="${upperY}" stroke="#999" stroke-dasharray="3,3" />
      <line x1="${padding}" y1="${lowerY}" x2="${width - padding}" y2="${lowerY}" stroke="#999" stroke-dasharray="3,3" />
    `;
  }

  return `<svg viewBox="0 0 ${width} ${height}"><line x1="${padding}" y1="${zeroY}" x2="${width - padding}" y2="${zeroY}" stroke="#333" />${sigLines}${bars}</svg>`;
}

function overlaySVG(seriesA, seriesB, { width = 840, height = 320, labelA = "actual", labelB = "predicted" } = {}) {
  const { x, y } = scaleFns([...seriesA, ...seriesB], width, height);
  const pointsA = seriesA.map((v, i) => `${x(i)},${y(v)}`).join(" ");
  const pointsB = seriesB.map((v, i) => `${x(i)},${y(v)}`).join(" ");
  return `
    <svg viewBox="0 0 ${width} ${height}">
      <polyline points="${pointsA}" fill="none" stroke="#1c2733" stroke-width="1.5" />
      <polyline points="${pointsB}" fill="none" stroke="#c1440e" stroke-width="1.5" stroke-dasharray="4,3" />
    </svg>
    <p class="note">Solid: ${labelA} · dashed red: ${labelB}</p>
  `;
}

// ---------- tab content ----------

function renderTabs(activeId) {
  tabsEl.innerHTML = "";
  for (const t of TABS) {
    const btn = document.createElement("button");
    btn.textContent = t.label;
    if (t.id === activeId) btn.classList.add("active");
    btn.addEventListener("click", () => show(t.id));
    tabsEl.appendChild(btn);
  }
}

async function show(id) {
  renderTabs(id);
  contentEl.innerHTML = "<p class='note'>Loading…</p>";
  if (id === "forecast") return renderForecast();
  if (id === "acfpacf") return renderAcfPacf();
  if (id === "backtest") return renderBacktest();
}

async function renderForecast() {
  const [seriesRes, forecastRes] = await Promise.all([
    fetch(`${API_URL}/api/series`).then((r) => r.json()),
    fetch(`${API_URL}/api/forecast?horizon=30`).then((r) => r.json()),
  ]);
  const recentHistory = seriesRes.series.slice(-90); // last 90 days for readability
  const chart = lineChartSVG({ history: recentHistory, forecast: forecastRes.predictions });

  contentEl.innerHTML = `
    <h2>Last 90 days + 30-day forecast fan</h2>
    ${chart}
    <div style="margin-top: 12px;">
      <span class="stat">Trend slope: <b>${forecastRes.slope_per_day}</b>/day</span>
      <span class="stat">Residual std: <b>${forecastRes.residual_std}</b></span>
    </div>
    <p class="note">Seasonal effect by weekday: ${JSON.stringify(forecastRes.seasonal_by_weekday)}</p>
  `;
}

async function renderAcfPacf() {
  const [acfRes, pacfRes] = await Promise.all([
    fetch(`${API_URL}/api/acf?max_lag=40`).then((r) => r.json()),
    fetch(`${API_URL}/api/pacf?max_lag=40`).then((r) => r.json()),
  ]);
  const n = 730;
  const sigBand = 1.96 / Math.sqrt(n);

  contentEl.innerHTML = `
    <h2>Autocorrelation (ACF), lags 1-40</h2>
    <p class="note">A spike every 7 lags is the weekly seasonality showing up in the correlation structure.</p>
    ${barChartSVG(acfRes.values.slice(1), { sigBand })}
    <h2>Partial autocorrelation (PACF), lags 1-40</h2>
    <p class="note">PACF strips out the indirect correlation carried through intermediate lags — dashed lines mark the ~95% significance band.</p>
    ${barChartSVG(pacfRes.values.slice(1), { sigBand })}
  `;
}

async function renderBacktest() {
  const data = await fetch(`${API_URL}/api/backtest?holdout=60`).then((r) => r.json());
  const actualValues = data.actual.map((a) => a.value);
  const predictedValues = data.predicted.map((p) => p.forecast);

  const chart = overlaySVG(actualValues, predictedValues);

  contentEl.innerHTML = `
    <h2>Backtest: last ${data.holdout_days} days held out, forecast vs actual</h2>
    ${chart}
    <div style="margin-top: 12px;">
      <span class="stat">MAE: <b>${data.mae}</b></span>
      <span class="stat">RMSE: <b>${data.rmse}</b></span>
    </div>
    <p class="note">Model was trained only on data before this window, then forecast forward to compare against what actually happened.</p>
  `;
}

show("forecast");
