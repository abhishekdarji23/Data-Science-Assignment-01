function scoreClass(score) {
  if (score >= 90) return "good";
  if (score >= 60) return "mid";
  return "bad";
}

function scoreHtml(report) {
  const cls = scoreClass(report.compliance_score);
  return `
    <div class="score-display">
      <div class="score-circle ${cls}">${report.compliance_score}%</div>
      <div class="score-meta">
        <strong>${report.variant === "clean" ? "Clean" : "Flawed"} pipeline</strong><br/>
        ${report.checks_passed} / ${report.checks_total} checks passed
      </div>
    </div>`;
}

function pipelineHtml(report) {
  const p = report.pipeline_summary;
  return `
    <table>
      <tr><th>Rows (raw)</th><td>${p.n_rows_raw}</td></tr>
      <tr><th>Rows used for training</th><td>${p.n_rows_used_for_training}</td></tr>
      <tr><th>Feature columns</th><td>${p.feature_columns.join(", ")}</td></tr>
      <tr><th>Split method</th><td>${p.split_method}</td></tr>
      <tr><th>Model MAE</th><td>${p.model_mae_seconds}s (baseline: ${p.baseline_mae_seconds}s)</td></tr>
      <tr><th>Model R&sup2;</th><td>${p.model_r2}</td></tr>
    </table>`;
}

function checksByPhaseHtml(report) {
  return Object.entries(report.checks_by_phase)
    .map(
      ([phase, checks]) => `
      <div class="phase-group">
        <div class="phase-title">${phase}</div>
        ${checks
          .map(
            (c) => `<div class="check-card ${c.passed ? "pass" : "fail"}">
              <span class="check-title">${c.title}</span>
              <span class="check-badge ${c.passed ? "pass" : "fail"}">${c.passed ? "PASS" : "FAIL"}</span>
              <div class="check-detail">${c.detail}</div>
            </div>`
          )
          .join("")}
      </div>`
    )
    .join("");
}

function showPanels(...ids) {
  ["score-panel", "pipeline-panel", "checks-panel", "compare-panel"].forEach((id) => {
    document.getElementById(id).style.display = ids.includes(id) ? "block" : "none";
  });
}

async function runAudit(variant) {
  const res = await fetch(`/api/audit/run?variant=${variant}`, { method: "POST" });
  const report = await res.json();

  document.getElementById("score-box").innerHTML = scoreHtml(report);
  document.getElementById("pipeline-box").innerHTML = pipelineHtml(report);
  document.getElementById("checks-box").innerHTML = checksByPhaseHtml(report);
  showPanels("score-panel", "pipeline-panel", "checks-panel");
}

async function runCompare() {
  const res = await fetch("/api/audit/compare", { method: "POST" });
  const data = await res.json();

  document.getElementById("compare-box").innerHTML = `
    <div class="compare-columns">
      <div>
        <h3>Clean pipeline</h3>
        ${scoreHtml(data.clean)}
        ${checksByPhaseHtml(data.clean)}
      </div>
      <div>
        <h3>Flawed pipeline</h3>
        ${scoreHtml(data.flawed)}
        ${checksByPhaseHtml(data.flawed)}
      </div>
    </div>`;
  showPanels("compare-panel");
}

document.getElementById("run-clean-btn").addEventListener("click", () => runAudit("clean"));
document.getElementById("run-flawed-btn").addEventListener("click", () => runAudit("flawed"));
document.getElementById("run-compare-btn").addEventListener("click", runCompare);
