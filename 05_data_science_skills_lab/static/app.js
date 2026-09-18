const PHASE_ORDER = ["Data Understanding", "Data Preparation", "Modeling", "Evaluation"];

async function loadDatasetSummary() {
  const res = await fetch("/api/dataset/summary");
  if (!res.ok) return;
  const d = await res.json();
  document.getElementById("dataset-subtitle").textContent =
    `Working dataset: ${d.n_rows} rows × ${d.n_columns} columns. Click "Execute" on any skill to run it live.`;
}

async function loadSkills() {
  const res = await fetch("/api/skills");
  const container = document.getElementById("phases-container");
  if (!res.ok) {
    container.innerHTML = `<p class="hint">Could not load skills — is the backend running?</p>`;
    return;
  }
  const skills = await res.json();

  container.innerHTML = "";
  PHASE_ORDER.forEach((phase) => {
    const phaseSkills = skills.filter((s) => s.crisp_dm_phase === phase);
    if (phaseSkills.length === 0) return;

    const section = document.createElement("section");
    section.className = "phase-section";
    section.innerHTML = `<div class="phase-title">${phase}</div><div class="skill-grid" id="grid-${phase.replace(/\s/g, "")}"></div>`;
    container.appendChild(section);

    const grid = section.querySelector(".skill-grid");
    phaseSkills.forEach((skill) => {
      const card = document.createElement("div");
      card.className = "skill-card";
      card.innerHTML = `
        <h3>${skill.title}</h3>
        <div class="desc">${skill.description}</div>
        <button class="exec-btn" data-skill="${skill.id}">Execute skill</button>
        <div class="result-box" id="result-${skill.id}"></div>
      `;
      grid.appendChild(card);
    });
  });

  document.querySelectorAll(".exec-btn").forEach((btn) => {
    btn.addEventListener("click", () => executeSkill(btn.dataset.skill, btn));
  });
}

async function executeSkill(skillId, btn) {
  btn.disabled = true;
  btn.textContent = "Running…";
  const resultBox = document.getElementById(`result-${skillId}`);
  resultBox.innerHTML = "";

  try {
    const res = await fetch(`/api/skills/${skillId}/execute`, { method: "POST" });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      resultBox.innerHTML = `<p class="hint">Error: ${err.detail || res.statusText}</p>`;
      return;
    }
    const result = await res.json();
    renderResult(resultBox, result);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run again";
  }
}

function renderResult(container, result) {
  let html = `<div class="result-summary">${result.summary}</div>`;

  if (result.display_type === "metrics") {
    html += `<div class="metric-grid">${result.data
      .map((m) => `<div class="metric-cell"><div class="m-label">${m.label}</div><div class="m-value">${m.value}</div></div>`)
      .join("")}</div>`;
  } else if (result.display_type === "table") {
    const { columns, rows } = result.data;
    html += `<table class="result-table"><thead><tr>${columns
      .map((c) => `<th>${c}</th>`)
      .join("")}</tr></thead><tbody>${rows
      .map((r) => `<tr>${r.map((cell) => `<td>${cell}</td>`).join("")}</tr>`)
      .join("")}</tbody></table>`;
  } else if (result.display_type === "bar_chart") {
    const { labels, values } = result.data;
    const max = Math.max(...values, 0.0001);
    html += labels
      .map((label, i) => {
        const pct = Math.max(2, (values[i] / max) * 100);
        return `<div class="bar-row">
          <div class="bar-label">${label}</div>
          <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
          <div class="bar-value">${values[i]}</div>
        </div>`;
      })
      .join("");
  } else if (result.display_type === "list") {
    html += `<ul class="result-list">${result.data.map((item) => `<li>${item}</li>`).join("")}</ul>`;
  }

  container.innerHTML = html;
}

loadDatasetSummary();
loadSkills();
