const API_URL = "http://localhost:8010";

const tabsEl = document.getElementById("tabs");
const contentEl = document.getElementById("content");

const PHASES = [
  { id: "phase1", label: "1. Business" },
  { id: "phase2", label: "2. Data Understanding" },
  { id: "phase3", label: "3. Data Prep" },
  { id: "phase4", label: "4. Modeling" },
  { id: "phase5", label: "5. Evaluation" },
  { id: "phase6", label: "6. Deployment" },
  { id: "phase7", label: "7. Monitoring" },
];

let recordsCache = null;

function renderTabs(activeId) {
  tabsEl.innerHTML = "";
  for (const p of PHASES) {
    const btn = document.createElement("button");
    btn.textContent = p.label;
    if (p.id === activeId) btn.classList.add("active");
    btn.addEventListener("click", () => show(p.id));
    tabsEl.appendChild(btn);
  }
}

async function show(id) {
  renderTabs(id);
  contentEl.innerHTML = "<p class='note'>Loading…</p>";

  if (id === "phase6") return renderPhase6();

  const res = await fetch(`${API_URL}/api/${id}`);
  const data = await res.json();
  contentEl.innerHTML = `<h2>${data.phase}</h2><pre>${JSON.stringify(data, null, 2)}</pre>`;
}

async function renderPhase6() {
  if (!recordsCache) {
    const res = await fetch(`${API_URL}/api/records`);
    recordsCache = (await res.json()).records;
  }

  contentEl.innerHTML = `
    <h2>6. Deployment — find similar profiles</h2>
    <p class="note">Pick a record; the LSH index returns its nearest neighbors by cosine similarity on the prepared feature vector.</p>
    <select id="recordSelect"></select>
    <input id="kInput" type="number" value="5" min="1" max="20" style="width: 60px" />
    <button class="action" id="searchBtn">Find similar</button>
    <div id="phase6Result"></div>
  `;

  const select = document.getElementById("recordSelect");
  for (const r of recordsCache) {
    const opt = document.createElement("option");
    opt.value = r.id;
    opt.textContent = `#${r.id} — age ${r.age}, ${r.education}, ${r.occupation}, ${r.hours_per_week}h/wk, ${r.income}`;
    select.appendChild(opt);
  }

  document.getElementById("searchBtn").addEventListener("click", async () => {
    const id = select.value;
    const k = document.getElementById("kInput").value;
    const res = await fetch(`${API_URL}/api/phase6/similar/${id}?k=${k}`);
    const data = await res.json();
    const resultEl = document.getElementById("phase6Result");

    if (!res.ok) {
      resultEl.innerHTML = `<p class="note">Error: ${data.detail}</p>`;
      return;
    }

    resultEl.innerHTML = `
      <p class="note">Scanned ${data.num_candidates_scanned} of ${recordsCache.length} records in ${data.elapsed_ms} ms.</p>
      <table>
        <thead><tr><th>id</th><th>age</th><th>education</th><th>occupation</th><th>hours/wk</th><th>income</th><th>cosine sim</th></tr></thead>
        <tbody>
          ${data.similar
            .map(
              (s) => `<tr>
                <td>${s.record.id}</td><td>${s.record.age}</td><td>${s.record.education}</td>
                <td>${s.record.occupation}</td><td>${s.record.hours_per_week}</td>
                <td>${s.record.income}</td><td>${s.cosine_similarity}</td>
              </tr>`
            )
            .join("")}
        </tbody>
      </table>`;
  });
}

show("phase1");
