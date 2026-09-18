let TOPICS = [];
let activeTopic = null;
const renderedPanels = {};

async function init() {
  const res = await fetch("/api/topics");
  TOPICS = await res.json();
  const nav = document.getElementById("tab-nav");
  nav.innerHTML = TOPICS.map((t, i) =>
    `<button class="tab-btn ${i === 0 ? "active" : ""}" data-id="${t.id}">${t.title}</button>`
  ).join("");
  nav.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.id));
  });
  switchTab(TOPICS[0].id);
}

function switchTab(topicId) {
  activeTopic = topicId;
  document.querySelectorAll(".tab-btn").forEach((b) => b.classList.toggle("active", b.dataset.id === topicId));
  const container = document.getElementById("topic-container");
  container.innerHTML = "";
  if (topicId === "naive_bayes") renderNaiveBayes(container);
  else if (topicId === "model_evaluation") renderModelEvaluation(container);
  else if (topicId === "calculus_gradient_descent") renderCalculus(container);
  else if (topicId === "chain_rule_backprop") renderBackprop(container);
}

// ---------------------------------------------------------------------
// Shared: quiz + interview prep (used by every topic)
// ---------------------------------------------------------------------
async function renderQuizSection(topicId, container) {
  const panel = document.createElement("section");
  panel.className = "panel";
  panel.innerHTML = `<h2>Quiz</h2><div class="quiz-body">Loading…</div>`;
  container.appendChild(panel);

  const res = await fetch(`/api/quiz/${topicId}`);
  const questions = await res.json();
  const body = panel.querySelector(".quiz-body");
  body.innerHTML = questions
    .map(
      (q) => `<div class="quiz-question" data-qid="${q.id}">
        <div class="prompt">${q.prompt}</div>
        ${q.options.map((opt, i) => `<label><input type="radio" name="${q.id}" value="${i}"> ${opt}</label>`).join("")}
        <div class="quiz-feedback" style="display:none;"></div>
      </div>`
    )
    .join("") + `<button class="run-btn" id="quiz-submit">Submit quiz</button><div class="quiz-score" id="quiz-score"></div>`;

  body.querySelector("#quiz-submit").addEventListener("click", async () => {
    const answers = {};
    questions.forEach((q) => {
      const sel = body.querySelector(`input[name="${q.id}"]:checked`);
      if (sel) answers[q.id] = parseInt(sel.value, 10);
    });
    const gradeRes = await fetch(`/api/quiz/${topicId}/grade`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ answers }),
    });
    const result = await gradeRes.json();
    result.results.forEach((r) => {
      const qDiv = body.querySelector(`.quiz-question[data-qid="${r.id}"]`);
      const fb = qDiv.querySelector(".quiz-feedback");
      fb.style.display = "block";
      fb.className = "quiz-feedback " + (r.correct ? "correct" : "incorrect");
      fb.textContent = (r.correct ? "✅ Correct. " : "❌ Not quite. ") + r.explanation;
    });
    body.querySelector("#quiz-score").textContent = `Score: ${result.score} / ${result.total}`;
  });
}

async function renderInterviewSection(topicId, container) {
  const panel = document.createElement("section");
  panel.className = "panel";
  panel.innerHTML = `<h2>Interview prep</h2><div class="interview-body">Loading…</div>`;
  container.appendChild(panel);

  const res = await fetch(`/api/interview-questions/${topicId}`);
  const questions = await res.json();
  panel.querySelector(".interview-body").innerHTML = questions
    .map(
      (q) => `<details class="interview-q">
        <summary>${q.question}</summary>
        <div class="answer">${q.model_answer}</div>
      </details>`
    )
    .join("");
}

// ---------------------------------------------------------------------
// Topic 1: Naive Bayes
// ---------------------------------------------------------------------
async function renderNaiveBayes(container) {
  const panel = document.createElement("section");
  panel.className = "panel";
  panel.innerHTML = `
    <h2>Naive Bayes</h2>
    <p class="intuition">Naive Bayes classifies text by multiplying together each word's likelihood ratio between classes
    (it's "naive" because it assumes words are independent given the class — untrue in reality, but a useful approximation).
    Type a message below and see exactly which words pushed the live classifier toward spam or ham.</p>
    <form id="nb-form">
      <label>Message
        <textarea id="nb-text" rows="2" maxlength="500">win a free prize click now</textarea>
      </label>
      <button type="submit" class="run-btn">Classify live</button>
    </form>
    <div class="result-box" id="nb-result"></div>
  `;
  container.appendChild(panel);

  panel.querySelector("#nb-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = panel.querySelector("#nb-text").value;
    const res = await fetch("/api/naive-bayes/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    const box = panel.querySelector("#nb-result");
    const badgeCls = data.predicted_label === "spam" ? "spam" : "ham";
    const words = data.word_contributions
      .map((w) => {
        const cls = w.spam_lean_score > 0 ? "spam" : "ham";
        const bg = w.spam_lean_score > 0 ? "#fdecea" : "#eaf7ec";
        return `<span class="word-chip" style="background:${bg}">${w.word} (${w.spam_lean_score > 0 ? "+" : ""}${w.spam_lean_score})</span>`;
      })
      .join(" ");
    box.innerHTML = `
      <p><span class="badge ${badgeCls}">${data.predicted_label.toUpperCase()}</span>
      P(spam)=${(data.prob_spam * 100).toFixed(1)}% · P(ham)=${(data.prob_ham * 100).toFixed(1)}%</p>
      <p class="hint">Positive score = pushes toward spam, negative = pushes toward ham (log-likelihood ratio per word):</p>
      <p>${words || "<em>No known words found — try a message closer to the training examples below.</em>"}</p>
    `;
  });
  panel.querySelector("#nb-form").dispatchEvent(new Event("submit"));

  const exRes = await fetch("/api/naive-bayes/examples");
  const examples = await exRes.json();
  const exPanel = document.createElement("section");
  exPanel.className = "panel";
  exPanel.innerHTML = `<h3>Training examples</h3>
    <p class="hint"><strong>Spam:</strong> ${examples.spam.slice(0, 5).join(" · ")}</p>
    <p class="hint"><strong>Ham:</strong> ${examples.ham.slice(0, 5).join(" · ")}</p>`;
  container.appendChild(exPanel);

  await renderQuizSection("naive_bayes", container);
  await renderInterviewSection("naive_bayes", container);
}

// ---------------------------------------------------------------------
// Topic 2: Model Evaluation
// ---------------------------------------------------------------------
async function renderModelEvaluation(container) {
  const panel = document.createElement("section");
  panel.className = "panel";
  panel.innerHTML = `
    <h2>Model Evaluation</h2>
    <p class="intuition">Every classifier outputs a probability; the THRESHOLD you pick to call something "positive" is a choice,
    not a fact — and it trades precision against recall. Drag the slider and watch the confusion matrix, precision, and recall
    all recompute live from the same 150 held-out probability scores.</p>
    <form id="eval-form">
      <label>Decision threshold: <span id="thresh-val">0.50</span>
        <input type="range" id="eval-threshold" min="0" max="1" step="0.01" value="0.5" />
      </label>
      <label>Cost of a false negative (Type II), relative to false positive (Type I)
        <input type="number" id="cost-fn" value="1" min="0" max="20" step="0.5" />
      </label>
    </form>
    <div class="result-box" id="eval-result"></div>
    <canvas id="roc-canvas" width="420" height="320"></canvas>
  `;
  container.appendChild(panel);

  const rocRes = await fetch("/api/eval/roc-curve");
  const roc = await rocRes.json();
  drawRocCurve(panel.querySelector("#roc-canvas"), roc);

  async function refresh() {
    const threshold = parseFloat(panel.querySelector("#eval-threshold").value);
    const costFn = parseFloat(panel.querySelector("#cost-fn").value);
    panel.querySelector("#thresh-val").textContent = threshold.toFixed(2);
    const res = await fetch("/api/eval/confusion-matrix", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ threshold, cost_fp: 1.0, cost_fn: costFn }),
    });
    const m = await res.json();
    panel.querySelector("#eval-result").innerHTML = `
      <table>
        <tr><th></th><th>Predicted positive</th><th>Predicted negative</th></tr>
        <tr><th>Actual positive</th><td>${m.confusion_matrix.true_positive} (TP)</td><td>${m.confusion_matrix.false_negative_type2} (FN, Type II)</td></tr>
        <tr><th>Actual negative</th><td>${m.confusion_matrix.false_positive_type1} (FP, Type I)</td><td>${m.confusion_matrix.true_negative} (TN)</td></tr>
      </table>
      <p class="hint">Precision ${(m.precision * 100).toFixed(1)}% · Recall ${(m.recall * 100).toFixed(1)}% · F1 ${(m.f1 * 100).toFixed(1)}% · Accuracy ${(m.accuracy * 100).toFixed(1)}% · ROC-AUC ${roc.auc} · Weighted cost ${m.total_cost}</p>
    `;
  }
  panel.querySelector("#eval-threshold").addEventListener("input", refresh);
  panel.querySelector("#cost-fn").addEventListener("input", refresh);
  await refresh();

  await renderQuizSection("model_evaluation", container);
  await renderInterviewSection("model_evaluation", container);
}

function drawRocCurve(canvas, roc) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height, pad = 40;
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = "#ccc";
  ctx.beginPath();
  ctx.moveTo(pad, H - pad); ctx.lineTo(W - pad, H - pad); // x-axis
  ctx.moveTo(pad, H - pad); ctx.lineTo(pad, pad);           // y-axis
  ctx.stroke();
  // diagonal (random classifier reference)
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = "#aaa";
  ctx.beginPath();
  ctx.moveTo(pad, H - pad); ctx.lineTo(W - pad, pad);
  ctx.stroke();
  ctx.setLineDash([]);
  // ROC curve
  ctx.strokeStyle = "#16213e";
  ctx.lineWidth = 2;
  ctx.beginPath();
  roc.fpr.forEach((x, i) => {
    const px = pad + x * (W - 2 * pad);
    const py = H - pad - roc.tpr[i] * (H - 2 * pad);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  });
  ctx.stroke();
  ctx.fillStyle = "#333";
  ctx.font = "11px sans-serif";
  ctx.fillText("False Positive Rate →", pad, H - 10);
  ctx.save();
  ctx.translate(12, H - pad);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("True Positive Rate →", 0, 0);
  ctx.restore();
  ctx.fillText(`AUC = ${roc.auc}`, W - pad - 70, pad + 14);
}

// ---------------------------------------------------------------------
// Topic 3: Calculus & Gradient Descent
// ---------------------------------------------------------------------
async function renderCalculus(container) {
  const panel = document.createElement("section");
  panel.className = "panel";
  const fnRes = await fetch("/api/calculus/functions");
  const functions = await fnRes.json();

  panel.innerHTML = `
    <h2>Calculus &amp; Gradient Descent</h2>
    <p class="intuition">The derivative <code>f'(x)</code> is the slope of <code>f</code> at a point. Gradient descent repeatedly
    steps <code>x</code> in the OPPOSITE direction of that slope: <code>x_new = x_old - learning_rate × f'(x_old)</code> — walking
    downhill toward a minimum. Try a learning rate that's too large and watch it diverge instead.</p>
    <form id="gd-form">
      <label>Function
        <select id="gd-fn">${functions.map((f) => `<option value="${f.id}">${f.label}</option>`).join("")}</select>
      </label>
      <label>Starting x <input type="number" id="gd-start" value="-5" step="0.5" /></label>
      <label>Learning rate <input type="number" id="gd-lr" value="0.1" step="0.05" min="0.01" max="3" /></label>
      <button type="submit" class="run-btn">Run gradient descent</button>
    </form>
    <div class="result-box" id="gd-result"></div>
    <canvas id="gd-canvas" width="420" height="320"></canvas>
  `;
  container.appendChild(panel);

  panel.querySelector("#gd-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const function_id = panel.querySelector("#gd-fn").value;
    const start_x = parseFloat(panel.querySelector("#gd-start").value);
    const learning_rate = parseFloat(panel.querySelector("#gd-lr").value);
    const res = await fetch("/api/calculus/gradient-descent", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ function_id, start_x, learning_rate, n_steps: 40 }),
    });
    const data = await res.json();
    panel.querySelector("#gd-result").innerHTML = `
      <p class="hint">${data.diverged ? "⚠️ Diverged — learning rate too large for this function." :
        data.converged ? `✅ Converged near x = ${data.final_x}` : `Still moving after ${data.trajectory.length - 1} steps — final x = ${data.final_x}`}</p>
    `;
    drawGradientDescent(panel.querySelector("#gd-canvas"), data, functions.find((f) => f.id === function_id));
  });
  panel.querySelector("#gd-form").dispatchEvent(new Event("submit"));

  await renderQuizSection("calculus_gradient_descent", container);
  await renderInterviewSection("calculus_gradient_descent", container);
}

function drawGradientDescent(canvas, data, fnMeta) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height, pad = 40;
  ctx.clearRect(0, 0, W, H);

  const xs = data.trajectory.map((t) => t.x);
  const ys = data.trajectory.map((t) => t.f_x);
  const xMin = Math.min(...xs) - 1, xMax = Math.max(...xs) + 1;
  const yMin = Math.min(...ys) - 1, yMax = Math.max(...ys) + 1;
  const sx = (x) => pad + ((x - xMin) / (xMax - xMin || 1)) * (W - 2 * pad);
  const sy = (y) => H - pad - ((y - yMin) / (yMax - yMin || 1)) * (H - 2 * pad);

  ctx.strokeStyle = "#ccc";
  ctx.beginPath();
  ctx.moveTo(pad, H - pad); ctx.lineTo(W - pad, H - pad);
  ctx.moveTo(pad, H - pad); ctx.lineTo(pad, pad);
  ctx.stroke();

  // path of descent
  ctx.strokeStyle = "#4d9fff";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  data.trajectory.forEach((t, i) => {
    const px = sx(t.x), py = sy(t.f_x);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  });
  ctx.stroke();

  data.trajectory.forEach((t, i) => {
    ctx.fillStyle = i === data.trajectory.length - 1 ? "#d32f2f" : "#16213e";
    ctx.beginPath();
    ctx.arc(sx(t.x), sy(t.f_x), i === data.trajectory.length - 1 ? 5 : 2.5, 0, 2 * Math.PI);
    ctx.fill();
  });

  ctx.fillStyle = "#333";
  ctx.font = "11px sans-serif";
  ctx.fillText("x", W - pad, H - pad + 14);
  ctx.fillText("f(x)", pad - 25, pad - 8);
  ctx.fillText(fnMeta ? fnMeta.label : "", pad, 16);
}

// ---------------------------------------------------------------------
// Topic 4: Chain Rule & Backprop
// ---------------------------------------------------------------------
async function renderBackprop(container) {
  const panel = document.createElement("section");
  panel.className = "panel";
  panel.innerHTML = `
    <h2>Chain Rule &amp; Backpropagation</h2>
    <p class="intuition">A tiny network: 2 inputs → 1 hidden neuron (sigmoid) → 1 output (sigmoid). The chain rule lets us compute
    how much each weight contributed to the final loss by multiplying local derivatives along the path from the loss back to that
    weight. This IS backpropagation — a deep network just chains more of these same steps together.</p>
    <form id="bp-form">
      <label>x1 <input type="number" id="bp-x1" value="1.0" step="0.1" /></label>
      <label>x2 <input type="number" id="bp-x2" value="0.5" step="0.1" /></label>
      <label>w1 <input type="number" id="bp-w1" value="0.6" step="0.1" /></label>
      <label>w2 <input type="number" id="bp-w2" value="-0.3" step="0.1" /></label>
      <label>w3 <input type="number" id="bp-w3" value="0.9" step="0.1" /></label>
      <label>target (0 or 1) <input type="number" id="bp-target" value="1" step="1" min="0" max="1" /></label>
      <button type="submit" class="run-btn">Run forward + backward pass</button>
    </form>
    <div class="result-box" id="bp-result"></div>
  `;
  container.appendChild(panel);

  panel.querySelector("#bp-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const payload = {
      x1: parseFloat(panel.querySelector("#bp-x1").value),
      x2: parseFloat(panel.querySelector("#bp-x2").value),
      w1: parseFloat(panel.querySelector("#bp-w1").value),
      w2: parseFloat(panel.querySelector("#bp-w2").value),
      w3: parseFloat(panel.querySelector("#bp-w3").value),
      target: parseFloat(panel.querySelector("#bp-target").value),
    };
    const [fwdRes, checkRes] = await Promise.all([
      fetch("/api/backprop/forward-backward", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }),
      fetch("/api/backprop/gradient-check", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }),
    ]);
    const fwd = await fwdRes.json();
    const check = await checkRes.json();

    const fwdRows = Object.entries(fwd.forward).map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
    const chainRows = Object.entries(fwd.backward_chain).map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
    const gradRows = Object.entries(fwd.gradients).map(([k, v]) =>
      `<tr><td>${k}</td><td>${v}</td><td>${check.numerical[k]}</td></tr>`
    ).join("");

    panel.querySelector("#bp-result").innerHTML = `
      <h3>Forward pass</h3>
      <table>${fwdRows}</table>
      <h3>Backward pass — chain rule, term by term</h3>
      <table>${chainRows}</table>
      <h3>Final gradients: analytic (chain rule) vs. numerical (finite difference) check</h3>
      <table><tr><th></th><th>Analytic</th><th>Numerical check</th></tr>${gradRows}</table>
      <p class="hint">Max difference between analytic and numerical: ${check.max_abs_difference} — this is how you'd verify a hand-implemented backward pass is correct.</p>
    `;
  });
  panel.querySelector("#bp-form").dispatchEvent(new Event("submit"));

  await renderQuizSection("chain_rule_backprop", container);
  await renderInterviewSection("chain_rule_backprop", container);
}

init();
