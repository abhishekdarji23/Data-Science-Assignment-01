let cart = new Set();
let allItems = [];

function renderItemGrid() {
  const grid = document.getElementById("item-grid");
  grid.innerHTML = "";
  allItems.forEach((item) => {
    const chip = document.createElement("div");
    chip.className = "item-chip" + (cart.has(item) ? " selected" : "");
    chip.textContent = item.replace(/_/g, " ");
    chip.onclick = () => {
      if (cart.has(item)) cart.delete(item);
      else cart.add(item);
      renderItemGrid();
      updateCartAndRecommendations();
    };
    grid.appendChild(chip);
  });
}

async function loadItems() {
  const res = await fetch("/api/items");
  if (!res.ok) {
    document.getElementById("item-grid").textContent = "Rules not mined yet — run `python src/train.py`.";
    return;
  }
  allItems = await res.json();
  renderItemGrid();
}

async function updateCartAndRecommendations() {
  const cartBox = document.getElementById("cart-box");
  const recBox = document.getElementById("rec-box");

  if (cart.size === 0) {
    cartBox.textContent = "Cart is empty.";
    recBox.innerHTML = "";
    return;
  }
  cartBox.innerHTML = `<strong>Cart:</strong> ${[...cart].map((i) => i.replace(/_/g, " ")).join(", ")}`;

  const res = await fetch("/api/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ items: [...cart] }),
  });
  if (!res.ok) {
    recBox.textContent = "Could not fetch recommendations.";
    return;
  }
  const data = await res.json();
  if (data.recommendations.length === 0) {
    recBox.innerHTML = `<p class="hint">No strong rules match this cart yet — try adding more items.</p>`;
    return;
  }
  recBox.innerHTML = `
    <p class="hint">${data.matched_rule_count} rule(s) matched. Top recommendations:</p>
    <div class="rec-grid">
      ${data.recommendations
        .map(
          (r) => `<div class="rec-card">
            <div class="item-name">${r.item.replace(/_/g, " ")}</div>
            <div class="why">because you have: ${r.because_of.map((x) => x.replace(/_/g, " ")).join(", ")}</div>
            <div class="stats">lift ${r.lift} · confidence ${(r.confidence * 100).toFixed(0)}%</div>
          </div>`
        )
        .join("")}
    </div>`;
}

async function loadRules() {
  const res = await fetch("/api/rules?limit=15");
  if (!res.ok) {
    document.getElementById("rules-box").textContent = "Rules not mined yet — run `python src/train.py`.";
    return;
  }
  const rules = await res.json();
  const rows = rules
    .map(
      (r) => `<tr>
        <td>${r.antecedents.join(", ").replace(/_/g, " ")}</td>
        <td>→</td>
        <td>${r.consequents.join(", ").replace(/_/g, " ")}</td>
        <td>${r.support}</td>
        <td>${(r.confidence * 100).toFixed(0)}%</td>
        <td>${r.lift}</td>
      </tr>`
    )
    .join("");
  document.getElementById("rules-box").innerHTML = `
    <table>
      <thead><tr><th>If cart has</th><th></th><th>Then recommend</th><th>Support</th><th>Confidence</th><th>Lift</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

async function loadMetrics() {
  const res = await fetch("/api/metrics");
  if (!res.ok) {
    document.getElementById("metrics-box").textContent = "Rules not mined yet — run `python src/train.py`.";
    return;
  }
  const m = await res.json();
  const rows = [
    ["Algorithm", m.algorithm],
    ["Transactions", m.n_transactions],
    ["Unique items", m.n_unique_items],
    ["Avg basket size", m.avg_basket_size],
    ["Frequent itemsets found", m.n_frequent_itemsets],
    ["Rules kept", m.n_rules],
    ["Min support / confidence / lift", `${m.min_support} / ${m.min_confidence} / ${m.min_lift}`],
  ]
    .map(([k, v]) => `<tr><th>${k}</th><td>${v}</td></tr>`)
    .join("");
  document.getElementById("metrics-box").innerHTML = `<table>${rows}</table>`;
}

loadItems();
loadRules();
loadMetrics();
