const NYC_CENTER = [40.75, -73.98];
const map = L.map("map").setView(NYC_CENTER, 12);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);

let pickup = null;
let dropoff = null;
let pickupMarker = null;
let dropoffMarker = null;
let routeLine = null;

function resetPoints() {
  pickup = null;
  dropoff = null;
  if (pickupMarker) map.removeLayer(pickupMarker);
  if (dropoffMarker) map.removeLayer(dropoffMarker);
  if (routeLine) map.removeLayer(routeLine);
  pickupMarker = dropoffMarker = routeLine = null;
}

map.on("click", (e) => {
  const { lat, lng } = e.latlng;
  if (!pickup) {
    pickup = { lat, lng };
    pickupMarker = L.circleMarker([lat, lng], { color: "green", radius: 8 }).addTo(map).bindPopup("Pickup").openPopup();
  } else if (!dropoff) {
    dropoff = { lat, lng };
    dropoffMarker = L.circleMarker([lat, lng], { color: "red", radius: 8 }).addTo(map).bindPopup("Dropoff").openPopup();
    routeLine = L.polyline([[pickup.lat, pickup.lng], [dropoff.lat, dropoff.lng]], { color: "#16213e", dashArray: "6 6" }).addTo(map);
  } else {
    resetPoints();
    pickup = { lat, lng };
    pickupMarker = L.circleMarker([lat, lng], { color: "green", radius: 8 }).addTo(map).bindPopup("Pickup").openPopup();
  }
});

// Default the datetime input to "now" for convenience
document.getElementById("pickup_datetime").value = new Date().toISOString().slice(0, 16);

document.getElementById("trip-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!pickup || !dropoff) {
    alert("Click the map twice: once for pickup, once for dropoff.");
    return;
  }
  const payload = {
    pickup_lat: pickup.lat,
    pickup_lon: pickup.lng,
    dropoff_lat: dropoff.lat,
    dropoff_lon: dropoff.lng,
    passenger_count: parseInt(document.getElementById("passenger_count").value, 10),
    pickup_datetime: document.getElementById("pickup_datetime").value,
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
  document.getElementById("result-panel").style.display = "block";
  document.getElementById("r-duration").textContent = `${data.predicted_duration_minutes} min`;
  document.getElementById("r-distance").textContent = `${data.distance_miles} mi`;
  document.getElementById("r-fare").textContent = `$${data.estimated_fare_usd}`;
  document.getElementById("r-speed").textContent = data.avg_speed_mph ? `${data.avg_speed_mph} mph` : "–";
});

// Load model metrics for the "admin" panel
fetch("/api/metrics")
  .then((r) => r.ok ? r.json() : Promise.reject(r))
  .then((m) => {
    const rows = [
      ["Model", m.model],
      ["Train rows", m.n_train.toLocaleString()],
      ["Test rows", m.n_test.toLocaleString()],
      ["MAE (seconds)", m.mae_seconds],
      ["RMSE (seconds)", m.rmse_seconds],
      ["RMSLE", m.rmsle],
      ["R²", m.r2],
      ["Split strategy", m.split],
    ].map(([k, v]) => `<tr><th>${k}</th><td>${v}</td></tr>`).join("");
    document.getElementById("metrics-box").innerHTML = `<table>${rows}</table>`;
  })
  .catch(() => {
    document.getElementById("metrics-box").textContent = "Model not trained yet — run `python src/train.py`.";
  });
