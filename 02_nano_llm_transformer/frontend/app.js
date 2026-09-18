const API_URL = "http://localhost:8002";

const promptEl = document.getElementById("prompt");
const outputEl = document.getElementById("output");
const statusEl = document.getElementById("status");
const sendBtn = document.getElementById("send");

async function generate() {
  const prompt = promptEl.value.trim();
  if (!prompt) return;

  sendBtn.disabled = true;
  statusEl.textContent = "generating...";
  outputEl.textContent = "";

  try {
    const res = await fetch(`${API_URL}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, max_new_tokens: 100, temperature: 0.7, mode: "chat" }),
    });
    if (!res.ok) throw new Error(`Server returned ${res.status}`);
    const data = await res.json();
    outputEl.textContent = data.text || "(empty response)";
    statusEl.textContent = `checkpoint: ${data.checkpoint}`;
  } catch (err) {
    outputEl.textContent = `Error: ${err.message}. Is the backend running on ${API_URL}?`;
    statusEl.textContent = "";
  } finally {
    sendBtn.disabled = false;
  }
}

sendBtn.addEventListener("click", generate);
promptEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    generate();
  }
});
