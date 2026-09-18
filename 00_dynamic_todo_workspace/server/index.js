import express from "express";
import cors from "cors";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { randomUUID } from "crypto";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DATA_FILE = path.join(__dirname, "tasks.json");
const PORT = process.env.PORT || 5000;

const app = express();
app.use(cors());
app.use(express.json());

// ---------- persistence ----------
function loadTasks() {
  if (!fs.existsSync(DATA_FILE)) return [];
  try {
    return JSON.parse(fs.readFileSync(DATA_FILE, "utf-8"));
  } catch {
    return [];
  }
}

function saveTasks(tasks) {
  fs.writeFileSync(DATA_FILE, JSON.stringify(tasks, null, 2));
}

let tasks = loadTasks();

// ---------- SSE telemetry ----------
const clients = new Set();

function broadcast(event, payload) {
  const line = `event: ${event}\ndata: ${JSON.stringify(payload)}\n\n`;
  for (const res of clients) res.write(line);
}

app.get("/api/events", (req, res) => {
  res.set({
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
  });
  res.flushHeaders();
  res.write(`event: hello\ndata: ${JSON.stringify({ connected: clients.size + 1 })}\n\n`);

  clients.add(res);
  broadcast("presence", { connected: clients.size });

  req.on("close", () => {
    clients.delete(res);
    broadcast("presence", { connected: clients.size });
  });
});

// ---------- task routes ----------
app.get("/api/tasks", (req, res) => {
  res.json(tasks);
});

app.post("/api/tasks", (req, res) => {
  const { title, priority = "normal", dueDate = null, tag = "general" } = req.body;
  if (!title || !title.trim()) {
    return res.status(400).json({ error: "Title is required." });
  }
  const task = {
    id: randomUUID(),
    title: title.trim(),
    priority,
    dueDate,
    tag,
    status: "open",
    createdAt: new Date().toISOString(),
  };
  tasks = [task, ...tasks];
  saveTasks(tasks);
  broadcast("task:created", task);
  res.status(201).json(task);
});

app.patch("/api/tasks/:id", (req, res) => {
  const idx = tasks.findIndex((t) => t.id === req.params.id);
  if (idx === -1) return res.status(404).json({ error: "Task not found." });

  const before = tasks[idx];
  const updated = { ...before, ...req.body, id: before.id };
  tasks[idx] = updated;
  saveTasks(tasks);

  const event =
    req.body.status === "done" && before.status !== "done"
      ? "task:completed"
      : "task:updated";
  broadcast(event, updated);
  res.json(updated);
});

app.delete("/api/tasks/:id", (req, res) => {
  const task = tasks.find((t) => t.id === req.params.id);
  if (!task) return res.status(404).json({ error: "Task not found." });
  tasks = tasks.filter((t) => t.id !== req.params.id);
  saveTasks(tasks);
  broadcast("task:deleted", { id: task.id, title: task.title });
  res.status(204).end();
});

app.listen(PORT, () => {
  console.log(`Workspace server listening on http://localhost:${PORT}`);
});
