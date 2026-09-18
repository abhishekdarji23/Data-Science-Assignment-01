import { useEffect, useMemo, useRef, useState } from "react";

const PRIORITIES = ["low", "normal", "urgent"];
const FILTERS = ["all", "open", "done"];

function timeAgo(iso) {
  const diff = Math.max(0, Date.now() - new Date(iso).getTime());
  const s = Math.floor(diff / 1000);
  if (s < 5) return "just now";
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  return `${h}h ago`;
}

export default function App() {
  const [tasks, setTasks] = useState([]);
  const [log, setLog] = useState([]);
  const [connected, setConnected] = useState(false);
  const [presence, setPresence] = useState(1);
  const [filter, setFilter] = useState("all");
  const [title, setTitle] = useState("");
  const [priority, setPriority] = useState("normal");
  const [tag, setTag] = useState("general");
  const [dueDate, setDueDate] = useState("");
  const sourceRef = useRef(null);

  useEffect(() => {
    fetch("/api/tasks")
      .then((r) => r.json())
      .then(setTasks)
      .catch(() => {});

    const es = new EventSource("/api/events");
    sourceRef.current = es;

    es.onopen = () => setConnected(true);
    es.onerror = () => setConnected(false);

    const pushLog = (kind, text) =>
      setLog((prev) => [{ id: crypto.randomUUID(), kind, text, at: new Date().toISOString() }, ...prev].slice(0, 30));

    es.addEventListener("hello", (e) => {
      const { connected } = JSON.parse(e.data);
      setPresence(connected);
    });
    es.addEventListener("presence", (e) => {
      const { connected } = JSON.parse(e.data);
      setPresence(connected);
    });
    es.addEventListener("task:created", (e) => {
      const task = JSON.parse(e.data);
      setTasks((prev) => [task, ...prev.filter((t) => t.id !== task.id)]);
      pushLog("created", `“${task.title}” added to the ledger`);
    });
    es.addEventListener("task:updated", (e) => {
      const task = JSON.parse(e.data);
      setTasks((prev) => prev.map((t) => (t.id === task.id ? task : t)));
      pushLog("updated", `“${task.title}” edited`);
    });
    es.addEventListener("task:completed", (e) => {
      const task = JSON.parse(e.data);
      setTasks((prev) => prev.map((t) => (t.id === task.id ? task : t)));
      pushLog("completed", `“${task.title}” marked done`);
    });
    es.addEventListener("task:deleted", (e) => {
      const { id, title } = JSON.parse(e.data);
      setTasks((prev) => prev.filter((t) => t.id !== id));
      pushLog("deleted", `“${title}” removed`);
    });

    return () => es.close();
  }, []);

  const visible = useMemo(() => {
    if (filter === "all") return tasks;
    return tasks.filter((t) => t.status === filter);
  }, [tasks, filter]);

  const stats = useMemo(
    () => ({
      open: tasks.filter((t) => t.status === "open").length,
      done: tasks.filter((t) => t.status === "done").length,
      urgent: tasks.filter((t) => t.priority === "urgent" && t.status === "open").length,
    }),
    [tasks]
  );

  async function addTask(e) {
    e.preventDefault();
    if (!title.trim()) return;
    await fetch("/api/tasks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, priority, tag, dueDate: dueDate || null }),
    });
    setTitle("");
    setDueDate("");
  }

  async function toggleDone(task) {
    await fetch(`/api/tasks/${task.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status: task.status === "done" ? "open" : "done" }),
    });
  }

  async function removeTask(id) {
    await fetch(`/api/tasks/${id}`, { method: "DELETE" });
  }

  return (
    <div className="workspace">
      <header className="workspace__header">
        <div>
          <p className="workspace__eyebrow">live task ledger</p>
          <h1>Ledger</h1>
        </div>
        <div className="status">
          <span className={`status__dot ${connected ? "status__dot--live" : ""}`} />
          <span>{connected ? "Connected" : "Reconnecting…"}</span>
          <span className="status__divider">·</span>
          <span>{presence} watching</span>
        </div>
      </header>

      <main className="workspace__body">
        <section className="ledger">
          <form className="composer" onSubmit={addTask}>
            <input
              className="composer__title"
              placeholder="What needs doing?"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
            />
            <select value={priority} onChange={(e) => setPriority(e.target.value)}>
              {PRIORITIES.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
            <input
              className="composer__tag"
              placeholder="tag"
              value={tag}
              onChange={(e) => setTag(e.target.value)}
            />
            <input type="date" value={dueDate} onChange={(e) => setDueDate(e.target.value)} />
            <button type="submit">Add to ledger</button>
          </form>

          <div className="filters">
            {FILTERS.map((f) => (
              <button
                key={f}
                className={`filters__pill ${filter === f ? "filters__pill--active" : ""}`}
                onClick={() => setFilter(f)}
              >
                {f}
              </button>
            ))}
            <span className="filters__stats">
              {stats.open} open · {stats.urgent} urgent · {stats.done} done
            </span>
          </div>

          <ul className="tasks">
            {visible.length === 0 && <li className="tasks__empty">Nothing here yet. Add the first line above.</li>}
            {visible.map((t) => (
              <li key={t.id} className={`task task--${t.priority} ${t.status === "done" ? "task--done" : ""}`}>
                <button
                  className="task__check"
                  aria-label={t.status === "done" ? "Mark open" : "Mark done"}
                  onClick={() => toggleDone(t)}
                >
                  {t.status === "done" ? "✓" : ""}
                </button>
                <div className="task__body">
                  <p className="task__title">{t.title}</p>
                  <p className="task__meta">
                    {t.tag} · {t.priority}
                    {t.dueDate ? ` · due ${t.dueDate}` : ""}
                  </p>
                </div>
                <button className="task__remove" onClick={() => removeTask(t.id)} aria-label="Delete task">
                  ×
                </button>
              </li>
            ))}
          </ul>
        </section>

        <aside className="telemetry">
          <h2>Telemetry</h2>
          <p className="telemetry__hint">Every change here streams to anyone watching this ledger, in real time.</p>
          <ul className="telemetry__log">
            {log.length === 0 && <li className="telemetry__empty">Waiting for activity…</li>}
            {log.map((entry) => (
              <li key={entry.id} className={`telemetry__entry telemetry__entry--${entry.kind}`}>
                <span className="telemetry__dot" />
                <span className="telemetry__text">{entry.text}</span>
                <span className="telemetry__time">{timeAgo(entry.at)}</span>
              </li>
            ))}
          </ul>
        </aside>
      </main>
    </div>
  );
}
