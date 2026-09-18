# 09 — FlowForge (DAG Execution Engine)

A small workflow engine: define tasks and their dependencies as a graph,
validate it, and run it — each task moves through
`pending → running → success` (or `failed` / `skipped` if something upstream
failed). Kept deliberately simple: plain Node/Express backend, plain
HTML/JS frontend, no build step, no framework.

This is a reproduction of `09_flowforge_dag_engine` from the source
portfolio. The original is described there as a TypeScript/Kahn's-algorithm
DAG engine — I kept **Kahn's algorithm** (that's the interesting part: it
detects cycles and groups tasks into parallel-safe "levels" in one pass),
but wrote the backend in plain JavaScript rather than TypeScript, since the
goal here was a small, obviously-correct implementation rather than a type-
system showcase.

## Stack

- **Backend:** Node.js + Express, in-memory graph state, no database
- **Frontend:** one `index.html` + one `app.js`, polling the backend every
  ~400ms while a run is in progress — no WebSockets/SSE needed at this scale

## Running it

```bash
# Terminal 1 — backend (port 8009)
cd server
npm install
npm run dev

# Terminal 2 — frontend
cd client
python3 -m http.server 5182
# open http://localhost:5182
```

## How it works

- **`server/dag.js`** — the actual engine, in ~90 lines:
  - `topologicalLevels(nodes)` runs Kahn's algorithm: repeatedly pull out
    every node with no unresolved dependencies, put them in the same
    "level", and remove them from the graph. If nodes remain that never
    reach zero dependencies, there's a cycle — and the error message says
    exactly which nodes are stuck in it.
  - `runGraph(nodes, { onUpdate })` uses those levels to simulate execution:
    everything in a level "runs" concurrently (via `setTimeout` standing in
    for real work), and a node marked `shouldFail: true` in the graph
    definition causes everything downstream of it to be marked `skipped`
    rather than run.
- The frontend lets you edit the graph as JSON directly (id, label,
  dependsOn, duration, optional shouldFail), validate it to see the
  computed levels or a cycle error, then run it and watch the status table
  update live.

## Try breaking it

The default graph is a small two-branch pipeline (a data branch and a
lint/build branch that both feed into `deploy`). Worth trying in the UI:

- Add a task whose `dependsOn` points back at something that (transitively)
  depends on it → **Validate** reports the cycle instead of hanging.
- Set `"shouldFail": true` on `train_model` and re-run → `evaluate_model`
  and `deploy` come back as `skipped`, not `pending` or `success`.
