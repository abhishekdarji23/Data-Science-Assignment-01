# 00 — Ledger (Dynamic Task Workspace)

A full-stack, reactive task workspace: a task ledger on the left, and a live
telemetry feed on the right that streams every change (created / updated /
completed / deleted) to anyone watching, over Server-Sent Events (SSE) — no
polling, no refresh.

This is a reproduction of `00_dynamic_todo_workspace` from the source
portfolio, built independently with a coding assistant rather than copied,
with a handful of intentional changes noted below.

## Stack

- **Backend:** Node.js + Express, SSE for live telemetry, JSON-file persistence
- **Frontend:** React + Vite, no UI framework/library beyond React itself

## Running it

```bash
# Terminal 1 — backend (port 5000)
cd server
npm install
npm run dev

# Terminal 2 — frontend (port 5173)
cd client
npm install
npm run dev
# open http://localhost:5173
```

Open the app in two browser tabs side by side — add or complete a task in
one tab and watch it (and the telemetry log) update instantly in the other.

## What it does

- Add tasks with a priority (`low` / `normal` / `urgent`), a tag, and an
  optional due date
- Mark tasks done, delete tasks, filter by all / open / done
- A live "watching" counter shows how many browser tabs are currently
  connected to the SSE stream
- Every mutation is broadcast as an SSE event and appears in the telemetry
  panel with a relative timestamp, so the panel doubles as an audit trail
  of the ledger's activity

## Changes made from the original

- **Richer task schema** — priority, tag, and due date, rather than a bare
  title/done flag
- **File-backed persistence** (`server/tasks.json`) instead of pure
  in-memory storage, so the ledger survives a server restart
- **Telemetry panel doubles as an activity log**, not just a live/connected
  indicator — each SSE event renders as a human-readable line ("'Write
  README' marked done") with a relative timestamp
- **Visual identity**: a ledger/paper-trail aesthetic (hairline dividers,
  Fraunces + IBM Plex Sans, a cool slate palette) rather than a card-based
  dashboard look

## Notes

- `server/tasks.json` is created automatically on first run and is
  git-ignored — delete it to reset the ledger to empty.
- The Vite dev server proxies `/api/*` to `localhost:5000`, so both must be
  running for the app to work.
