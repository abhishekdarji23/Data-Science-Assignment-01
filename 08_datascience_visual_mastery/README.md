# 📐 Data Science Visual Foundations

A small, real teaching tool covering 4 core data science / ML concepts —
**Naive Bayes**, **model evaluation** (confusion matrix, type I/II
errors, ROC-AUC, cost matrix, precision/recall tradeoff), **differential
calculus & gradient descent**, and **the chain rule & backpropagation** —
each with a genuine live simulation (not a canned animation), a
server-graded quiz, and interview-prep material.

Reproduced (scoped down, "not fancy," simple frontend + backend) from
Project 08 of
[`dlmastery/data_science_examples`](https://github.com/dlmastery/data_science_examples).
See [`PROMPTS.md`](PROMPTS.md) for exactly what was asked — this is the
one project in this series whose 4-topic core scope was reproduced
essentially as specified, with the main adjustment being *how* it's
served (see below).

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate     # optional but recommended
pip install -r requirements.txt

# Generate the two small curated datasets (Naive Bayes + eval demo)
python data/generate_data.py

# Run the app (no separate training step — every simulation runs live)
uvicorn app:app --reload --port 8008
```

Open **http://127.0.0.1:8008** — 4 tabs, one per topic. Each has a live
simulation, a quiz (graded server-side), and interview-prep questions.

Run the tests any time with:
```bash
python -m pytest tests/
```

## What's in the box

```
08_datascience_visual_mastery/
├── README.md                  ← you are here
├── PROMPTS.md                 ← what was asked, and how scope was adjusted
├── IMPLEMENTATION_PLANS.md    ← architecture & technical spec
├── WALKTHROUGH.md             ← code-ordered deep-dive tour (text, see note below)
├── SKILLS.md                  ← practices applied, with file/line pointers
├── AUDIT_REPORT.md            ← mathematical correctness audit, with commands to re-run it yourself
├── requirements.txt
├── data/
│   └── generate_data.py       ← 2 small curated datasets (Naive Bayes + eval demo)
├── src/
│   ├── naive_bayes.py         ← Topic 1: real MultinomialNB + word-level explanation
│   ├── eval_metrics.py        ← Topic 2: confusion matrix, ROC-AUC, cost matrix
│   ├── calculus.py             ← Topic 3: hand-derived derivatives + gradient descent
│   ├── backprop.py            ← Topic 4: hand chain rule + numerical gradient check
│   └── content.py             ← quiz questions (server-graded) + interview prep, all 4 topics
├── app.py                     ← FastAPI backend (CRISP-DM: Deployment)
├── static/                    ← plain HTML/CSS/JS frontend (4-tab layout, Canvas 2D plots)
└── tests/
    └── test_api.py            ← automated tests (13/13 passing, across all 4 topics)
```

## The 4 topics

1. **Naive Bayes** — a real classifier trained on 30 labeled messages;
   type any message and see exactly which words pushed it toward spam
   or ham, and by how much (the literal log-likelihood-ratio the
   algorithm computes).
2. **Model Evaluation** — drag a threshold slider and watch the
   confusion matrix, precision, recall, and a cost-weighted score all
   recompute live from the same 150 held-out scores; a plotted ROC curve
   with AUC.
3. **Calculus & Gradient Descent** — pick a function and a learning
   rate, watch gradient descent walk downhill step by step on a live
   canvas plot — including watching it diverge if the learning rate is
   too large.
4. **Chain Rule & Backpropagation** — run a forward and backward pass
   through a tiny 2-input, 1-hidden-neuron network by hand, see every
   intermediate chain-rule term, and see the analytic gradient checked
   against a numerical (finite-difference) gradient live.

## Honest limitations / deliberate adjustments

- **Served via FastAPI, not as a static github.io page** — every
  simulation here is a real computation (a trained classifier, live
  metric recomputation, actual gradient descent, hand-verified backprop).
  A static site has no server to run that on; porting all 4 to JavaScript
  would double the surface area to keep correct. See `PROMPTS.md` for the
  full reasoning.
- **Datasets are small and hand-curated on purpose** — a teaching tool
  benefits from a learner being able to see and reason about every row,
  not just aggregate statistics over thousands.
- **No video walkthrough is included.** Recording narration and uploading
  to YouTube isn't something this assistant can do. `WALKTHROUGH.md` is a
  written, code-ordered substitute, with a shot list at the bottom if you
  want to record one yourself in a few minutes.
- **This zip wasn't pushed to GitHub for you** — there's no GitHub write
  access available here. To publish it yourself:
  ```bash
  cd 08_datascience_visual_mastery
  git init && git add . && git commit -m "Data science visual foundations"
  git branch -M main
  git remote add origin https://github.com/<you>/<repo>.git
  git push -u origin main
  ```
- Deliberately **simple**, per your instruction: no React build, no
  charting library (plain Canvas 2D), no user accounts — see
  `IMPLEMENTATION_PLANS.md` §9 for the full list of what was left out and why.

## License

MIT — do whatever you like with it.
