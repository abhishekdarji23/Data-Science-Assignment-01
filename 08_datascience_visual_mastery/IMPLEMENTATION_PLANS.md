# IMPLEMENTATION_PLANS.md — Project 08: Data Science Visual Foundations

## 1. Objective

Teach 4 core data science / ML concepts — Naive Bayes, model evaluation,
differential calculus & gradient descent, and the chain rule &
backpropagation — through a REAL, live-executable simulation of each
(not a canned animation), plus a server-graded quiz and interview-prep
material per topic.

## 2. Architecture

```
┌────────────────────┐   GET /api/topics                     ┌──────────────────────┐
│  Browser (4-tab      │ ─────────────────────────────────▶ │   FastAPI app.py     │
│  teaching page)       │ ◀───────────────────────────────── │   (uvicorn)          │
└────────────────────┘   POST /api/{topic}/... (per-topic)     └──────────┬───────────┘
                                                                           │ calls
                          ┌────────────────────────────────────────────────┼────────────────────────────────┐
                          │                                                │                                │
                ┌─────────▼──────────┐               ┌─────────▼──────────┐            ┌──────────▼──────────┐
                │ src/naive_bayes.py │               │ src/eval_metrics.py │            │ src/calculus.py       │
                │ (real MultinomialNB)│               │ (confusion/ROC)     │            │ (analytic derivatives)│
                └─────────────────────┘               └─────────────────────┘            └────────────────────────┘
                                                                                            ┌──────────────────────┐
                                                                                            │ src/backprop.py       │
                                                                                            │ (hand chain rule)     │
                                                                                            └──────────────────────┘
                          All 4 topics also share:
                          ┌────────────────────────────────────────────────────────────────┐
                          │ src/content.py — quiz questions (graded server-side) +           │
                          │                  interview-prep material, for every topic         │
                          └────────────────────────────────────────────────────────────────┘
```

Unlike the other projects in this series, there's no single "train once,
persist a model" step — every simulation IS the live computation, run
fresh on every request (a Naive Bayes fit at import time from a tiny
fixed dataset, a confusion matrix recomputed from a fixed probability
set, gradient descent/backprop computed from user-supplied numbers
directly). This matches the pedagogical goal: the user should be able to
change an input and immediately see genuine, freshly-computed output.

## 3. Data

Two small, hand-curated datasets (`data/generate_data.py`), appropriate
for a teaching tool where a learner benefits from being able to see and
reason about every row:

- `spam_ham_synthetic.csv` — 30 short labeled messages (15 spam, 15 ham)
  for the Naive Bayes demo.
- `eval_test_set_synthetic.csv` — 150 (true_label, predicted_probability)
  pairs, constructed so positives skew high and negatives skew low WITH
  deliberate overlap, so moving the decision threshold visibly trades
  precision against recall.

## 4. Topic-by-topic implementation

### Topic 1 — Naive Bayes (`src/naive_bayes.py`)
A real `sklearn.naive_bayes.MultinomialNB` fit on the spam/ham dataset.
`classify(text)` returns not just the predicted label but each word's
`log P(word|spam) - log P(word|ham)` — the literal per-word quantity
Naive Bayes sums (in log-space) to reach its decision, so the
"explanation" IS how the algorithm works, not a separate add-on.

### Topic 2 — Model Evaluation (`src/eval_metrics.py`)
`confusion_at_threshold(threshold, cost_fp, cost_fn)` recomputes the full
confusion matrix, precision, recall, F1, accuracy, and a cost-weighted
total from the SAME 150 fixed probability scores every time — this is
the live simulation: the threshold slider changes nothing about the
data, only the decision rule applied to it. `roc_curve_points()` returns
the full ROC curve + AUC via `sklearn.metrics.roc_curve`/`roc_auc_score`.

### Topic 3 — Calculus & Gradient Descent (`src/calculus.py`)
3 named functions, each with a HAND-DERIVED analytic derivative (no
autograd) — e.g. `f(x) = (x-3)² + 2` → `f'(x) = 2(x-3)`.
`gradient_descent()` runs the literal update rule
`x_new = x_old - learning_rate * f'(x_old)` step by step, including a
deliberately-included failure mode: a too-large learning rate causes
genuine divergence (verified in `AUDIT_REPORT.md` §2), not a scripted
animation of one.

### Topic 4 — Chain Rule & Backprop (`src/backprop.py`)
The smallest network with a genuine hidden layer: 2 inputs → 1 sigmoid
hidden neuron → 1 sigmoid output. `forward_backward()` computes the
forward pass AND the backward pass by hand, term by term, exactly
following the chain rule (see the function's docstring for the full
derivation). `numerical_gradient_check()` verifies every analytic
gradient against a finite-difference numerical gradient — this is the
standard way anyone implementing backprop by hand checks their work, and
it's included as a live "trust but verify" demo, not just asserted
correct (verified to ~5-6 decimal places across random inputs — see
`AUDIT_REPORT.md` §1).

## 5. Quizzes & interview prep (`src/content.py`)

3-4 multiple-choice questions per topic. `GET /api/quiz/{topic_id}`
returns prompts and options WITHOUT `correct_index` or `explanation` —
those are only revealed by `POST /api/quiz/{topic_id}/grade` after
submission, so the answer key never ships in the page source. 2-3
interview-prep questions per topic with model answers are served
directly (not graded — they're study material, not a quiz).

## 6. API contract (`app.py`)

| endpoint | method | purpose |
|---|---|---|
| `/` | GET | serves the frontend |
| `/api/health` | GET | liveness check |
| `/api/topics` | GET | the 4 topics' metadata |
| `/api/naive-bayes/examples` | GET | training examples for display |
| `/api/naive-bayes/classify` | POST | `{text}` → prediction + word-level explanation |
| `/api/eval/dataset-summary` | GET | dataset size/class balance |
| `/api/eval/confusion-matrix` | POST | `{threshold, cost_fp, cost_fn}` → full confusion matrix + metrics |
| `/api/eval/roc-curve` | GET | ROC curve points + AUC |
| `/api/calculus/functions` | GET | available functions for gradient descent |
| `/api/calculus/gradient-descent` | POST | `{function_id, start_x, learning_rate, n_steps}` → step-by-step trajectory |
| `/api/backprop/forward-backward` | POST | `{x1,x2,w1,w2,w3,target}` → forward values + chain-rule gradients |
| `/api/backprop/gradient-check` | POST | same inputs → analytic vs. numerical gradient comparison |
| `/api/quiz/{topic_id}` | GET | quiz questions, no answers |
| `/api/quiz/{topic_id}/grade` | POST | `{answers: {qid: index}}` → score + per-question correctness + explanations |
| `/api/interview-questions/{topic_id}` | GET | interview prep Q&A |

## 7. Frontend

Plain HTML/CSS/JS (no build step, no charting library — the ROC curve
and gradient-descent trajectory are drawn with raw HTML5 Canvas 2D calls,
which is enough for 2 simple line plots). A 4-tab layout, one tab per
topic; each tab lazily renders its simulation, quiz, and interview-prep
sections on first click.

## 8. Verification performed

- `python -c "..."` sanity checks for each topic individually: obvious
  spam/ham messages classify correctly with sensible word contributions;
  the precision/recall tradeoff moves in the expected direction as
  threshold changes; gradient descent converges near the true minimum at
  a reasonable learning rate and genuinely diverges at a too-large one;
  backprop's analytic gradients matched numerical (finite-difference)
  gradients to ~5-6 decimal places across 20 random trials
- `uvicorn app:app` + `curl` against every endpoint across all 4 topics,
  plus the quiz grading flow and a 404 check for an unknown topic — all
  returned expected shapes and correct values
- `python -m pytest tests/` — 13/13 automated tests pass, including a
  monotonicity test (raising the threshold never decreases precision or
  increases recall) and a loop that validates every topic's quiz and
  interview questions exist and are gradeable

## 9. Explicit non-goals (kept out of scope per "not fancy, simple frontend and backend")

- No actual github.io static-site deployment (see `PROMPTS.md` for why — the simulations need a real backend)
- No charting library (Chart.js etc.) — plain Canvas 2D line plots instead
- No user accounts / quiz-progress persistence across sessions
- No additional topics beyond the 4 specified
- No containerization/CI — this is a local, single-process demo
