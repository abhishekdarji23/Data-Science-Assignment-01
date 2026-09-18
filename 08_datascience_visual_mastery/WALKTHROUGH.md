# WALKTHROUGH.md — Architectural Deep-Dive

> No video walkthrough is included with this build: recording and uploading
> a narrated video (e.g. to YouTube) isn't something this assistant can do —
> there's no video/audio generation or YouTube-upload capability available.
> This document is the substitute: a code-ordered tour you (or a screen
> recorder) can follow to produce one in a few minutes if you want it, plus
> a suggested shot list at the bottom.

## Reading order (this *is* the tour)

1. **`data/generate_data.py`** — the two curated datasets. Read the
   module docstring for why hand-curated beats a large anonymized
   dataset for a teaching tool, then look at `generate_eval_data()`'s
   use of `rng.beta(a=5, b=2, ...)` for positives and `rng.beta(a=2,
   b=5, ...)` for negatives — two distributions that overlap in the
   middle, which is exactly what makes threshold choice matter later.

2. **`src/naive_bayes.py`** — `classify()`. The key line is the
   `log_prob_spam[idx] - log_prob_ham[idx]` computation — read the
   docstring's explanation of why this specific quantity is what Naive
   Bayes itself sums (in log-space) to make its decision, not an
   approximation of it.

3. **`src/eval_metrics.py`** — `confusion_at_threshold()`. Notice it
   takes `cost_fp`/`cost_fn` as parameters — read the docstring for why
   a cost matrix matters (a missed fraud case vs. a false alarm aren't
   equally costly in most real problems).

4. **`src/calculus.py`** — `FUNCTIONS` dict: each entry pairs a function
   with its HAND-derived `f_prime`. Then `gradient_descent()` — the loop
   body is literally `x = x - learning_rate * grad`, nothing more. The
   `diverged`/`converged` checks at the bottom are what let the frontend
   show "this learning rate is too large" as a real, computed outcome.

5. **`src/backprop.py`** — read the big docstring on `forward_backward()`
   first; it lays out the full chain-rule derivation in comments before
   any code. Then read the function itself top to bottom — forward pass,
   then backward pass, each line computing exactly one local derivative
   from the docstring's list. `numerical_gradient_check()` right below
   it is the verification step — read it to see how finite-difference
   checking works (nudge a weight by ±epsilon, see how much the loss
   moves, compare to the analytic gradient).

6. **`src/content.py`** — `QUIZZES` and `INTERVIEW_QUESTIONS` dicts (just
   data), then `get_quiz()` vs. `grade_quiz()` — note `get_quiz()`
   strips `correct_index` and `explanation` before returning, so the
   answer key genuinely never reaches the browser until grading.

7. **`app.py`** — thin FastAPI wiring; every endpoint is a 1-3 line
   pass-through to the `src/` module that does the real work. Worth
   noting: `GradientDescentRequest` and `BackpropRequest` both bound
   their numeric fields (e.g. `learning_rate: le=3.0`) so a wildly
   out-of-range input from the UI can't hang the server.

8. **`static/app.js`** — `init()` loads topics and builds tabs;
   `switchTab()` lazily renders each topic panel on first visit. Each
   `render*()` function (e.g. `renderCalculus`) follows the same shape:
   render the form → wire up its submit handler → fetch + render results
   → call the shared `renderQuizSection()` and `renderInterviewSection()`
   at the end. `drawRocCurve()` and `drawGradientDescent()` are the two
   places doing raw Canvas 2D drawing — both just map data coordinates
   to pixel coordinates and draw lines/points, no library needed.

## How a request flows end-to-end (using gradient descent as the example)

```
user picks a function, start_x, and learning_rate, submits
  → static/app.js POSTs to /api/calculus/gradient-descent
    → app.py: GradientDescentRequest validates bounds
      → calculus.py: gradient_descent() runs x = x - lr*f'(x) in a loop,
        recording every step, until it converges, diverges, or hits n_steps
    ← {trajectory: [...], converged, diverged, final_x}
  ← static/app.js draws the trajectory on the canvas and states the outcome
```

## Suggested shot list (if you want to record your own walkthrough video)

1. `python data/generate_data.py` running, show the two datasets' printed summaries
2. `uvicorn app:app --reload --port 8008` starting up
3. Browser, Naive Bayes tab: type a spammy message, then a normal one, show the word-contribution chips flipping sign
4. Browser, Model Evaluation tab: drag the threshold slider from low to high, narrate precision rising as recall falls, point at the ROC curve and AUC
5. Browser, Calculus tab: run gradient descent with a reasonable learning rate (converges), then a large one (diverges) — show both trajectories on the canvas
6. Browser, Backprop tab: run the forward/backward pass, point out the analytic-vs-numerical gradient table matching closely
7. Any tab: take the quiz, submit, show per-question feedback and explanations appearing
8. Terminal: `python -m pytest tests/` passing
9. Code editor: 30 seconds each on `naive_bayes.py`'s word-contribution line and `backprop.py`'s docstring derivation — the two clearest examples of "rigorous math, not just a claim"
