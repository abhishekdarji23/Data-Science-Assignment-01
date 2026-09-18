# SKILLS.md — Skills Demonstrated / Applied in This Build

Scoped-down equivalent of the source repo's agent-skills catalog. Each item
below is a real, checkable practice applied somewhere in this codebase
(file + line pointer), not just a claim.

| # | Skill | Where applied |
|---|---|---|
| 1 | Explaining an algorithm via the exact quantity it computes | `src/naive_bayes.py: classify()` — the "word contribution" shown to the user IS `log P(word\|spam) - log P(word\|ham)`, the literal per-word term Naive Bayes sums, not a separate post-hoc explainability method |
| 2 | Teaching the precision/recall tradeoff as a live, recomputed fact | `src/eval_metrics.py: confusion_at_threshold()` — every metric recomputes from the same 150 fixed scores as the threshold changes, so the tradeoff is observed, not asserted |
| 3 | Cost-sensitive evaluation | `src/eval_metrics.py: confusion_at_threshold(cost_fp, cost_fn)` — demonstrates why "best" threshold depends on the real-world cost of each error type, not just accuracy |
| 4 | Hand-derived (not autograd) calculus, matched to the algorithm | `src/calculus.py: FUNCTIONS` — every function ships its own analytic `f_prime`, so the derivative shown is exactly what a calculus student would compute by hand |
| 5 | Demonstrating algorithm failure modes on purpose | `src/calculus.py: gradient_descent()` — a too-large learning rate is allowed to genuinely diverge (checked via `diverged`), not just described in prose |
| 6 | Hand-implementing backpropagation via the chain rule | `src/backprop.py: forward_backward()` — every local derivative in the chain (`dyhat/dz2`, `da1/dz1`, etc.) is computed and returned individually, not collapsed into a single opaque gradient |
| 7 | Verifying a hand-implemented gradient (the professional standard) | `src/backprop.py: numerical_gradient_check()` — finite-difference verification of the analytic gradients, matching to ~5-6 decimal places across random trials (see `AUDIT_REPORT.md`) |
| 8 | Never shipping quiz answers to the client before grading | `src/content.py: get_quiz()` strips `correct_index`/`explanation`; only `grade_quiz()` (called after submission) reveals them |
| 9 | Bounded, validated numeric inputs for user-supplied math | `app.py`: every simulation request model bounds its numeric fields (e.g. `learning_rate: le=3.0`) so an extreme input can't hang or crash the server |
| 10 | Automated testing of mathematical properties, not just HTTP status codes | `tests/test_api.py: test_confusion_matrix_precision_recall_tradeoff` — asserts the actual monotonic relationship (higher threshold ⇒ precision↑, recall↓), and `test_gradient_descent_diverges_with_large_learning_rate` — asserts real divergence, not just a 200 response |
| 11 | Documentation-as-artifact | This file, plus `PROMPTS.md`, `IMPLEMENTATION_PLANS.md`, `WALKTHROUGH.md`, `AUDIT_REPORT.md`, `README.md` |

## Explicitly not attempted here (see `IMPLEMENTATION_PLANS.md` §9)

A static github.io-hosted version (the simulations need a real backend —
see `PROMPTS.md`), a charting library for the two plots (plain Canvas 2D
was used instead), user accounts or quiz-progress persistence, and
additional topics beyond the 4 specified were all left out — per your
explicit instruction to keep this "not fancy," with the main aim being a
simple, working frontend and backend.
