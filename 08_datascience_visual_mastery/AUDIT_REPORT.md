# AUDIT_REPORT.md — Mathematical Correctness Audit

**Scope**: Project 08, Data Science Visual Foundations.
**Method**: Static inspection + scripted checks run against this exact codebase (commands shown below so you can re-run them yourself).

This is a teaching tool making explicit mathematical claims (derivatives,
gradients, monotonic tradeoffs), so this audit's job is to verify those
claims are actually true of the code's output, not just plausible-looking.

## 1. Backprop's analytic (chain rule) gradients match numerical gradients

**Check**: `src/backprop.py: numerical_gradient_check()` should agree
with the hand-derived analytic gradients to several decimal places,
across many different random inputs (not just one cherry-picked example).

```
$ python3 -c "... 50 random trials of numerical_gradient_check ..."
50 random trials, worst max_abs_difference: 5e-06
mean max_abs_difference: 3.84e-06
```

**Result: PASS.** The chain-rule derivation in `forward_backward()`'s
docstring is not just correct-looking — it produces gradients that agree
with independent finite-difference computation to ~5-6 decimal places
across 50 random weight/input combinations, which is the standard bar
for "this backward pass is implemented correctly."

## 2. Gradient descent genuinely converges and genuinely diverges (not scripted)

**Check**: with a reasonable learning rate, `gradient_descent()` should
land near the function's true analytic minimum; with too large a
learning rate, it should genuinely diverge (not just report a flag).

```
$ python3 -c "... gradient_descent('quadratic_bowl', start_x=-5, learning_rate=0.1, n_steps=40) ..."
Converges (lr=0.1, 40 steps): True final_x= 2.99894 (true min = 3.0)

$ python3 -c "... gradient_descent('quadratic_bowl', start_x=-5, learning_rate=1.5, n_steps=30) ..."
Diverges (lr=1.5): True num steps recorded: 18 last x: 1048579.0
```

**Result: PASS.** `final_x = 2.99894` is within 0.002 of the true
analytic minimum (x=3, from `f(x)=(x-3)²+2`'s derivative `f'(x)=2(x-3)=0`
at x=3). The divergence case shows `x` genuinely growing past 1,000,000
within 18 steps — a real numerical blow-up, not a hardcoded "diverged"
flag.

## 3. Precision is NOT strictly monotonic in threshold — and the code correctly reflects that

**Check**: a common misconception is that raising the classification
threshold always increases precision. This audit checked that claim
directly across a full threshold sweep, rather than just two widely-
spaced points.

```
$ python3 -c "... confusion_at_threshold at 9 thresholds from 0.1 to 0.9 ..."
thresholds: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
precisions: [0.4255, 0.4878, 0.6186, 0.7532, 0.8261, 0.9259, 0.9756, 0.963, 1.0]
recalls:    [1.0, 1.0, 1.0, 0.9667, 0.95, 0.8333, 0.6667, 0.4333, 0.1167]
recall non-increasing across full sweep: True
precision non-decreasing across full sweep: False (dips slightly from 0.9756 at t=0.7 to 0.963 at t=0.8)
```

**Result: this is mathematically EXPECTED, not a bug.** Recall
(`TP/(TP+FN)`) is provably non-increasing as the threshold rises: raising
the threshold can only remove points from the predicted-positive set,
which can only decrease TP and increase FN. Precision (`TP/(TP+FP)`) has
no such guarantee — removing a true positive between two threshold
values (while removing few or no false positives) can make precision
DIP locally, exactly as observed between t=0.7 and t=0.8 here. This is a
genuinely useful, correct nuance for a model-evaluation teaching tool to
get right rather than oversimplify. `tests/test_api.py`'s
`test_confusion_matrix_precision_recall_tradeoff` test compares only two
widely-separated thresholds (0.2 vs. 0.8), where the overall upward trend
holds — it does not (and should not) claim strict monotonicity at every
step.

## 4. Naive Bayes word-contribution signs are directionally correct

**Check**: a word that appears heavily in spam training examples and
never in ham should get a positive (spam-leaning) contribution score,
and vice versa.

From the live classifier: "now", "free", and "click" (common spam
training words) scored `+2.087`, `+1.8`, `+1.8` respectively; "can",
"we", and "for" (common ham training words) scored `-1.091`, `-1.091`,
`-0.685`. **Result: PASS** — signs and relative magnitudes match the
training data's actual word distribution.

## 5. Quiz answer keys never reach the client before grading

**Check**: `GET /api/quiz/{topic_id}` must not include `correct_index` or
`explanation` for any question.

**Result: PASS** (by inspection and by `tests/test_api.py:
test_every_topic_has_a_gradeable_quiz`, which explicitly asserts
`"correct_index" not in q` for every question returned by the GET
endpoint, for all 4 topics).

## 6. Known limitation (disclosed, not hidden)

The Naive Bayes and model-evaluation datasets are small and
hand-curated by design (see `PROMPTS.md`/`IMPLEMENTATION_PLANS.md` for
why — a teaching tool benefits from inspectable data). This means the
Naive Bayes classifier's vocabulary is limited to the ~30 training
messages' words; an unusual message may fall back on very few or no
known words (surfaced explicitly via the `unknown_words` field in the
API response, not silently ignored).

## Summary

| Check | Result |
|---|---|
| Backprop analytic gradients match numerical gradients | PASS (50/50 trials, max diff 5e-06) |
| Gradient descent genuinely converges (reasonable lr) | PASS (within 0.002 of true minimum) |
| Gradient descent genuinely diverges (large lr) | PASS (real numerical blow-up) |
| Recall is non-increasing across a full threshold sweep | PASS (provable property, confirmed) |
| Precision is NOT falsely claimed to be strictly monotonic | PASS (correct nuance, not oversimplified) |
| Naive Bayes word-contribution signs match training data | PASS |
| Quiz answers withheld until grading | PASS (all 4 topics) |
| Data realism | Disclosed limitation — small, curated by design |

**Overall**: every mathematical claim this teaching tool makes was
independently checked against either a closed-form analytic answer, a
numerical verification, or a provable property — not just eyeballed.
