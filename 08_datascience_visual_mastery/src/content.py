"""
content.py
----------
Quiz questions (graded server-side, so the correct answer is never sent
to the browser until after grading) and interview-prep material for all
4 topics.
"""

TOPICS = [
    {"id": "naive_bayes", "title": "Naive Bayes",
     "tagline": "A live spam classifier, and why it's called 'naive'."},
    {"id": "model_evaluation", "title": "Model Evaluation",
     "tagline": "Confusion matrix, type I/II errors, ROC-AUC, and the precision/recall tradeoff."},
    {"id": "calculus_gradient_descent", "title": "Calculus & Gradient Descent",
     "tagline": "Derivatives, and how they steer a model toward a minimum."},
    {"id": "chain_rule_backprop", "title": "Chain Rule & Backpropagation",
     "tagline": "How a tiny network's weights get their gradients."},
]

# Each question: id, prompt, options (list), correct_index (0-based), explanation
QUIZZES = {
    "naive_bayes": [
        {
            "id": "nb_q1",
            "prompt": "Why is Naive Bayes called 'naive'?",
            "options": [
                "It assumes all features (e.g. words) are conditionally independent given the class",
                "It only works on small datasets",
                "It never uses probability",
                "It always predicts the majority class",
            ],
            "correct_index": 0,
            "explanation": "Naive Bayes multiplies per-feature likelihoods together as if they were independent given the class — a simplifying (and technically often false) assumption, hence 'naive.' It works well in practice anyway for many text classification tasks.",
        },
        {
            "id": "nb_q2",
            "prompt": "In the spam classifier demo, a word that appears ONLY in spam messages in training will have:",
            "options": [
                "No effect on the prediction",
                "A strong positive (spam-leaning) contribution score",
                "A strong negative (ham-leaning) contribution score",
                "An error, since it can't be classified",
            ],
            "correct_index": 1,
            "explanation": "log P(word|spam) - log P(word|ham) is large and positive when a word is common in spam and rare/absent in ham — exactly what the demo's word_contributions score measures.",
        },
        {
            "id": "nb_q3",
            "prompt": "What does Naive Bayes actually compute to make a prediction?",
            "options": [
                "The Euclidean distance to the nearest training example",
                "P(class | features), via Bayes' theorem using P(features | class) and P(class)",
                "A weighted sum of features passed through a sigmoid",
                "The majority vote of many decision trees",
            ],
            "correct_index": 1,
            "explanation": "Naive Bayes applies Bayes' theorem: P(class|features) ∝ P(features|class) × P(class), using the independence assumption to make P(features|class) tractable as a product of per-feature terms.",
        },
    ],
    "model_evaluation": [
        {
            "id": "me_q1",
            "prompt": "A Type I error is:",
            "options": [
                "A false negative (missing a real positive)",
                "A false positive (flagging a negative as positive)",
                "Any incorrect prediction",
                "An error only possible in regression",
            ],
            "correct_index": 1,
            "explanation": "Type I error = false positive. Type II error = false negative. A mnemonic: Type I is the 'boy who cried wolf' (false alarm).",
        },
        {
            "id": "me_q2",
            "prompt": "If you raise the classification threshold (require higher predicted probability to call something positive), what typically happens?",
            "options": [
                "Precision tends to rise, recall tends to fall",
                "Precision tends to fall, recall tends to rise",
                "Both precision and recall always rise",
                "Threshold has no effect on precision or recall",
            ],
            "correct_index": 0,
            "explanation": "A higher threshold means only the most confident predictions get flagged positive — fewer false positives (higher precision), but more true positives get missed too (lower recall). Try this directly in the live simulation above.",
        },
        {
            "id": "me_q3",
            "prompt": "ROC-AUC of 0.5 means:",
            "options": [
                "The model is perfect",
                "The model performs no better than random guessing at ranking positives above negatives",
                "The model has 50% accuracy",
                "The model is overfit",
            ],
            "correct_index": 1,
            "explanation": "ROC-AUC measures ranking quality: the probability a random positive example is scored higher than a random negative one. 0.5 = chance level; 1.0 = perfect ranking.",
        },
        {
            "id": "me_q4",
            "prompt": "Why might you weight false negatives more heavily than false positives in a cost matrix?",
            "options": [
                "False negatives are always more common",
                "In some domains (e.g. medical screening, fraud), missing a real positive is far more costly than a false alarm",
                "It's required by scikit-learn",
                "It always maximizes accuracy",
            ],
            "correct_index": 1,
            "explanation": "A missed cancer diagnosis or missed fraud case is often far costlier than an unnecessary follow-up test or a declined-then-approved transaction — the cost matrix lets the 'best' threshold reflect real-world consequences, not just raw accuracy.",
        },
    ],
    "calculus_gradient_descent": [
        {
            "id": "gd_q1",
            "prompt": "The derivative f'(x) at a point tells you:",
            "options": [
                "The value of the function at that point",
                "The slope (instantaneous rate of change) of the function at that point",
                "The minimum value the function can take",
                "The number of times the function crosses zero",
            ],
            "correct_index": 1,
            "explanation": "f'(x) is the instantaneous slope — how fast f is increasing or decreasing right at x. Gradient descent uses this slope to decide which direction reduces f.",
        },
        {
            "id": "gd_q2",
            "prompt": "In gradient descent, why do we move in the NEGATIVE direction of the gradient?",
            "options": [
                "Negative numbers are easier to compute",
                "The gradient points in the direction of steepest INCREASE, so moving opposite decreases the function",
                "It's an arbitrary convention with no mathematical reason",
                "To avoid negative numbers in the output",
            ],
            "correct_index": 1,
            "explanation": "The gradient points toward steepest ascent. Since we want to MINIMIZE a loss function, we step in the opposite (negative gradient) direction — that's literally what x_new = x_old - learning_rate * f'(x_old) does.",
        },
        {
            "id": "gd_q3",
            "prompt": "What typically happens if the learning rate is too large?",
            "options": [
                "Convergence is faster and more stable",
                "The algorithm may overshoot the minimum and diverge (x oscillates and grows without bound)",
                "The gradient becomes zero immediately",
                "Nothing changes — learning rate has no effect",
            ],
            "correct_index": 1,
            "explanation": "Try it in the live simulation: a learning rate that's too large causes each step to overshoot past the minimum by more than the previous step, so x oscillates with growing magnitude instead of settling down.",
        },
    ],
    "chain_rule_backprop": [
        {
            "id": "bp_q1",
            "prompt": "The chain rule lets you compute the derivative of:",
            "options": [
                "Only linear functions",
                "A composition of functions, by multiplying the derivatives of each step",
                "Only functions with a single input",
                "Only functions without any minimum",
            ],
            "correct_index": 1,
            "explanation": "If y = f(g(x)), the chain rule says dy/dx = dy/dg * dg/dx — the derivative of a composition is the product of the derivatives of each link in the chain. Backpropagation is exactly this, applied through every layer of a network.",
        },
        {
            "id": "bp_q2",
            "prompt": "In the tiny network demo (2 inputs -> hidden neuron -> output), why does dL/dw1 depend on w3?",
            "options": [
                "It doesn't — w1's gradient is independent of w3",
                "Because the chain from the loss back to w1 passes THROUGH the output layer, which includes w3",
                "Because w1 and w3 are always set equal",
                "Only if the learning rate is negative",
            ],
            "correct_index": 1,
            "explanation": "dL/dw1 = dL/dyhat * dyhat/dz2 * dz2/da1 * da1/dz1 * dz1/dw1, and dz2/da1 = w3 — so yes, w1's gradient genuinely depends on w3's current value. This is why backprop must go backward: later-layer weights affect earlier-layer gradients.",
        },
        {
            "id": "bp_q3",
            "prompt": "What is 'backpropagation' actually short for, conceptually?",
            "options": [
                "Backward propagation of errors — applying the chain rule from the output loss back through each layer to get every weight's gradient",
                "A special type of neural network architecture",
                "Running the network backward to generate new training data",
                "A way to compress a trained model",
            ],
            "correct_index": 0,
            "explanation": "Backpropagation IS the chain rule, applied systematically layer by layer from the loss backward to every weight — nothing more mysterious than that, though frameworks like PyTorch automate it (autograd) for arbitrarily deep networks.",
        },
    ],
}

INTERVIEW_QUESTIONS = {
    "naive_bayes": [
        {"question": "Why does Naive Bayes often work well in practice despite its unrealistic independence assumption?",
         "model_answer": "Even when features aren't truly independent, Naive Bayes only needs to get the RANKING of class probabilities right, not the exact probability values — the independence assumption's errors often partially cancel out or don't change which class scores highest, especially for text classification where individual word signals are fairly strong."},
        {"question": "What's the difference between Multinomial and Gaussian Naive Bayes?",
         "model_answer": "Multinomial NB models feature counts (e.g. word frequencies) and is standard for text classification. Gaussian NB assumes each feature is normally distributed within each class and is used for continuous numeric features."},
        {"question": "How does Naive Bayes handle a word in a test message that never appeared in training?",
         "model_answer": "Without smoothing, an unseen word would get zero probability and (via the product) zero-out the entire class probability. Laplace/additive smoothing (sklearn's default) adds a small count to every word for every class so unseen words get a small nonzero probability instead of breaking the calculation."},
    ],
    "model_evaluation": [
        {"question": "When would you prioritize recall over precision?",
         "model_answer": "When missing a positive case is much more costly than a false alarm — e.g. disease screening, fraud detection, or safety-critical anomaly detection, where you'd rather over-flag and have a human review than silently miss a true case."},
        {"question": "Why is accuracy a poor metric for an imbalanced dataset?",
         "model_answer": "A model that always predicts the majority class can achieve high accuracy while being completely useless — e.g. 99% accuracy on a 1%-positive-rate fraud dataset by never flagging anything. PR-AUC, F1, or recall/precision at a chosen threshold are more informative."},
        {"question": "What does the area under the ROC curve actually measure, geometrically and probabilistically?",
         "model_answer": "Geometrically, it's the integral of true positive rate over false positive rate as the threshold sweeps from 1 to 0. Probabilistically, it equals the probability that a randomly chosen positive example is ranked higher (has a higher predicted score) than a randomly chosen negative example."},
    ],
    "calculus_gradient_descent": [
        {"question": "What role does the learning rate play, and what are the risks of setting it too high or too low?",
         "model_answer": "The learning rate scales the step size taken in the direction of steepest descent. Too high causes overshooting and possible divergence (as demonstrated in the simulation); too low causes very slow convergence and risk of getting stuck on a long, flat plateau before reaching the minimum."},
        {"question": "How does gradient descent generalize from a 1D function to training a model with millions of parameters?",
         "model_answer": "The same update rule applies per-parameter: each weight moves opposite to the partial derivative of the loss with respect to that weight. In high dimensions, the 'gradient' is a vector of these partial derivatives, and the update is a vector subtraction — same core idea, more dimensions."},
        {"question": "What's the difference between a local minimum and a global minimum, and why does it matter for gradient descent?",
         "model_answer": "A local minimum is a point lower than all nearby points; a global minimum is the lowest point overall. Gradient descent only guarantees convergence to A minimum in the neighborhood it starts in — for non-convex functions (like most neural network loss surfaces) it may settle into a local minimum rather than the global one, though in practice this is often good enough."},
    ],
    "chain_rule_backprop": [
        {"question": "Why is backpropagation more efficient than computing each weight's gradient independently from scratch?",
         "model_answer": "Backprop reuses intermediate derivative terms computed once during the backward pass (like dL/dz2 in the demo) across multiple weight gradients, rather than recomputing the whole forward pass separately for every single weight — this is what makes training networks with millions of parameters computationally feasible."},
        {"question": "What is the vanishing gradient problem, and how does it relate to the chain rule?",
         "model_answer": "Because backprop multiplies many local derivatives together across layers, if each one is consistently small (e.g. sigmoid's derivative maxes out at 0.25), the product shrinks exponentially with depth — gradients reaching early layers become vanishingly small, so those layers barely update. This is why ReLU and other activations with larger derivative ranges are preferred in deep networks."},
        {"question": "How would you verify that a hand-implemented backward pass is correct?",
         "model_answer": "Numerical gradient checking: perturb each weight by a small epsilon in both directions, compute the resulting change in loss, and compare that finite-difference estimate to the analytic gradient from your backward pass — they should agree to several decimal places, exactly what numerical_gradient_check() does in this demo."},
    ],
}


def get_quiz(topic_id: str) -> list[dict]:
    """Returns quiz questions WITHOUT the correct_index or explanation —
    those are only revealed by grade_quiz() after the user submits."""
    questions = QUIZZES.get(topic_id, [])
    return [{"id": q["id"], "prompt": q["prompt"], "options": q["options"]} for q in questions]


def grade_quiz(topic_id: str, answers: dict[str, int]) -> dict:
    """answers: {question_id: selected_option_index}"""
    questions = QUIZZES.get(topic_id, [])
    results = []
    n_correct = 0
    for q in questions:
        selected = answers.get(q["id"])
        is_correct = selected == q["correct_index"]
        if is_correct:
            n_correct += 1
        results.append({
            "id": q["id"],
            "correct": is_correct,
            "selected_index": selected,
            "correct_index": q["correct_index"],
            "explanation": q["explanation"],
        })
    return {
        "topic_id": topic_id,
        "score": n_correct,
        "total": len(questions),
        "results": results,
    }


def get_interview_questions(topic_id: str) -> list[dict]:
    return INTERVIEW_QUESTIONS.get(topic_id, [])
